"""Utilities for explainability metrics and data loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Mapping, Optional, Sequence

import h5py
import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    ndcg_score,
)


@dataclass
class ExplainabilityMetrics:
    """Container for aggregated explainability scores."""

    explanation_type: str
    evaluated_class: Optional[int]
    model_mode: str
    num_slides: int
    num_slides_with_instance_labels: int
    num_slides_with_evidence: int
    num_instances_evaluated: int
    instance_macro_f1: Optional[float]
    instance_balanced_accuracy: Optional[float]
    attention_ndcg: Optional[float]
    attention_auprc: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        record: Dict[str, Optional[float]] = {
            "explanation_type": self.explanation_type,
            "evaluated_class": self.evaluated_class,
            "model_mode": self.model_mode,
            "num_slides": float(self.num_slides),
            "num_slides_with_instance_labels": float(self.num_slides_with_instance_labels),
            "num_slides_with_evidence": float(self.num_slides_with_evidence),
            "num_instances_evaluated": float(self.num_instances_evaluated),
            "instance_macro_f1": self.instance_macro_f1,
            "instance_balanced_accuracy": self.instance_balanced_accuracy,
            "attention_ndcg": self.attention_ndcg,
            "attention_auprc": self.attention_auprc,
        }
        return record


def _resolve_data_dir(dataset, slide_row: Mapping[str, object]) -> str:
    data_dir = getattr(dataset, "data_dir", None)
    if isinstance(data_dir, dict):
        source = slide_row.get("source")
        if source is None:
            raise KeyError(
                "Dataset exposes multiple data roots but the slide metadata "
                "does not include a 'source' column to disambiguate entries."
            )
        try:
            resolved = data_dir[source]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Unknown data source {source!r} for slide {slide_row.get('slide_id')}") from exc
        if not isinstance(resolved, str):
            raise TypeError(
                f"Expected data directory for source {source!r} to be a string, got {type(resolved)!r}"
            )
        return resolved
    if not isinstance(data_dir, str):
        raise TypeError(
            "Dataset must expose a string 'data_dir' attribute pointing to the HDF5 root. "
            f"Got {data_dir!r}"
        )
    return data_dir


def iter_explainability_batches(dataset) -> Iterator[Dict[str, object]]:
    """Yield dictionary-style batches with interpretability metadata for each slide.

    Each yielded batch contains:

    - ``slide_id``: identifier of the slide/bag.
    - ``label``: slide-level target as stored in ``dataset.slide_data``.
    - ``features``: NumPy array of patch-level features.
    - ``coords``: NumPy array of coordinates matching the features.
    - ``numbers``: Optional NumPy array with digit identifiers (if available).
    - ``instance_labels``: Optional per-instance targets.
    - ``evidence``: Optional mapping of class indices to relevance scores.
    - ``width`` and ``height``: Canvas size read from ``dataset.shape_dict`` when provided.
    - ``h5_path``: Resolved path to the backing HDF5 file.
    """

    if not hasattr(dataset, "slide_data"):
        raise AttributeError("Dataset must expose a 'slide_data' attribute with slide metadata")

    slide_frame = dataset.slide_data.reset_index(drop=True)
    shape_dict = getattr(dataset, "shape_dict", None)

    for _, row in slide_frame.iterrows():
        slide_id = str(row["slide_id"])
        label = int(row["label"])
        data_root = _resolve_data_dir(dataset, row)
        h5_path = f"{data_root}/h5_files/{slide_id}.h5"

        with h5py.File(h5_path, "r") as handle:
            features = np.asarray(handle["features"], dtype=np.float32)
            coords = np.asarray(handle["coords"], dtype=np.int64)
            numbers = np.asarray(handle["numbers"]) if "numbers" in handle else None
            instance_labels = (
                np.asarray(handle["instance_labels"], dtype=np.int64)
                if "instance_labels" in handle
                else None
            )
            evidence = None
            if "evidence" in handle:
                evidence_group = handle["evidence"]
                evidence = {
                    int(key): np.asarray(evidence_group[key], dtype=np.float32)
                    for key in evidence_group.keys()
                }

        width = height = None
        if isinstance(shape_dict, Mapping) and slide_id in shape_dict:
            width, height = shape_dict[slide_id]

        yield {
            "slide_id": slide_id,
            "label": label,
            "features": features,
            "coords": coords,
            "numbers": numbers,
            "instance_labels": instance_labels,
            "evidence": evidence,
            "width": width,
            "height": height,
            "h5_path": h5_path,
        }


def _normalise_attention(attention: torch.Tensor) -> np.ndarray:
    attention = attention.squeeze(0)
    if attention.ndim != 1:
        raise ValueError(f"Expected 1D attention scores, received tensor of shape {tuple(attention.shape)}")
    values = attention.detach().cpu().numpy().astype(np.float64)
    total = float(values.sum())
    if total > 0:
        values = values / total
    return values


def evaluate_explainability(
    model: torch.nn.Module,
    dataset,
    *,
    model_type: str,
    explanation_type: str = "both",
    evaluated_class: Optional[int] = None,
    model_mode: str = "validation",
    device: Optional[torch.device] = None,
) -> ExplainabilityMetrics:
    """Compute interpretability metrics for a trained model on a dataset.

    Parameters
    ----------
    model:
        Loaded Bayes-MIL model ready for inference.
    dataset:
        Dataset or split exposing ``slide_data`` and HDF5-backed features.
    model_type:
        Model identifier, used to route spatial-variant models that expect
        coordinates and slide dimensions.
    explanation_type:
        Which group of metrics to compute. Accepted values: ``"instance"``,
        ``"attention"``, or ``"both"``.
    evaluated_class:
        Optional class index used when reducing multi-class scores to binary.
        When ``None``, instance metrics fall back to macro-averaging across all
        available classes and attention metrics default to the slide label.
    model_mode:
        Keyword indicating the forward-pass mode. ``"validation"`` enables the
        evaluation branches in the models.
    device:
        Torch device for running inference. Defaults to the model's first
        parameter device.
    """

    requested = {item.strip().lower() for item in explanation_type.split(",")}
    if "both" in requested:
        requested = {"instance", "attention"}
    valid = {"instance", "attention"}
    if not requested.issubset(valid):
        raise ValueError(
            f"Unsupported explanation types {requested - valid}. "
            "Expected any combination of 'instance' and 'attention'."
        )
    should_compute_instance = "instance" in requested
    should_compute_attention = "attention" in requested

    if device is None:
        device = next(model.parameters()).device

    validation_flag = model_mode.lower() in {"validation", "val", "eval", "evaluation", "test"}

    instance_targets: list[np.ndarray] = []
    instance_predictions: list[np.ndarray] = []
    num_slides_with_instance_labels = 0
    num_instances_evaluated = 0

    ndcg_scores: list[float] = []
    auprc_scores: list[float] = []
    num_slides_with_evidence = 0

    model.eval()

    for batch in iter_explainability_batches(dataset):
        features = batch["features"].astype(np.float32, copy=False)
        features_tensor = torch.from_numpy(features).to(device)

        forward_kwargs = {"validation": validation_flag, "return_instance_outputs": True}
        if model_type.endswith("spvis"):
            coords = batch["coords"]
            height = batch.get("height")
            width = batch.get("width")
            if height is None or width is None:
                coords_array = np.asarray(coords)
                width = int(coords_array[:, 0].max()) if coords_array.size else 0
                height = int(coords_array[:, 1].max()) if coords_array.size else 0
            with torch.no_grad():
                outputs = model(features_tensor, coords, height, width, **forward_kwargs)
        else:
            with torch.no_grad():
                outputs = model(features_tensor, **forward_kwargs)

        if not isinstance(outputs, Sequence) or len(outputs) < 6:
            raise RuntimeError(
                "Model did not return instance outputs. Ensure the checkpoint "
                "was trained with the updated return_instance_outputs path."
            )

        *_, attention_tensor, details = outputs
        if isinstance(details, Mapping):
            instance_logits = details.get("instance_logits")
            attention_scores = details.get("attention", attention_tensor)
        else:
            instance_logits = None
            attention_scores = attention_tensor

        if should_compute_instance and instance_logits is not None:
            instance_labels = batch.get("instance_labels")
            if instance_labels is not None:
                num_slides_with_instance_labels += 1
                logits = instance_logits.detach().cpu()
                predictions = torch.argmax(logits, dim=1).numpy()
                targets = np.asarray(instance_labels, dtype=np.int64)
                if targets.shape[0] != predictions.shape[0]:
                    raise ValueError(
                        f"Instance label length {targets.shape[0]} does not match predictions {predictions.shape[0]} "
                        f"for slide {batch['slide_id']}"
                    )
                instance_predictions.append(predictions)
                instance_targets.append(targets)
                num_instances_evaluated += int(targets.shape[0])

        if should_compute_attention and attention_scores is not None:
            evidence = batch.get("evidence")
            if isinstance(evidence, Mapping) and evidence:
                class_id = evaluated_class
                if class_id is None:
                    class_id = int(batch["label"])
                if class_id in evidence:
                    ground_truth = np.asarray(evidence[class_id], dtype=np.float32)
                    if ground_truth.shape[0] != attention_scores.shape[-1]:
                        raise ValueError(
                            f"Evidence length {ground_truth.shape[0]} does not match attention scores "
                            f"{attention_scores.shape[-1]} for slide {batch['slide_id']}"
                        )
                    attention_values = _normalise_attention(attention_scores)
                    if np.any(ground_truth):
                        ndcg_scores.append(float(ndcg_score(ground_truth.reshape(1, -1), attention_values.reshape(1, -1))))
                        auprc_scores.append(float(average_precision_score(ground_truth, attention_values)))
                    num_slides_with_evidence += 1

    instance_macro_f1: Optional[float]
    instance_balanced_accuracy: Optional[float]
    if should_compute_instance and instance_targets:
        flat_targets = np.concatenate(instance_targets)
        flat_preds = np.concatenate(instance_predictions)
        if evaluated_class is None:
            instance_macro_f1 = float(f1_score(flat_targets, flat_preds, average="macro", zero_division=0))
            instance_balanced_accuracy = float(balanced_accuracy_score(flat_targets, flat_preds))
        else:
            binary_targets = (flat_targets == evaluated_class).astype(np.int64)
            binary_preds = (flat_preds == evaluated_class).astype(np.int64)
            instance_macro_f1 = float(f1_score(binary_targets, binary_preds, average="binary", zero_division=0))
            instance_balanced_accuracy = float(balanced_accuracy_score(binary_targets, binary_preds))
    else:
        instance_macro_f1 = None
        instance_balanced_accuracy = None

    attention_ndcg: Optional[float]
    attention_auprc: Optional[float]
    if should_compute_attention and ndcg_scores:
        attention_ndcg = float(np.mean(ndcg_scores))
        attention_auprc = float(np.mean(auprc_scores)) if auprc_scores else None
    else:
        attention_ndcg = None
        attention_auprc = None

    return ExplainabilityMetrics(
        explanation_type=",".join(sorted(requested)),
        evaluated_class=evaluated_class,
        model_mode=model_mode,
        num_slides=len(dataset.slide_data),
        num_slides_with_instance_labels=num_slides_with_instance_labels,
        num_slides_with_evidence=num_slides_with_evidence,
        num_instances_evaluated=num_instances_evaluated,
        instance_macro_f1=instance_macro_f1,
        instance_balanced_accuracy=instance_balanced_accuracy,
        attention_ndcg=attention_ndcg,
        attention_auprc=attention_auprc,
    )
