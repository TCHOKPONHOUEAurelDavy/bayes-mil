"""Utilities for explainability metrics and data loading."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterator, Mapping, Optional, Sequence, Set

import h5py
import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
import re


INSTANCE_EXPLANATION_TYPES: Set[str] = {"learn", "learn-modified", "learn-plus"}
ATTENTION_EXPLANATION_TYPES: Set[str] = {
    "int-attn-coeff",
    "int-built-in",
    "int-computed",
    "int-clf",
}
ALL_EXPLANATION_TYPES: Set[str] = INSTANCE_EXPLANATION_TYPES | ATTENTION_EXPLANATION_TYPES


# Mapping from high-level model families to the explanation heads they expose.
MODEL_EXPLANATION_GROUPS: Dict[str, FrozenSet[str]] = {
    "attention_mil": frozenset({"learn", "int-attn-coeff", "int-computed"}),
    "additive_mil": frozenset({"learn", "int-attn-coeff", "int-built-in", "int-computed"}),
    "conjunctive_mil": frozenset({"learn", "int-attn-coeff", "int-built-in"}),
    "trans_mil": frozenset({"learn", "int-attn-coeff", "int-computed"}),
}


# Alias table allowing different model identifiers to share the same explanation
# configuration. The keys are normalised to lower-case before lookup.
MODEL_TYPE_GROUPS: Dict[str, str] = {
    # Explicit experiment scripts.
    "attention_mil": "attention_mil",
    "attention-mil": "attention_mil",
    "additive_mil": "additive_mil",
    "additive-mil": "additive_mil",
    "conjunctive_mil": "conjunctive_mil",
    "conjunctive-mil": "conjunctive_mil",
    "trans_mil": "trans_mil",
    "trans-mil": "trans_mil",
}


# Map the Bayes-MIL CLI model suffixes onto the same explanation group labels.
_BMIL_SUFFIX_GROUPS: Dict[str, Set[str]] = {
    "attention_mil": {"vis", "enc", "spvis", "a", "f"},
    "additive_mil": {"addvis", "addenc", "addspvis"},
    "conjunctive_mil": {"conjvis", "convis", "conjenc", "conenc", "conjspvis", "conspvis"},
}

for group_name, suffixes in _BMIL_SUFFIX_GROUPS.items():
    for suffix in suffixes:
        MODEL_TYPE_GROUPS[f"bmil-{suffix}"] = group_name
        MODEL_TYPE_GROUPS[f"bmil_{suffix}"] = group_name


@dataclass(frozen=True)
class ExplanationSelection:
    """Describe which explanation heads will be evaluated for a model."""

    requested: FrozenSet[str]
    available: FrozenSet[str]
    instance: FrozenSet[str]
    attention: FrozenSet[str]
    ignored: FrozenSet[str]


@dataclass
class ExplainabilityMetrics:
    """Container for aggregated explainability scores for a single explanation name."""

    explanation_type: str
    metric_family: str
    model_mode: str
    model_identifier: Optional[str] = None
    num_slides: int
    num_slides_with_instance_labels: int
    num_slides_with_evidence: int
    num_instances_evaluated: int
    instance_macro_f1: Optional[float]
    instance_balanced_accuracy: Optional[float]
    attention_ndcg: Optional[float]
    attention_auprc2: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        record: Dict[str, Optional[float]] = {
            "explanation_type": self.explanation_type,
            "metric_family": self.metric_family,
            "model_mode": self.model_mode,
            "model_identifier": self.model_identifier,
            "num_slides": float(self.num_slides),
            "num_slides_with_instance_labels": float(self.num_slides_with_instance_labels),
            "num_slides_with_evidence": float(self.num_slides_with_evidence),
            "num_instances_evaluated": float(self.num_instances_evaluated),
            "instance_macro_f1": self.instance_macro_f1,
            "instance_balanced_accuracy": self.instance_balanced_accuracy,
            "attention_ndcg": self.attention_ndcg,
            "attention_ndgcn": self.attention_ndcg,
            "attention_auprc2": self.attention_auprc2,
            "attention_auprc": self.attention_auprc2,
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


def _parse_explanation_request(selection: str) -> Set[str]:
    """Normalise the user provided explanation selection string."""

    if not selection:
        return set()

    tokens = {
        token.strip().lower()
        for token in re.split(r"[\s,]+", selection)
        if token.strip()
    }
    if not tokens:
        return set()
    if "all" in tokens:
        tokens = set(ALL_EXPLANATION_TYPES)
    unknown = tokens - ALL_EXPLANATION_TYPES
    if unknown:
        raise ValueError(
            "Unsupported explanation types {}. Expected any of {} or 'all'.".format(
                sorted(unknown), sorted(ALL_EXPLANATION_TYPES)
            )
        )
    return tokens


def _resolve_model_explanations(model_type: Optional[str]) -> Set[str]:
    """Return the explanation heads supported by the requested model."""

    if not model_type:
        return set(ALL_EXPLANATION_TYPES)

    key = model_type.lower()
    group_name = MODEL_TYPE_GROUPS.get(key)

    if group_name is None:
        # Retry using underscore-delimited keys so "bmil-vis" and "bmil_vis"
        # both map to the same configuration.  Unknown model names fall back to
        # the full explanation set to remain backward compatible.
        underscore_key = key.replace("-", "_")
        group_name = MODEL_TYPE_GROUPS.get(underscore_key)

    if group_name is None:
        return set(ALL_EXPLANATION_TYPES)

    return set(MODEL_EXPLANATION_GROUPS.get(group_name, ALL_EXPLANATION_TYPES))


def resolve_explanation_selection(model_type: str, explanation_type: str) -> ExplanationSelection:
    """Resolve the explanation heads that should be evaluated for a model."""

    requested = _parse_explanation_request(explanation_type)
    available = _resolve_model_explanations(model_type)

    available_instance = available & INSTANCE_EXPLANATION_TYPES
    available_attention = available & ATTENTION_EXPLANATION_TYPES

    if requested:
        instance = requested & available_instance
        attention = requested & available_attention
    else:
        instance = set(available_instance)
        attention = set(available_attention)

    ignored = requested - (instance | attention)

    return ExplanationSelection(
        requested=frozenset(requested),
        available=frozenset(available),
        instance=frozenset(instance),
        attention=frozenset(attention),
        ignored=frozenset(ignored),
    )


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


def _compute_ndgcn(ground_truth: np.ndarray, attention_values: np.ndarray) -> float:
    """Compute the NDCGN score following the reference implementation."""

    # Convert inputs to double precision arrays to avoid surprises when dividing
    # small values. ``ground_truth`` may contain both positive and negative
    # scores; only positive evidence should contribute to the gain of a patch.
    relevance = np.maximum(ground_truth.astype(np.float64, copy=False), 0.0)
    scores = attention_values.astype(np.float64, copy=False)

    if relevance.size == 0:
        return 0.0

    # Sort patches by the attention weight assigned by the model.
    ranked_indices = np.argsort(scores)[::-1]
    ranked_relevance = relevance[ranked_indices]

    # Positions are 1-indexed in the logarithmic discount term. We use the
    # ``log2(i + 2)`` formulation so that the top-ranked element is divided by
    # ``log2(2) = 1``.
    positions = np.arange(ranked_relevance.size, dtype=np.float64)
    discounts = np.log2(positions + 2.0)

    dcg = float(np.sum(ranked_relevance / discounts))

    # Compute the ideal DCG by ordering patches based on the ground-truth
    # relevance, ensuring we divide by the same discount factors as above.
    ideal_indices = np.argsort(relevance)[::-1]
    ideal_relevance = relevance[ideal_indices]
    ideal_dcg = float(np.sum(ideal_relevance / discounts))

    if ideal_dcg == 0.0:
        return 0.0
    return dcg / ideal_dcg


def evaluate_explainability(
    model: torch.nn.Module,
    dataset,
    *,
    model_type: str,
    explanation_type: str = "all",
    model_mode: str = "validation",
    model_identifier: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Sequence[ExplainabilityMetrics]:
    """Compute interpretability metrics for one or more explanation names.

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
        Comma separated list of explanation names. Supported instance-centric
        names: ``learn``, ``learn-modified`` and ``learn-plus``. Supported
        attention-centric names: ``int-attn-coeff``, ``int-built-in``,
        ``int-computed`` and ``int-clf``. The special value ``all`` evaluates
        every available name.
    model_mode:
        Keyword indicating the forward-pass mode. ``"validation"`` enables the
        evaluation branches in the models.
    model_identifier:
        Optional string stored alongside the aggregated metrics so downstream
        consumers can trace which checkpoint produced each explanation.
    device:
        Torch device for running inference. Defaults to the model's first
        parameter device.
    """

    selection = resolve_explanation_selection(model_type, explanation_type)
    selected_instance_types = sorted(selection.instance)
    selected_attention_types = sorted(selection.attention)

    if not selected_instance_types and not selected_attention_types:
        return []

    should_compute_instance = bool(selected_instance_types)
    should_compute_attention = bool(selected_attention_types)

    if device is None:
        device = next(model.parameters()).device

    validation_flag = model_mode.lower() in {"validation", "val", "eval", "evaluation", "test"}
    total_slides = len(dataset.slide_data)

    instance_targets: list[np.ndarray] = []
    instance_predictions: list[np.ndarray] = []
    num_slides_with_instance_labels = 0
    num_instances_evaluated = 0

    ndcg_scores: list[float] = []
    auprc2_scores: list[float] = []
    num_slides_with_evidence = 0

    model.eval()

    for batch in iter_explainability_batches(dataset):
        # Move slide features to the evaluation device. The features are stored
        # as NumPy arrays in the HDF5 files; we reuse the existing memory when
        # converting to ``float32`` to avoid unnecessary copies.
        features = batch["features"].astype(np.float32, copy=False)
        features_tensor = torch.from_numpy(features).to(device)

        # Every forward pass must request ``return_instance_outputs`` so that we
        # have access to the per-patch logits and attention coefficients.
        forward_kwargs = {"validation": validation_flag, "return_instance_outputs": True}
        if model_type.endswith("spvis"):
            coords = batch["coords"]
            height = batch.get("height")
            width = batch.get("width")
            if height is None or width is None:
                # Some datasets do not store the canvas size; fall back to the
                # maximal coordinates observed in the slide as an approximation.
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
                # Convert logits to discrete predictions and cache both the
                # predictions and ground-truth targets to aggregate metrics
                # across slides.
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
                class_id = int(batch["label"])
                if class_id in evidence:
                    ground_truth = np.asarray(evidence[class_id], dtype=np.float32)
                    if ground_truth.shape[0] != attention_scores.shape[-1]:
                        raise ValueError(
                            f"Evidence length {ground_truth.shape[0]} does not match attention scores "
                            f"{attention_scores.shape[-1]} for slide {batch['slide_id']}"
                        )
                    # Normalise the attention vector so that it forms a proper
                    # distribution before computing ranking-based metrics.
                    attention_values = _normalise_attention(attention_scores)

                    # The original study evaluates attention explanations by
                    # comparing the ranked attention scores against positive
                    # (supporting) and negative (contradicting) evidence masks.
                    evidence_pos = (ground_truth > 0).astype(np.int64, copy=False)
                    evidence_neg = (ground_truth < 0).astype(np.int64, copy=False)

                    auprc_components: list[float] = []

                    # When the slide contains a non-trivial mix of positive and
                    # non-positive evidence we compute the NDCGN score and the
                    # positive branch of the AUPRC2 metric.
                    if 0 < evidence_pos.sum() < evidence_pos.shape[0]:
                        ndcg_scores.append(
                            _compute_ndgcn(np.maximum(ground_truth, 0.0), attention_values)
                        )
                        auprc_components.append(
                            float(average_precision_score(evidence_pos, attention_values))
                        )

                    # The second branch measures how well the attention avoids
                    # highlighting known negative evidence; this is implemented
                    # by flipping the attention scores before evaluating the
                    # precision-recall curve.
                    if 0 < evidence_neg.sum() < evidence_neg.shape[0]:
                        auprc_components.append(
                            float(average_precision_score(evidence_neg, -attention_values))
                        )

                    if auprc_components:
                        auprc2_scores.append(float(np.mean(auprc_components)))
                    num_slides_with_evidence += 1

    instance_macro_f1: Optional[float]
    instance_balanced_accuracy: Optional[float]
    if should_compute_instance and instance_targets:
        flat_targets = np.concatenate(instance_targets)
        flat_preds = np.concatenate(instance_predictions)
        # Multi-class setting: compute the macro F1 across all classes and the
        # balanced accuracy exactly as implemented in the reference submission
        # via sklearn.
        instance_macro_f1 = float(
            f1_score(flat_targets, flat_preds, average="macro", zero_division=0)
        )
        instance_balanced_accuracy = float(balanced_accuracy_score(flat_targets, flat_preds))
    else:
        instance_macro_f1 = None
        instance_balanced_accuracy = None

    attention_ndcg: Optional[float]
    attention_auprc2: Optional[float]
    if should_compute_attention and ndcg_scores:
        attention_ndcg = float(np.mean(ndcg_scores))
        attention_auprc2 = float(np.mean(auprc2_scores)) if auprc2_scores else None
    else:
        attention_ndcg = None
        attention_auprc2 = None

    metrics: list[ExplainabilityMetrics] = []

    if should_compute_instance:
        for name in selected_instance_types:
            metrics.append(
                ExplainabilityMetrics(
                    explanation_type=name,
                    metric_family="instance",
                    model_mode=model_mode,
                    num_slides=total_slides,
                    num_slides_with_instance_labels=num_slides_with_instance_labels,
                    num_slides_with_evidence=num_slides_with_evidence,
                    num_instances_evaluated=num_instances_evaluated,
                    instance_macro_f1=instance_macro_f1,
                    instance_balanced_accuracy=instance_balanced_accuracy,
                    attention_ndcg=None,
                    attention_auprc2=None,
                    model_identifier=model_identifier,
                )
            )

    if should_compute_attention:
        for name in selected_attention_types:
            metrics.append(
                ExplainabilityMetrics(
                    explanation_type=name,
                    metric_family="attention",
                    model_mode=model_mode,
                    model_identifier=model_identifier,
                    num_slides=total_slides,
                    num_slides_with_instance_labels=num_slides_with_instance_labels,
                    num_slides_with_evidence=num_slides_with_evidence,
                    num_instances_evaluated=num_instances_evaluated,
                    instance_macro_f1=None,
                    instance_balanced_accuracy=None,
                    attention_ndcg=attention_ndcg,
                    attention_auprc2=attention_auprc2,
                )
            )

    return metrics
