"""Helpers defining synthetic MNIST MIL tasks used for interpretability tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import torch


@dataclass(frozen=True)
class EvidenceBundle:
    """Container describing the metadata attached to a MIL bag."""

    target: int
    evidence: Mapping[int, torch.Tensor]
    instance_labels: Optional[torch.Tensor]


def _count_digits(numbers: torch.Tensor, num_numbers: int) -> torch.Tensor:
    return torch.bincount(numbers, minlength=num_numbers)


def _positions(numbers: torch.Tensor, value: int) -> torch.Tensor:
    return (numbers == value).to(torch.float32)


def fourbags_metadata(
    numbers: torch.Tensor, num_numbers: int = 10, threshold: int = 1
) -> EvidenceBundle:
    number_count = _count_digits(numbers, num_numbers)
    c1_number, c2_number = 8, 9
    if number_count[c1_number] >= threshold > number_count[c2_number]:
        target = 1
    elif number_count[c2_number] >= threshold > number_count[c1_number]:
        target = 2
    elif number_count[c1_number] >= threshold and number_count[c2_number] >= threshold:
        target = 3
    else:
        target = 0
    num_positions = {
        c1_number: _positions(numbers, c1_number),
        c2_number: _positions(numbers, c2_number),
    }
    evidence = {
        0: -num_positions[c1_number] - num_positions[c2_number],
        1: num_positions[c1_number] - num_positions[c2_number],
        2: -num_positions[c1_number] + num_positions[c2_number],
        3: num_positions[c1_number] + num_positions[c2_number],
    }
    return EvidenceBundle(target=target, evidence=evidence, instance_labels=None)


def evenodd_metadata(numbers: torch.Tensor) -> EvidenceBundle:
    even_numbers = torch.tensor([0, 2, 4, 6, 8], device=numbers.device)
    odd_numbers = torch.tensor([1, 3, 5, 7, 9], device=numbers.device)
    number_count = _count_digits(numbers, 10)
    if number_count[even_numbers].sum() > number_count[odd_numbers].sum():
        target = 1
    else:
        target = 0
    pos_evidence = torch.isin(numbers, even_numbers).to(torch.float32)
    neg_evidence = torch.isin(numbers, odd_numbers).to(torch.float32)
    evidence = {0: neg_evidence - pos_evidence, 1: pos_evidence - neg_evidence}
    instance_labels = torch.zeros_like(numbers)
    instance_labels[torch.isin(numbers, even_numbers)] = 1
    return EvidenceBundle(target=target, evidence=evidence, instance_labels=instance_labels)


def adjacentpairs_metadata(
    numbers: torch.Tensor, num_numbers: int = 10, threshold: int = 1
) -> EvidenceBundle:
    number_count = _count_digits(numbers, num_numbers)
    evidence_thr = 5
    digits_with_threshold = (number_count >= threshold).nonzero().squeeze().tolist()
    pos_tuples = []
    if isinstance(digits_with_threshold, list) and len(digits_with_threshold) > 1:
        digits_with_threshold = [
            digit for digit in digits_with_threshold if digit < evidence_thr
        ]
        for idx, num_0 in enumerate(digits_with_threshold):
            num_1 = digits_with_threshold[(idx + 1) % len(digits_with_threshold)]
            if (num_0 + 1) == num_1:
                pos_tuples.append([num_0, num_1])
    if len(pos_tuples) >= threshold:
        target = 1
    else:
        target = 0
    if pos_tuples:
        flat = torch.tensor(pos_tuples, device=numbers.device).flatten()
        pos_evidence = torch.isin(numbers, flat).to(torch.float32)
    else:
        pos_evidence = torch.zeros_like(numbers, dtype=torch.float32)
    evidence = {0: -pos_evidence, 1: pos_evidence}
    return EvidenceBundle(target=target, evidence=evidence, instance_labels=None)


def fourbagsplus_metadata(numbers: torch.Tensor) -> EvidenceBundle:
    bag_numbers = numbers
    has_3 = (bag_numbers == 3).any().item()
    has_5 = (bag_numbers == 5).any().item()
    has_1 = (bag_numbers == 1).any().item()
    has_7 = (bag_numbers == 7).any().item()
    if has_3 and has_5:
        bag_label = 1
    elif has_1 and not has_7:
        bag_label = 2
    elif has_1 and has_7:
        bag_label = 3
    else:
        bag_label = 0
    instance_labels = torch.zeros_like(bag_numbers)
    instance_labels[(bag_numbers == 3) | (bag_numbers == 5)] = 1
    instance_labels[bag_numbers == 7] = 3
    if has_7:
        instance_labels[bag_numbers == 1] = 3
    else:
        instance_labels[bag_numbers == 1] = 1
    evidence = {
        0: -(
            (bag_numbers == 3).to(torch.float32)
            + (bag_numbers == 5).to(torch.float32)
            + (bag_numbers == 1).to(torch.float32)
            + (bag_numbers == 7).to(torch.float32)
        ),
        1: (bag_numbers == 3).to(torch.float32)
        + (bag_numbers == 5).to(torch.float32)
        - (bag_numbers == 1).to(torch.float32)
        - (bag_numbers == 7).to(torch.float32),
        2: (bag_numbers == 1).to(torch.float32)
        - (bag_numbers == 7).to(torch.float32)
        - (bag_numbers == 3).to(torch.float32)
        - (bag_numbers == 5).to(torch.float32),
        3: (bag_numbers == 1).to(torch.float32)
        + (bag_numbers == 7).to(torch.float32)
        - (bag_numbers == 3).to(torch.float32)
        - (bag_numbers == 5).to(torch.float32),
    }
    return EvidenceBundle(target=bag_label, evidence=evidence, instance_labels=instance_labels)


TASK_METADATA_FNS: Dict[str, callable] = {
    "mnist_fourbags": fourbags_metadata,
    "mnist_even_odd": evenodd_metadata,
    "mnist_adjacent_pairs": adjacentpairs_metadata,
    "mnist_fourbags_plus": fourbagsplus_metadata,
}
