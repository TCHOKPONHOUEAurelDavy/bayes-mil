"""Dataset definitions used to synthesise MNIST-based MIL bags.

The implementations are direct adaptations of the helper classes the user
provided.  We expose one dataset class per task so that the generator can build
bags by reusing the original labelling rules without any additional
interpretability-specific logic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class NumberMILDataset(Dataset):
    def __init__(
        self,
        num_numbers: int,
        num_bags: int,
        num_instances: int,
        noise: float = 1.0,
        threshold: int = 1,
        sampling: str = "hierarchical",
        features_type: str = "mnist_pixels",
        features_path: Optional[Path] = None,
        mnist_root: Optional[Path] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.num_numbers = num_numbers
        self.num_bags = num_bags
        self.num_instances = num_instances
        self.thr = threshold
        self.numbers = []

        rng = np.random.default_rng(seed)

        while len(self.numbers) < num_bags:
            if sampling == "unique":
                sampled_numbers = rng.choice(
                    np.arange(self.num_numbers), size=num_instances, replace=False
                )
                self.numbers.append(sampled_numbers)
            elif sampling == "uniform":
                sampled_numbers = rng.choice(
                    np.arange(self.num_numbers), size=num_instances, replace=True
                )
                self.numbers.append(sampled_numbers)
            elif sampling == "hierarchical":
                sampled_numbers = np.where(
                    rng.integers(0, 2, size=num_numbers) == 1
                )[0]
                if len(sampled_numbers) > 0:
                    self.numbers.append(
                        rng.choice(sampled_numbers, size=num_instances, replace=True)
                    )
            else:
                raise ValueError(f"Unknown sampling strategy: {sampling}")

        self.numbers = np.concatenate(self.numbers)

        if features_type == "onehot":
            self.features = (
                rng.normal(
                    loc=0.0, scale=noise, size=(num_bags * num_instances, num_numbers)
                )
                + np.eye(num_numbers)[self.numbers]
            )
        elif features_type == "mnist_pixels":
            mnist_root = (
                Path(mnist_root)
                if mnist_root is not None
                else Path.home() / ".torch" / "datasets"
            )
            dataset = datasets.MNIST(
                root=str(mnist_root),
                train=True,
                download=True,
                transform=transforms.ToTensor(),
            )
            data_dict: Dict[int, torch.Tensor] = {
                idx: dataset.data[dataset.targets == idx].float() / 255.0
                for idx in range(10)
            }
            features_list = []
            for n in self.numbers:
                digit_bank = data_dict[int(n)]
                sample_idx = rng.integers(0, digit_bank.shape[0])
                features_list.append(digit_bank[sample_idx].reshape(-1))
            stacked = torch.stack(features_list)
            if noise > 0:
                stacked = stacked + noise * torch.randn_like(stacked)
            self.features = stacked.view(num_bags, num_instances, -1)
        elif features_type == "mnist_resnet18":
            if features_path is None:
                raise ValueError("`features_path` must be provided for mnist_resnet18")
            data_dict = {
                idx: torch.load(os.path.join(features_path, f"class_{idx}.pt"))
                for idx in range(10)
            }
            features_list = [
                data_dict[int(n)][rng.integers(0, data_dict[int(n)].shape[0])]
                for n in self.numbers
            ]
            stacked = torch.stack(features_list)
            self.features = stacked.view(num_bags, num_instances, -1)
            if noise > 0:
                self.features = self.features + noise * torch.randn_like(self.features)
        else:
            raise ValueError(f"Unknown features type: {features_type}")

        self.numbers = torch.tensor(self.numbers.reshape(num_bags, num_instances))
        if isinstance(self.features, np.ndarray):
            features_tensor = torch.tensor(
                self.features.reshape(num_bags, num_instances, -1),
                dtype=torch.float32,
            )
        else:
            features_tensor = self.features.to(torch.float32)
        self.features = features_tensor
        self.num_features = self.features.shape[-1]

    @property
    def num_classes(self) -> int:
        raise NotImplementedError()

    def __len__(self) -> int:
        return self.num_bags

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "features": self.features[idx],
            "bag_size": self.num_instances,
            "numbers": self.numbers[idx],
            "sample_ids": {},
        }


class FourBagsDataset(NumberMILDataset):
    @property
    def num_classes(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        c1_number, c2_number = 8, 9
        item = super().__getitem__(idx)
        number_count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        if number_count[c1_number] >= self.thr > number_count[c2_number]:
            targets = torch.tensor([1])
        elif number_count[c2_number] >= self.thr > number_count[c1_number]:
            targets = torch.tensor([2])
        elif (
            number_count[c1_number] >= self.thr and number_count[c2_number] >= self.thr
        ):
            targets = torch.tensor([3])
        else:
            targets = torch.tensor([0])
        num_positions = {
            c1_number: (item["numbers"] == c1_number) * 1,
            c2_number: (item["numbers"] == c2_number) * 1,
        }
        evidence = {
            0: -num_positions[c1_number] - num_positions[c2_number],
            1: num_positions[c1_number] - num_positions[c2_number],
            2: -num_positions[c1_number] + num_positions[c2_number],
            3: num_positions[c1_number] + num_positions[c2_number],
        }
        return {**item, "targets": targets, "evidence": evidence}


class EvenOddDataset(NumberMILDataset):
    @property
    def num_classes(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        even_numbers = torch.tensor([0, 2, 4, 6, 8])
        odd_numbers = torch.tensor([1, 3, 5, 7, 9])
        item = super().__getitem__(idx)
        number_count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        if number_count[even_numbers].sum() > number_count[odd_numbers].sum():
            targets = torch.tensor([1])
        else:
            targets = torch.tensor([0])
        pos_evidence = torch.isin(item["numbers"], even_numbers) * 1
        neg_evidence = torch.isin(item["numbers"], odd_numbers) * 1
        evidence = {0: neg_evidence - pos_evidence, 1: pos_evidence - neg_evidence}

        instance_labels = torch.zeros_like(item["numbers"])
        instance_labels[torch.isin(item["numbers"], even_numbers)] = 1

        return {
            **item,
            "targets": targets,
            "evidence": evidence,
            "instance_labels": instance_labels,
        }


class AdjacentPairsDataset(NumberMILDataset):
    @property
    def num_classes(self) -> int:
        return 2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        number_count = torch.bincount(item["numbers"], minlength=self.num_numbers)
        evidence_thr = 5
        numbers = (number_count >= self.thr).nonzero().squeeze().tolist()
        pos_tuples = []
        if isinstance(numbers, list) and len(numbers) > 1:
            numbers = list(filter(lambda x: x < evidence_thr, numbers))
            for idy, num_0 in enumerate(numbers):
                num_1 = numbers[(idy + 1) % len(numbers)]
                if (num_0 + 1) == num_1:
                    pos_tuples.append([num_0, num_1])
        if len(pos_tuples) >= self.thr:
            targets = torch.tensor([1])
        else:
            targets = torch.tensor([0])
        if pos_tuples:
            flat = torch.tensor(pos_tuples).flatten()
            pos_evidence = torch.isin(item["numbers"], flat) * 1
        else:
            pos_evidence = torch.zeros_like(item["numbers"], dtype=torch.int64)
        return {
            **item,
            "targets": targets,
            "evidence": {0: -pos_evidence, 1: pos_evidence},
        }


class FourBagsPlusDataset(NumberMILDataset):
    @property
    def num_classes(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = super().__getitem__(idx)
        bag_numbers = item["numbers"]

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
        instance_labels[bag_numbers == 3] = 1
        instance_labels[bag_numbers == 5] = 1
        instance_labels[bag_numbers == 7] = 3
        if has_7:
            instance_labels[bag_numbers == 1] = 3
        else:
            instance_labels[bag_numbers == 1] = 1

        evidence = {
            0: -(
                (bag_numbers == 3).float()
                + (bag_numbers == 5).float()
                + (bag_numbers == 1).float()
                + (bag_numbers == 7).float()
            ),
            1: (bag_numbers == 3).float()
            + (bag_numbers == 5).float()
            - (bag_numbers == 1).float()
            - (bag_numbers == 7).float(),
            2: (bag_numbers == 1).float()
            - (bag_numbers == 7).float()
            - (bag_numbers == 3).float()
            - (bag_numbers == 5).float(),
            3: (bag_numbers == 1).float()
            + (bag_numbers == 7).float()
            - (bag_numbers == 3).float()
            - (bag_numbers == 5).float(),
        }
        item["targets"] = torch.tensor([bag_label])
        item["instance_labels"] = instance_labels
        item["evidence"] = evidence

        return item


DATASET_CLASSES = {
    "mnist_fourbags": FourBagsDataset,
    "mnist_even_odd": EvenOddDataset,
    "mnist_adjacent_pairs": AdjacentPairsDataset,
    "mnist_fourbags_plus": FourBagsPlusDataset,
}
