import json
import os
import logging
import random
import copy
from abc import ABC, abstractmethod
from typing import List, Dict, Set, Any, Generator

logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """
    Abstract Base Class for all datasets.

    Any future dataset (e.g. CSVDataset, HuggingFaceDataset)
    must inherit from this class and implement the abstract methods.
    """

    def __init__(self):
        self.data: List[Dict[str, Any]] = []

    @abstractmethod
    def load(self, file_path: str, **kwargs) -> None:
        """
        Specific loading logic must be implemented by subclasses.
        """
        pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.data[idx]

    def __iter__(self) -> Generator[Dict[str, Any], None, None]:
        for item in self.data:
            yield item

    def gen_variants(self, record: Dict[str, Any], var_num: int = 5) -> List[Dict[str, Any]]:
        """
        Default implementation: returns the record as-is without variants.
        """
        return [record]

    def iter_with_variants(self, var_num: int = 5) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Yields a batch of variants for each record.
        """
        for item in self.data:
            yield self.gen_variants(item, var_num=var_num)

    def filter_by_ids(self, processed_ids: Set[str]) -> None:
        """
        Filter data based on the processed ID set
        """
        original_count = len(self)

        self.data = [
            item for item in self.data
            if str(item.get('id')) not in processed_ids
        ]

        filtered_count = original_count - len(self.data)
        if filtered_count > 0:
            logger.info(f"Filtered out {filtered_count} processed items. Remaining: {len(self.data)}")


class DeceptionDataset(BaseDataset):
    def __init__(self, file_path: str, limit: int = -1):
        super().__init__()
        self.load(file_path, limit=limit)

    def load(self, file_path: str, limit: int = -1) -> None:
        if not os.path.exists(file_path):
            logger.error(f"Dataset file not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.info(f"Loading deception evaluation dataset from: {file_path}")

        count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if 0 < limit <= count:
                    break

                line = line.strip()
                if not line: continue

                try:
                    entry = json.loads(line)

                    required_keys = ["id", "category", "base"]
                    if not all(key in entry for key in required_keys):
                        logger.warning(f"Skipping incomplete data at line {count + 1}: missing keys")
                        continue

                    self.data.append(entry)
                    count += 1
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line.")
                    continue

        logger.info(f"Successfully loaded {len(self.data)} samples.")


def create_dataset(dataset_type: str, file_path: str, limit: int = -1) -> BaseDataset:
    """
    Factory function to create dataset safely.
    """
    dataset_mapping = {
        "DeceptionDataset": DeceptionDataset,
    }

    dataset_class = dataset_mapping.get(dataset_type)

    if not dataset_class:
        raise ValueError(f"Unknown Dataset type: '{dataset_type}'. Supported types: {list(dataset_mapping.keys())}")

    return dataset_class(file_path, limit)
