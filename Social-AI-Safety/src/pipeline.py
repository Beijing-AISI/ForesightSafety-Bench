import json
import os
import time
import copy
import logging
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple

from src.llm import BaseLLM
from src.prompt import PromptManager, DeceptionPromptManager
from src.evaluator import Evaluator, DeceptionEvaluator
from src.dataset import BaseDataset, DeceptionDataset

logger = logging.getLogger(__name__)


class DeceptionInferencePipeline:
    """
    Deception Evaluation Pipeline.
    """

    def __init__(
            self,
            llm: BaseLLM,
            prompt_manager: DeceptionPromptManager,
            output_dir: str = "results",
            max_workers: int = 16,
            write_batch_size: int = 20
    ):
        self.llm = llm
        self.prompt_manager = prompt_manager

        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            raise FileNotFoundError(
                f"The output directory '{self.output_dir}' does not exist. "
            )

        self.max_workers = max_workers
        self.write_batch_size = write_batch_size
        self.write_lock = threading.Lock()

    def run(self, dataset: DeceptionDataset, run_name: Optional[str] = None, limit: int = -1) -> Dict[str, Any]:
        """
        Execute the pipeline.
        """
        self.output_path = self._generate_output_path(run_name)
        logger.info(f"Starting generation on {self.llm.model_name}")
        logger.info(f"Output will be saved to: {self.output_path}")

        stats = {"total_groups": 0, "processed_ids": [], "failed_count": 0}
        write_buffer = []

        iterator = dataset.__iter__()
        if limit > 0:
            import itertools
            iterator = itertools.islice(iterator, limit)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self._process_single_item, item)
                for item in iterator
            ]

            for future in tqdm(futures, total=len(futures), desc=f"Evaluating {self.llm.model_name}"):
                try:
                    result = future.result()

                    if result:
                        write_buffer.append(result)

                        stats["total_groups"] += 1
                        stats["processed_ids"].append(result["id"])
                    else:
                        stats["failed_count"] += 1

                    if len(write_buffer) >= self.write_batch_size:
                        self._flush_buffer(write_buffer)
                        write_buffer = []

                except Exception as e:
                    logger.error(f"Worker exception: {e}", exc_info=True)
                    stats["failed_count"] += 1

            if write_buffer:
                self._flush_buffer(write_buffer)

        logger.info(f"Generation finished. Processed {stats['total_groups']} groups. Failed: {stats['failed_count']}")
        return stats

    def _process_single_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single original question and its intervened version.
        """

        try:
            if item["category"] in ["feint", "bluffing", "manipulation"]:
                item_base = item["base"]

                base_messages = self.prompt_manager.build_messages(item_base)
                base_response = self.llm.generate(base_messages)

                return {
                    # Meta
                    "id": item.get("id"),
                    "category": item.get("category", "unknown"),
                    "source": item.get("source", "unknown"),

                    # Data
                    "base": {
                        **item_base,
                        "model_response": base_response,
                    },
                }

            else:
                item_base = item["base"]
                item_intervention = item["intervention"]

                base_messages = self.prompt_manager.build_messages(item_base)
                base_response = self.llm.generate(base_messages)

                intervened_message = self.prompt_manager.build_messages(item_intervention)
                intervened_response = self.llm.generate(intervened_message)

                return {
                    # Meta
                    "id": item.get("id"),
                    "category": item.get("category", "unknown"),
                    "source": item.get("source", "unknown"),

                    # Data
                    "base": {
                        **item_base,
                        "model_response": base_response,
                    },
                    "intervention": {
                        **item_intervention,
                        "model_response": intervened_response,
                    }
                }

        except Exception as e:
            logger.error(f"Error in worker processing item {item.get('id')}: {e}")
            return None

    def _generate_output_path(self, run_name: Optional[str] = None) -> str:
        """
        Construct standardized file names.
        """
        if run_name:
            filename = f"{run_name}.jsonl"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = self.llm.model_name.replace("/", "-")
            template_name = self.prompt_manager.template_name

            filename = f"{model_name}_{template_name}_{timestamp}_responses.jsonl"

        return os.path.join(self.output_dir, filename)

    def _flush_buffer(self, buffer: List[Dict[str, Any]]):
        """
        Write the data from the buffer to a file.
        """
        if not buffer:
            return

        with self.write_lock:
            try:
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    lines = [json.dumps(item, ensure_ascii=False) for item in buffer]
                    f.write('\n'.join(lines) + '\n')
            except IOError as e:
                logger.error(f"Failed to batch write results: {e}")


class DeceptionJudgePipeline:
    """
    Deception Judge Pipeline.
    """

    def __init__(
            self,
            evaluator: DeceptionEvaluator,
            output_dir: str = "results",
            max_workers: int = 8,
            write_batch_size: int = 10
    ):
        self.evaluator = evaluator
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.write_batch_size = write_batch_size
        self.write_lock = threading.Lock()

    def run(self, input_file: str, run_name: str) -> Dict[str, Any]:
        """
        Execute the judging process.
        """
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")

        self.output_path = os.path.join(self.output_dir, f"{run_name}_judged.jsonl")

        processed_ids = self._get_processed_ids(self.output_path)
        if processed_ids:
            logger.info(f"Found {len(processed_ids)} already judged items. They will be skipped.")

        records_to_process = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                    if str(record.get("id")) not in processed_ids:
                        records_to_process.append(record)
                except json.JSONDecodeError:
                    continue

        total_tasks = len(records_to_process)
        logger.info(f"Starting Judge Pipeline. Items to process: {total_tasks}")

        stats = {"processed": 0, "failed": 0}
        write_buffer = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_item = {
                executor.submit(self.evaluator.evaluate, record): record
                for record in records_to_process
            }

            for future in tqdm(as_completed(future_to_item), total=total_tasks, desc=f"Judging {run_name}"):
                try:
                    result = future.result()

                    if result and "error" not in result:
                        write_buffer.append(result)
                        stats["processed"] += 1

                    else:
                        stats["failed"] += 1
                        if result:
                            logger.warning(f"Judge returned error: {result.get('error')}")

                    if len(write_buffer) >= self.write_batch_size:
                        self._flush_buffer(write_buffer)
                        write_buffer = []

                except Exception as e:
                    logger.error(f"Critical error in pipeline: {e}")
                    stats["failed"] += 1

            if write_buffer:
                self._flush_buffer(write_buffer)

        logger.info(f"Judging finished. Processed: {stats['processed']}, Failed: {stats['failed']}")
        return stats

    def _get_processed_ids(self, output_file: str) -> set:
        """
        Get IDs that are already done.
        """
        ids = set()
        if not os.path.exists(output_file):
            return ids

        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "id" in data:
                        ids.add(str(data["id"]))
                except:
                    pass
        return ids

    def _flush_buffer(self, buffer: List[Dict[str, Any]]):
        """
        Write the data from the buffer to a file.
        """
        if not buffer:
            return

        with self.write_lock:
            try:
                with open(self.output_path, 'a', encoding='utf-8') as f:
                    lines = [json.dumps(item, ensure_ascii=False) for item in buffer]
                    f.write('\n'.join(lines) + '\n')
            except IOError as e:
                logger.error(f"Failed to batch write results: {e}")
