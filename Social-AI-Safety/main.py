import os
import sys
import random
import argparse
import logging
import yaml
import json
from typing import Set, Dict, Any
from dotenv import load_dotenv

from src.dataset import BaseDataset, create_dataset
from src.llm import BaseLLM, create_llm
from src.prompt import DeceptionPromptManager
from src.evaluator import Evaluator
from src.pipeline import DeceptionInferencePipeline

load_dotenv(override=True)


def load_yaml(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Config file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def setup_logging(log_file: str = "run.log", log_dir: str = "./logs") -> None:
    log_path = os.path.join(log_dir, log_file)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, mode='a', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def set_seed(seed: int) -> None:
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)


def get_processed_ids(output_path: str) -> Set[str]:
    if not os.path.exists(output_path):
        return set()

    processed_ids = set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    data = json.loads(line)
                    processed_ids.add(str(data['id']))
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Failed to read existing output file: {e}")

    return processed_ids


def parse_args():
    parser = argparse.ArgumentParser(description="Config-driven Evaluation Pipeline")

    group_cfg = parser.add_argument_group("Configuration Files")
    group_cfg.add_argument("--dataset-config", type=str, default="configs/datasets/deceptioneval.yaml",
                           help="Path to dataset config")
    group_cfg.add_argument("--prompt-config", type=str, default="configs/prompts/deception.yaml",
                           help="Path to prompt config")
    group_cfg.add_argument("--llm-config", type=str, default="configs/llms/gpt-5.1.yaml",
                           help="Path to LLM client config")
    group_cfg.add_argument("--gen-config", type=str, default="configs/gen_config.yaml",
                           help="Path to generation params config")

    group_run = parser.add_argument_group("Runtime Args")
    group_run.add_argument("--output-dir", type=str, default="results", )
    group_run.add_argument("--run-name", type=str, default=None, help="Name for this run (output filename)")
    group_run.add_argument("--limit", type=int, default=-1, help="Limit number of groups to process (for debugging)")
    group_run.add_argument("--workers", type=int, default=16, help="Parallel workers")
    group_run.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    group_run.add_argument("--force", action="store_true", help="Force restart (ignore existing results)")

    return parser.parse_args()


def main():
    args = parse_args()

    log_name = f"{args.run_name}.log" if args.run_name else "run.log"
    setup_logging(log_name)
    logger = logging.getLogger(__name__)

    logger.info("=" * 30)
    logger.info(f"Setting random seed to: {args.seed}")
    set_seed(args.seed)
    logger.info(f"Initializing Pipeline with args: {vars(args)}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables!")

    # init pipeline
    try:
        # Load Dataset
        logger.info(f"Loading Dataset Config: {args.dataset_config}")
        dataset_config = load_yaml(args.dataset_config)
        dataset = create_dataset(**dataset_config)

        # Init LLM
        logger.info(f"Loading LLM Config: {args.llm_config} & {args.gen_config}")
        llm_config = load_yaml(args.llm_config)
        gen_config = load_yaml(args.gen_config)
        llm = create_llm(**llm_config, gen_config=gen_config)

        # Init PromptManager
        logger.info(f"Loading Prompt Config: {args.prompt_config}")
        prompt_cfg = load_yaml(args.prompt_config)
        prompt_manager = DeceptionPromptManager(**prompt_cfg)

        # Init Evaluator
        evaluator = Evaluator()

        # Init Pipeline
        pipeline = DeceptionInferencePipeline(
            llm=llm,
            prompt_manager=prompt_manager,
            output_dir=args.output_dir,
            max_workers=args.workers,
        )

    except Exception as e:
        logger.critical(f"Failed to initialize components: {e}", exc_info=True)
        sys.exit(1)

    output_path = os.path.join(args.output_dir, f"{args.run_name}.jsonl")
    output_path = output_path.replace("\\", "/")
    if not args.force:
        logger.info(f"Checking for existing progress in: {output_path}")
        processed_ids = get_processed_ids(output_path)

        if processed_ids:
            logger.info(f"Found {len(processed_ids)} processed records.")
            try:
                dataset.filter_by_ids(processed_ids)
            except Exception as e:
                logger.warning(f"Filtering dataset failed due to the following error {e}. \nUsing full dataset.")

            if len(dataset) == 0:
                logger.info("All data has been processed. Exiting.")
                sys.exit(0)
        else:
            logger.info("No existing records found. Starting from scratch.")
    else:
        logger.warning("Force mode enabled. Ignoring existing results.")

    try:
        logger.info(f"Evaluating start.")

        stats = pipeline.run(
            dataset=dataset,
            limit=args.limit,
            run_name=args.run_name
        )

        logger.info(f"Pipeline finished. Processed {stats['total_groups']} groups.")

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user.")
    except Exception as e:
        logger.critical(f"Pipeline failed during execution: {e}", exc_info=True)


if __name__ == "__main__":
    main()
