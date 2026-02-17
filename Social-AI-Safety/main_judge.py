import argparse
import logging
import yaml
import sys
import os
from dotenv import load_dotenv

from src.llm import create_llm
from src.evaluator import DeceptionEvaluator
from src.pipeline import DeceptionJudgePipeline

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


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


def main():
    parser = argparse.ArgumentParser(description="Run LLM-as-a-Judge")
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results_judged")
    parser.add_argument("--judge-config", type=str, default="configs/judge.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)

    args = parser.parse_args()

    run_name = args.input_file.replace('\\', '/').split("/")[-1].replace(".jsonl","")[0] \
        if args.run_name is None else args.run_name
    log_name = f"{run_name}_judging.log"
    setup_logging(log_name)

    logger.info("Loading configurations...")
    judge_cfg = load_yaml(args.judge_config)

    llm = create_llm(**judge_cfg["judge_llm"], gen_config=judge_cfg["judge_gen_config"])
    logger.info(f"Judge Model Initialized: {llm.model_name}")

    evaluator = DeceptionEvaluator(
        judge_llm=llm,
        correctness_config=judge_cfg["correctness_judge"],
        sycophancy_config=judge_cfg["sycophancy_judge"],
        truthfulness_config=judge_cfg["truthfulness_judge"],
        manipulation_config=judge_cfg["manipulation_judge"],
        feint_config=judge_cfg["feint_judge"],
        bluffing_config=judge_cfg["bluffing_judge"],
    )

    pipeline = DeceptionJudgePipeline(
        evaluator=evaluator,
        output_dir=args.output_dir,
        max_workers=args.workers
    )

    pipeline.run(
        input_file=args.input_file,
        run_name=run_name,
    )


if __name__ == "__main__":
    main()
