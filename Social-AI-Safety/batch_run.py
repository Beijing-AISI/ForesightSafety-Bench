import os
import glob
import time
import argparse
import logging
import subprocess
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [BATCH] - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--llm-configs", type=str, default="configs/llms")
    parser.add_argument("--prompt-configs", type=str, default="configs/prompts")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--max-parallel", type=int, default=16)

    return parser.parse_args()


def get_config_files(dir_path):
    if not os.path.exists(dir_path):
        logging.error(f"Directory {dir_path} does not exist")
        raise FileNotFoundError(f"Directory {dir_path} does not exist")
    search_path = os.path.join(dir_path, "*.yaml")
    files = glob.glob(search_path)
    if not files:
        logger.warning(f"No config files found in {dir_path}")
    files = [f.replace('\\', '/') for f in files]
    return files


def run_single_exp(llm_config, prompt_config, output_dir):
    llm_name = llm_config.split("/")[-1].lower().replace(".yaml", "")
    prompt_name = prompt_config.split("/")[-1].lower().replace(".yaml", "")
    run_name = f"{llm_name}_{prompt_name}"

    dataset_config = "configs/datasets/deceptioneval.yaml"
    # dataset_config = None
    # if 'zh' in prompt_name:
    #     dataset_config = "configs/datasets/cogtom-zh.yaml"
    # elif 'en' in prompt_name:
    #     dataset_config = "configs/datasets/cogtom-en.yaml"

    # llm_config_dict = yaml.safe_load(open(llm_config, "r"))
    # if "base_url" in llm_config_dict.keys():
    #     logger.warning(f"Require {llm_name} deployed using vllm. Skipping ...")
    #     return

    output_path = os.path.join(output_dir, f"{run_name}.jsonl")

    if os.path.exists(output_path):
        logger.warning(f"{run_name} already exists.")

    logger.info(f"Starting {run_name}...")

    cmd = [
        "python", "main.py",
        "--llm-config", llm_config,
        "--prompt-config", prompt_config,
        "--dataset-config", dataset_config,
        "--run-name", run_name,
        "--output-dir", output_dir,
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True)
        duration = time.time() - start_time
        logger.info(f"Finished {run_name} in {duration:.2f}s")

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed {run_name}")


def run(args):
    llms = get_config_files(args.llm_configs)
    prompts = get_config_files(args.prompt_configs)

    tasks = list(itertools.product(llms, prompts))

    logger.info(f"Found {len(llms)} LLMs, {len(prompts)} prompts.")
    logger.info(f"Total combinations to run: {len(tasks)}")

    with ThreadPoolExecutor(max_workers=args.max_parallel) as executor:
        futures = [
            executor.submit(run_single_exp, llm_config, prompt_config, args.output_dir)
            for llm_config, prompt_config in tasks
        ]

        for _ in as_completed(futures):
            pass

    logger.info(f"{len(futures)} tasks completed.")


def main():
    args = parse_args()
    run(args)
    return


if __name__ == "__main__":
    main()
