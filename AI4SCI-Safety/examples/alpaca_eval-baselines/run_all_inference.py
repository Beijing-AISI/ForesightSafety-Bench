# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/16 20:37
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : run_all_inference.py
# explain   : Iterate over all LLM and defense configurations and run Alpaca evaluation inference for each combination.

import os
import glob
import subprocess
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from queue import Queue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run all experiments for different LLM and defense configurations."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../../configs/tasks/alpaca_eval.yaml',
        help='Path to the main configuration file'
    )
    parser.add_argument(
        '--defense',
        type=str,
        default="../../configs/defenses/",
        help='Path to a single defense configuration file or directory containing multiple defense configuration files'
    )
    parser.add_argument(
        '--llm',
        type=str, default="../../configs/defenses/llms/",
        help='Path to a single LLM configuration file or directory containing multiple LLM configuration files'
    )
    parser.add_argument(
        '--llm-gen',
        type=str,
        default='../../configs/defenses/llm_gen/alpaca_eval.yaml',
        help='Path to the LLM generator file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="../../benchmarks/alpaca_eval",
        help='Output directory'
    )
    parser.add_argument(
        '--device-prefix',
        type=str,
        default='cuda:',
        help='Device prefix for GPUs (e.g., "cuda:")'
    )
    parser.add_argument(
        '--max-parallel',
        type=int,
        default=8,
        help='Maximum number of parallel processes'
    )
    parser.add_argument(
        '--include-revised',
        action='store_true',
        help='Include revised files from the same directories'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default="WARNING",
        help="Logging level, e.g., DEBUG, INFO, WARNING, ERROR"
    )
    return parser.parse_args()


def get_config_files(path, include_revised):
    """Get a list of .yaml files from a path, handling single files or directories with wildcards."""
    if path.endswith('.yaml'):
        return glob.glob(path)
    elif os.path.isdir(path):
        yaml_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.yaml')]
        if include_revised:
            revised_path = os.path.join(path, 'revised')
            if os.path.exists(revised_path):
                yaml_files += [os.path.join(revised_path, f) for f in os.listdir(revised_path) if f.endswith('.yaml')]
        return yaml_files
    else:
        raise ValueError(f"Path {path} is not a valid .yaml file or directory.")


def run_all_experiments(args):
    logging.basicConfig(level=eval(f'logging.{args.log_level}'), format='%(asctime)s - %(levelname)s - %(message)s')

    llm_files = get_config_files(args.llm, args.include_revised)
    defense_files = get_config_files(args.defense, args.include_revised)

    logging.info(f"Found {len(llm_files)} LLM configurations, {len(defense_files)} defense configurations.")

    # Prepare combinations of (llm, defense).
    tasks = [(llm_file, defense_file) for defense_file in defense_files for llm_file in llm_files if 'vllm_' not in llm_file]
    tasks = list(reversed(tasks))
    # Prepare a queue with available devices.
    # max_parallel = min(args.max_parallel, len(llm_files))
    max_parallel = args.max_parallel
    device_queue = Queue()
    for i in range(max_parallel):
        device_queue.put(f"{args.device_prefix}{i}")

    def run_task(llm_file, defense_file):
        device = device_queue.get()  # Get an available device from the queue.
        command = [
            "python", "alpaca_inference.py",
            "--config", args.config,
            "--llm", llm_file,
            "--defense", defense_file,
            "--output-dir", args.output_dir,
            "--llm-gen", args.llm_gen,
            "--visible",
            # "--device", device,
        ]
        try:
            print(f"Running command: {' '.join(command)} on {device}")
            subprocess.run(command, check=True)
            return True, command
        except subprocess.CalledProcessError:
            logging.error(f"Command failed: {' '.join(command)}")
            return False, command
        finally:
            device_queue.put(device)  # Put the device back into the queue when the task is done.

    successes, failures = [], []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [executor.submit(run_task, llm_file, defense_file) for llm_file, defense_file in tasks]
        for future in tqdm(futures, total=len(tasks), desc="Running Alpaca evaluation experiments"):
            success, command = future.result()
            if success:
                successes.append(" ".join(command))
            else:
                failures.append(" ".join(command))

    # Write failures to a file.
    with open("fails.txt", "w", encoding="utf-8") as f:
        for command in failures:
            f.write(f"Failed: {command}\n")

    logging.info(f"Experiments completed. Successes: {len(successes)}, Failures: {len(failures)}.")
    logging.info("Failed experiments logged in fails.txt.")


if __name__ == '__main__':
    args = parse_args()
    run_all_experiments(args)
