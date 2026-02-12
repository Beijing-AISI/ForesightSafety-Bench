# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/16 19:46
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : run_all_inference.py
# explain   : Iterate over all attack, defense, and LLM configurations and run inference for each combination.

import os
import subprocess
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any

import yaml
from tqdm import tqdm
from queue import Queue


def load_yaml(yaml_file) -> Dict[str, Any]:
    """Load a YAML file."""
    if yaml_file is None or not os.path.exists(yaml_file):
        return {}
    with open(yaml_file, "r") as file:
        return yaml.safe_load(file)


def parse_args():
    """Parse command line arguments for running all experiments."""
    parser = argparse.ArgumentParser(
        description="Run all experiments for different attack, defense, and LLM configurations."
    )
    parser.add_argument(
        '--config',
        type=str,
        default='../../configs/tasks/jbb.yaml',
        help='Path to the main configuration file'
    )
    parser.add_argument(
        '--llm',
        type=str, default="../../configs/defenses/llms/",
        help='Path to a single LLM configuration file or directory containing multiple LLM configuration files'
    )
    parser.add_argument(
        '--attack',
        type=str, default="../../configs/attacks/transfer",
        help='Path to a single attack configuration file or directory containing multiple attack configuration files'
    )
    parser.add_argument(
        '--defense',
        type=str,
        default="../../configs/defenses/",
        help='Path to a single defense configuration file or directory containing multiple defense configuration files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default="../../benchmarks/jbb",
        help='Output directory'
    )
    parser.add_argument(
        '--llm-gen',
        type=str,
        default='../../config/defenses/llm_gen/jbb.yaml',
        help='Path to the LLM generator file'
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
        help='Maximum number of parallel processes (should not exceed available GPUs)'
    )
    parser.add_argument(
        '--include-reserved',
        action='store_true',
        help='Include reserved files from the same directories'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default="WARNING",
        help="Logging level, e.g., DEBUG, INFO, WARNING, ERROR"
    )
    return parser.parse_args()


def get_config_files(path, include_reserved):
    """Get a list of .yaml files from a path, handling single files or directories."""
    if path.endswith('.yaml'):
        return [path]
    elif os.path.isdir(path):
        yaml_files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.yaml')]
        if include_reserved:
            reserved_path = os.path.join(path, 'reserved')
            if os.path.exists(reserved_path):
                yaml_files += [os.path.join(reserved_path, f) for f in os.listdir(reserved_path) if f.endswith('.yaml')]
        return yaml_files
    else:
        raise ValueError(f"Path {path} is neither a .yaml file nor a directory.")


def run_all_experiments(args):
    """Iterate over all LLM, attack, and defense configurations and run inference for each combination."""
    logging.basicConfig(level=eval(f'logging.{args.log_level}'), format='%(asctime)s - %(levelname)s - %(message)s')

    # Get all LLM, attack, and defense configuration files.
    llm_files = get_config_files(args.llm, args.include_reserved)
    attack_files = get_config_files(args.attack, args.include_reserved)
    defense_files = get_config_files(args.defense, args.include_reserved)

    logging.info(
        f"Found {len(llm_files)} LLM configurations, {len(attack_files)} attack configurations, and {len(defense_files)} defense configurations.")

    # Validate max parallel jobs against the number of available GPUs.
    max_parallel = args.max_parallel
    device_list = [f"{args.device_prefix}{i}" for i in range(max_parallel)]

    print(f"Found {len(device_list)} devices: {device_list}")

    # Prepare a queue with available devices.
    device_queue = Queue()
    for device in device_list:
        device_queue.put(device)

    # Prepare combinations of (llm, attack, defense).
    tasks_initial = [
        (llm_file, attack_file, defense_file)
        for attack_file in attack_files
        for defense_file in defense_files
        # for llm_file in llm_files if 'vllm' not in llm_file and 'claude' not in llm_file
        for llm_file in llm_files if 'vllm' not in llm_file # or 'claude' in llm_file
        # for llm_file in llm_files if '8k' in llm_file # or 'claude' in llm_file
    ]

    tasks = []
    for llm_file, attack_file, defense_file in tasks_initial:
        llm_dict = load_yaml(llm_file)
        llm_gen_dict = load_yaml(args.llm_gen)

        config_dict = load_yaml(args.config)
        config_dict["attacker"] = load_yaml(attack_file)
        config_dict["defender"] = load_yaml(defense_file)

        if llm_dict:
            config_dict["defender"]["target_llm_config"] = llm_dict
        else:
            llm_dict = config_dict["defender"]["target_llm_config"]

        if llm_gen_dict:
            config_dict["defender"]["target_llm_gen_config"].update(llm_gen_dict)

        output_file = os.path.join(
            args.output_dir,
            llm_dict["model_name"].replace("/", "_"),
            f'{config_dict["attacker"]["attacker_cls"]}_{config_dict["attacker"]["attacker_name"]}',
            f'{config_dict["defender"]["defender_cls"]}',
            "results.json",
        )

        if not os.path.exists(output_file):
            tasks.append((llm_file, attack_file, defense_file))

        else:
            logging.warning(f"Output file {output_file} already exists. Skipping.")

    print(f"Total {len(tasks_initial)} Tasks, {len(tasks)} to run.")

    # Function to run inference for a specific combination.
    def run_task(llm_file, attack_file, defense_file):
        device = device_queue.get()  # Get an available device from the queue.
        command = [
            "python", "jbb_inference.py",
            "--config", args.config,
            "--llm", llm_file,
            "--attack", attack_file,
            "--defense", defense_file,
            "--llm-gen", args.llm_gen,
            # "--device", device,
            "--output-dir", args.output_dir,
            "--visible",
        ]

        try:
            print(f"Running command: {' '.join(command)} on {device}")
            subprocess.run(command, check=True)
            return True, command
        except subprocess.CalledProcessError:
            logging.error(f"Command failed: {' '.join(command)}")
            return False, command  # Indicate failure.
        finally:
            device_queue.put(device)  # Put the device back into the queue when the task is done.

    # Use ThreadPoolExecutor for parallel execution.
    successes = []
    failures = []
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = [
            executor.submit(run_task, llm_file, attack_file, defense_file)
            for llm_file, attack_file, defense_file in tasks
        ]

        # Display progress using tqdm.
        for future in tqdm(futures, total=len(tasks), desc="Running experiments"):
            success, command = future.result()
            if success:
                successes.append(" ".join(command))
            else:
                failures.append(" ".join(command))

    # Write failures to a file.
    with open("fails.txt", "w", encoding="utf-8") as f:
        for command in failures:
            f.write(f"Failed: {command}\n")

    logging.info(f"All experiments completed. Successes: {len(successes)}, Failures: {len(failures)}.")
    logging.info("Failed experiments logged in fails.txt.")


if __name__ == '__main__':
    args = parse_args()
    run_all_experiments(args)
