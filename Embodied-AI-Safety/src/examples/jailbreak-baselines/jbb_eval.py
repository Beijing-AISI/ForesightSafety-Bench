# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/10/12 14:28
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : jbb_eval.py
# explain   :

import os
import json
from copy import deepcopy
import yaml
import argparse
import glob
import logging

from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict


# Load the YAML configuration file
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation pipeline")
    parser.add_argument('--config', type=str, default="../../configs/judges/judge.yaml", help='Path to YAML configuration file')
    parser.add_argument('--input-dir', type=str, default="../../benchmarks/jbb/", help='Input directory containing JSON files or YAML file containing list of files')
    parser.add_argument('--output-dir', type=str, default="../../benchmarks/jbb_judged/", help='Output directory')
    parser.add_argument('--num-workers', type=int, default=12, help='Number of workers for parallel processing')
    parser.add_argument('--log-level', type=str, default="WARNING", help='Logging level')
    return parser.parse_args()


def fill_llms_configs(d, parent_key='', llm_configs=None):
    for key, value in d.items():
        full_key = f'{parent_key}.{key}' if parent_key else key
        if isinstance(value, dict):
            fill_llms_configs(value, full_key, llm_configs)
        else:
            if key.endswith("llm_config") and not value:
                d[key] = llm_configs


def override_config(config_dict, args):
    # Override parameters from command line arguments
    fill_llms_configs(config_dict, "", llm_configs=config_dict["defender"]["target_llm_config"])
    return config_dict


def run_inference(pipe, messages, goal):
    result = pipe.parallel_judging(messages, goal)
    pipe.reset()
    return result


def process_file(json_file, args, attacker_config, defender_config, judge_configs, config_dict):
    # try:
    # Construct output file path
    generation_dict = yaml.safe_load(open(json_file.replace('results.json', 'config.yaml'), 'r'))
    config_to_save = generation_dict['judges'] = config_dict['judges']

    output_file = os.path.join(args.output_dir, os.path.relpath(json_file, args.input_dir))

    # Check if output file exists
    if os.path.exists(output_file):
        logging.warning(f"Output file {output_file} already exists, skipping.")
        return

    # Load data
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Initialize pipeline
    pipe = InferPipeline(
        InferPipelineConfig(
            attacker_config=attacker_config,
            defender_config=defender_config,
            judge_configs=judge_configs  # Can be 0, 1, or multiple judges
        ),
        verbose=False
    )

    if 'results' in data:
        data = data['results']

    # Process data
    for i, item in enumerate(tqdm(data, desc=json_file.split('/')[-4])):
        jailbroken = None
        goal = item['goal']

        for x in item['data']:
            messages = deepcopy(x['messages'])
            result = run_inference(pipe, messages, goal)
            x['judged'] = result

            if jailbroken is None:
                jailbroken = deepcopy(result)
            else:  # Select Max of each judge
                jailbroken = {k: max(jailbroken[k], result[k]) for k in jailbroken}

        item['jailbroken'] = jailbroken
        # print(item)

    # Save modified data to the output directory with the same relative file path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    yaml.dump(config_to_save, open(output_file.replace('results.json', 'config.yaml'), 'w'))
    with open(output_file, 'w') as f:
        json.dump({
            "config": config_to_save,
            "results": data
        }, f, indent=4)

    # except Exception as e:
    #     logging.error(f"Error processing file {json_file}: {e}")


def load_json_files_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        config = yaml.safe_load(file)
        return config.get('json_files', [])


def get_input_files(input_dir):
    if input_dir.endswith('.yaml') or input_dir.endswith('.yml'):
        logging.info(f"Input is a YAML file: {input_dir}")
        return load_json_files_from_yaml(input_dir)
    else:
        logging.info(f"Input is a directory: {input_dir}")
        return glob.glob(os.path.join(input_dir, '**', '*.json'), recursive=True)


if __name__ == '__main__':
    args = parse_args()
    # Setup logging
    logging.basicConfig(level=eval(f'logging.{args.log_level}'), format='%(asctime)s - %(levelname)s - %(message)s')

    # Load and possibly override the YAML configuration
    config_dict = load_config(args.config)
    config_dict = override_config(config_dict, args)

    # Convert YAML dictionary into attacker, defender, and judge configurations
    attacker_config, defender_config, judge_configs = parse_configs_from_dict(config_dict)

    # Get input files (from a directory or a YAML file)
    json_files = get_input_files(args.input_dir)

    # json_files = [f for f in json_files if 'NoneDefender' in f]
    # print(json_files)

    # Use ThreadPoolExecutor to process files concurrently
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = []
        for json_file in json_files:
            future = executor.submit(process_file, json_file, args, attacker_config, defender_config, judge_configs, config_dict)
            futures.append(future)

        # Display progress bar and handle any exceptions
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing files"):
            try:
                future.result()  # This will raise any exceptions caught in the threads
            except Exception as e:
                logging.error(f"Error in processing: {e}")

    logging.info(f"Results saved to {args.output_dir}")
