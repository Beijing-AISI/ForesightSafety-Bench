# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/16 19:46
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : jbb_inference.py
# explain   : Run inference using a specific attack and defense configuration.

import os
import json
from typing import Dict, Any
import warnings
import pandas as pd
import yaml
import argparse
import logging
from tqdm import tqdm

from panda_guard.llms import create_llm, HuggingFaceLLMConfig
from panda_guard.pipelines.inference import InferPipeline, InferPipelineConfig
from panda_guard.utils import parse_configs_from_dict

def normalize_cell(x):
    if pd.isna(x):
        return None
    if isinstance(x, str) and x.strip() == "":
        return None
    return x


def run_inference(config_dict):
    """Run inference using the provided configurations."""
    logging.basicConfig(
        level=eval(f"logging.{config_dict['args']['log_level']}"),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Load the main configuration and override attack/defense paths.
    # config_dict = load_yaml(args.config)

    # attack_dict = load_yaml(args.attack)
    # if attack_dict:
    #     config_dict["attacker"] = attack_dict
    # defense_dict = load_yaml(args.defense)
    # if defense_dict:
    #     config_dict["defender"] = defense_dict

    if config_dict["args"]["visible"]:
        logging.info(f"Loaded configuration: {config_dict}")

    llm_dict = config_dict["llm"]
    llm_gen_dict = config_dict["llm_gen"]

    if llm_dict:
        config_dict["defender"]["target_llm_config"] = llm_dict
    else:
        llm_dict = config_dict["defender"]["target_llm_config"]

    if config_dict["args"]["device"] is not None:
        config_dict["defender"]["target_llm_config"]["device_map"] = config_dict[
            "args"
        ]["device"]
        logging.WARNING("Override defender device mapping")

    if llm_gen_dict:
        config_dict["defender"]["target_llm_gen_config"].update(llm_gen_dict)

    # Update output file path.
    # print(config_dict)
    output_file = os.path.join(
        config_dict["output_dir"],
        llm_dict["model_name"].replace("/", "_"),
        f'{config_dict["attacker"]["attacker_cls"]}_{config_dict["attacker"]["attacker_name"]}',
        f'{config_dict["defender"]["defender_cls"]}',
        "results.json",
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        logging.warning(f"Output file {output_file} already exists. Skipping.")
        return

    # Load and parse attacker and defender configurations.
    attacker_config, defender_config, _ = parse_configs_from_dict(config_dict)

    # Load CSV input file.
    df = pd.read_csv(config_dict["misc"]["input_file"], encoding="utf-8")

    yaml.dump(
        config_dict, open(output_file.replace("results.json", "config.yaml"), "w")
    )

    # Initialize the pipeline.
    pipe = InferPipeline(
        InferPipelineConfig(
            attacker_config=attacker_config, defender_config=defender_config
        ),
        verbose=False,
    )

    # Run inference for each row.
    df = (
        df.head(config_dict["args"]["max_queries"])
        if config_dict["args"]["max_queries"]
        else df
    )
    # print(df)
    iterator = (
        tqdm(df.iterrows(), total=len(df), desc=config_dict["llm"]["model_name"])
        if config_dict["args"]["visible"]
        else df.iterrows()
    )
    results = []
    for _, row in iterator:
        goal = normalize_cell(row.get("Goal"))

        # Skip rows with empty Goal
        if goal is None:
            continue
        messages = [{"role": "user", "content": goal}]
        result = [
            pipe(
                messages,
                request_reformulated=row.get(attacker_config.attacker_name, None),
            )
            for _ in range(config_dict["args"]["repeats"])
        ]
        results.append({"goal": row["Goal"], "data": result})
        pipe.reset()

    # Save results to the output file.
    with open(output_file, "w") as outfile:
        json.dump({"config": config_dict, "results": results}, outfile, indent=4)

    logging.info(f"Results saved to {output_file}")
