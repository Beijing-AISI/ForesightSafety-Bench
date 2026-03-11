import os
import sys

sys.path.append(os.getcwd())

from omegaconf import OmegaConf
import click
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from tqdm import tqdm
from examples.exp_scripts.jbb_inference_exp import run_inference
import json


def build_combined_configs(config):
    config_list = []

    if config["external"] is not None:
        for cfg in config["external"]:
            config_list.append(
                OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)
            )

    if config["manual"] is not None:
        for cfg in config["manual"]:
            config_list.append(cfg["params"])

    return config_list


def transform_tasks(tasks, main_config):
    task_configs = []
    output_dir = os.path.join("./outputs", main_config["exp_prefix"])
    for attacker, defender, llm, data in tasks:
        cfg = copy.deepcopy(main_config)
        cfg["attacker"] = attacker
        cfg["defender"] = defender
        cfg["llm"] = llm
        cfg["misc"] = {}
        cfg["misc"]["input_file"] = data
        cfg["output_dir"] = output_dir
        task_configs.append(cfg)
    return task_configs


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to YAML config file",
)
def main(config):
    config = OmegaConf.load(config)
    config = OmegaConf.to_container(config)  # to dict

    logging.basicConfig(
        level=eval(f"logging.{config['args']['log_level']}"),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    log_dir = os.path.join('logs',config['exp_prefix'])
    os.makedirs(log_dir, exist_ok=True)

    attacker_list = build_combined_configs(config=config["attacker"])
    defender_list = build_combined_configs(config=config["defender"])
    llm_list = build_combined_configs(config=config["llm"])
    if len(llm_list) == 0:
        llm_list = [None]
    data_list = config["input_file"]

    tasks = [
        (attacker, defender, llm, data)
        for attacker in attacker_list
        for defender in defender_list
        for llm in llm_list
        for data in data_list
    ]

    task_configs = transform_tasks(tasks=tasks, main_config=config)

    def run_task(task_cfg):
        try:
            run_inference(task_cfg)
            return True, task_cfg
        except Exception as e:
            logging.INFO(f"run inference failed: {e}")
            return False, task_cfg

    successes = []
    failures = []
    with ThreadPoolExecutor(max_workers=config["args"]["max_parallel"]) as executor:
        futures = [executor.submit(run_task, task_cfg) for task_cfg in task_configs]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Running experiments"):
            try:
                success, cfg = future.result()
            except Exception:
                # If run_task unexpectedly raised, treat as failure
                logging.exception("Task raised exception")
                success, cfg = False, None  # or store the original cfg via a map (see below)

            if success:
                successes.append(cfg)
            else:
                failures.append(cfg)

    logging.info(
        f"All experiments completed. Successes: {len(successes)}, Failures: {len(failures)}."
    )


    # write failed
    fail_log_path = os.path.join(log_dir, "fails.json")

    failures_dict = {
        f"failure #{i}": cfg
        for i, cfg in enumerate(failures, start=1)
    }

    with open(fail_log_path, "w", encoding="utf-8") as f:
        json.dump(failures_dict, f, indent=2, ensure_ascii=False)

    logging.info(f"Failed experiments logged in {fail_log_path}.")


if __name__ == "__main__":
    main()
