# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/1 17:34
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : utils.py
# explain   : Utility functions for the panda-guard.

import abc
import subprocess
import sys
import time
from typing import Tuple, List, Dict, Type, TypeVar, Generic, Optional, Any
import importlib
from importlib.metadata import entry_points

import yaml


def is_user_turn(messages: List[Dict[str, str]]) -> bool:
    """
    Check if it's the user's turn based on the last message.

    :param messages: List of message dictionaries containing "role" and "content".
    :return: True if the last message is from the user, False otherwise.
    """
    return messages and len(messages) > 0 and messages[-1]["role"] == "user"


def is_assistant_turn(messages: List[Dict[str, str]]) -> bool:
    """
    Check if it's the assistant's turn based on the last message.

    :param messages: List of message dictionaries containing "role" and "content".
    :return: True if the last message is from the assistant, False otherwise.
    """
    return messages and len(messages) > 0 and messages[-1]["role"] == "assistant"


def load_class(config_cls_name: str, role_type: str) -> Any:
    """
    Dynamically load a class based on its name.

    :param config_cls_name: The name of the class to load.
    :param role_type: The type of role (e.g., "attacker", "defender", "judge").
    :return: The class object corresponding to the given name.
    """
    module = importlib.import_module(f"panda_guard.role.{role_type}")
    return getattr(module, config_cls_name + "Config")


def load_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


def parse_nested_config(config_cls, config_dict: Dict[str, Any]):
    """
    Parse nested configuration dictionaries into objects.

    :param config_cls: The class of the configuration.
    :param config_dict: Dictionary containing configuration data.
    :return: An instance of the configuration class.
    """
    module = importlib.import_module("panda_guard.llms")
    nested_config_dict = {}
    for key, value in config_dict.items():
        if "llm_gen_config" in key:
            nested_config_dict[key] = getattr(module, "LLMGenerateConfig")(**value)
        elif "llm_config" in key:
            nested_config_dict[key] = getattr(module, value["llm_type"] + "Config")(
                **value
            )
        else:
            nested_config_dict[key] = value
    return config_cls(**nested_config_dict)


def parse_configs_from_dict(config_dict: Dict[str, Any], return_dict: bool=False):
    """
    Convert a dictionary into configuration objects for the pipeline.

    :param config_dict: Dictionary containing configurations for attacker, defender, and judges.
    :return: A tuple containing the attacker configuration, defender configuration, and list of judge configurations.
    """
    # Load the attacker configuration.
    # 加载攻击者配置。
    attacker_config_dict = config_dict.get("attacker", {})
    AttackerClass = load_class(attacker_config_dict.get("attacker_cls"), "attacks")
    attacker_config = parse_nested_config(AttackerClass, attacker_config_dict)

    # Load the defender configuration.
    # 加载防御者配置。
    defender_config_dict = config_dict.get("defender", {})
    DefenderClass = load_class(defender_config_dict.get("defender_cls"), "defenses")
    defender_config = parse_nested_config(DefenderClass, defender_config_dict)

    # Load the judge configurations.
    # 加载评审配置。
    judge_configs = []
    judge_config_dicts = config_dict.get("judges", [])
    if judge_config_dicts is not None:
        for judge_config_dict in judge_config_dicts:
            JudgeClass = load_class(judge_config_dict.get("judge_cls"), "judges")
            judge_config = parse_nested_config(JudgeClass, judge_config_dict)
            judge_configs.append(judge_config)

    if return_dict:
        return {
            "attacker_config": attacker_config,
            "defender_config": defender_config,
            "judge_configs": judge_configs,
        }
    return attacker_config, defender_config, judge_configs


def get_gpu_memory_usage(device: str) -> Tuple[int, int, int]:
    """
    Get the total, used, and free GPU memory for a specified device.

    :param device: The GPU device identifier (e.g., "cuda:0").
    :return: A tuple containing the total, used, and free memory in MB.
    """
    gpu_id = device.split(":")[-1]
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,nounits,noheader",
            "--id=" + gpu_id,
        ],
        stdout=subprocess.PIPE,
        encoding="utf-8",
    )

    # Parse the output from nvidia-smi.
    # 解析nvidia-smi的输出。
    output = result.stdout.strip()
    total_mem, used_mem, free_mem = map(int, output.split(", "))
    return total_mem, used_mem, free_mem


def wait_for_gpu_memory(device: str, threshold: float = 0.8, check_interval: int = 5):
    """
    Wait until the specified GPU has sufficient free memory.

    :param device: The GPU device identifier (e.g., "cuda:0").
    :param threshold: The threshold of free memory ratio to proceed (e.g., 0.8 means 80% free).
    :param check_interval: Time interval (in seconds) between checks.
    """
    while True:
        total_mem, used_mem, free_mem = get_gpu_memory_usage(device)
        free_ratio = free_mem / total_mem

        print(
            f"GPU {device}: Total: {total_mem}MB, Used: {used_mem}MB, Free: {free_mem}MB ({free_ratio * 100:.2f}% free)"
        )

        if free_ratio >= threshold:
            print(
                f"GPU {device} has more than {threshold * 100}% free memory. Proceeding..."
            )
            break

        print(
            f"Waiting for GPU {device} to have more than {threshold * 100}% free memory..."
        )
        time.sleep(check_interval)


def process_end_eos(msg: str, eos_token: str):
    """
    Processes the end of a message by removing any trailing newline characters or EOS (End of Sequence) tokens.

    This function ensures that the message doesn't end with unwanted newline or EOS tokens, which might
    interfere with further processing or analysis.

    :param msg: The input message string that needs to be processed.
    :param eos_token: The EOS (End of Sequence) token to be removed, if it exists at the end of the message.
    :return: The processed message with trailing newline and EOS token removed, if any.
    """
    if msg.endswith("\n"):
        msg = msg[:-1]
    if msg.endswith(eos_token):
        msg = msg[: -len(eos_token)]

    return msg


T = TypeVar('T')

class ComponentRegistry(Generic[T]):
    """
    A registry for discovering, loading, and instantiating components via entry points.

    :param component_type: The type name of the component.
    :param base_class: The expected base class or interface that all registered components should inherit from.
    :param entry_point_group: The name of the entry point group to search for available components.
    """
    def __init__(self,
                 component_type: str,
                 base_class: Type[T],
                 entry_point_group: str
                 ):
        self.component_type = component_type
        self.base_class = base_class
        self.entry_point_group = entry_point_group
        self._components: Dict[str, Any] = {}
        self._loaded = False

    def _ensure_loaded(self):
        """
        Ensure that components have been discovered and loaded.
        """
        if not self._loaded:
            self._discover_components()
            self._loaded = True

    def _discover_components(self):
        """
        Discover available components using Python's entry point mechanism.
        """
        try:
            try:
                from importlib.metadata import entry_points
                eps = list(entry_points(group=self.entry_point_group))
            except (ImportError, TypeError):
                from pkg_resources import iter_entry_points
                eps = list(iter_entry_points(group=self.entry_point_group))

            for ep in eps:
                try:
                    self._components[ep.name] = ep
                except Exception as e:
                    import traceback
                    traceback.print_exc()
        except Exception as e:
            import traceback
            traceback.print_exc()

    def get_component_class(self, name: str) -> Type[T]:
        """
        Retrieve and load the class for a registered component by name.

        :param name: The name of the component to retrieve.
        :return: The loaded component class.
        :raises ValueError: If the component name is not found in the registry.
        """
        self._ensure_loaded()

        if name not in self._components:
            raise ValueError(f"Unknown {self.component_type}: {name}")
        if hasattr(self._components[name], 'load'):
            self._components[name] = self._components[name].load()

        return self._components[name]

    def create_component(self, config: Any) -> T:
        """
        Instantiate a component using the provided configuration.

        :param config: Configuration object that includes the component class name.
        :return: An instance of the component initialized with the provided config.
        """
        component_cls_name = getattr(config, f"{self.component_type}_cls")
        component_cls = self.get_component_class(component_cls_name)
        return component_cls(config)
