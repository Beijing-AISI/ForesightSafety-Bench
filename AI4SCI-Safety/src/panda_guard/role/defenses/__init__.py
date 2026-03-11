# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 20:57
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : __init__.py.py
# explain   :


from panda_guard.role.defenses.base import (
    BaseDefender,
    BaseDefenderConfig,
    REJECT_RESPONSE,
)

from panda_guard.utils import ComponentRegistry

defender_registry = ComponentRegistry[BaseDefender](
    "defender",
    BaseDefender,
    "panda_guard.defenders"
)

config_registry = ComponentRegistry[BaseDefenderConfig](
    "defender_config",
    BaseDefenderConfig,
    "panda_guard.defender_configs"
)

def create_defender(config: BaseDefenderConfig) -> BaseDefender:
    return defender_registry.create_component(config)

def __getattr__(name):
    try:
        if name.endswith("Config"):
            return config_registry.get_component_class(name)
        else:
            if name == "RepeDefender":
                from panda_guard.role.defenses.repe import RepeDefender
                from panda_guard.role.defenses.repe_utils import repe_pipeline_registry
                repe_pipeline_registry()
                return RepeDefender
            return defender_registry.get_component_class(name)
    except ValueError:
        raise AttributeError(f"module 'panda_guard.role.defenses' has no attribute '{name}'")