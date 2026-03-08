# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 20:58
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : __init__.py.py
# explain   :


from panda_guard.role.judges.base import BaseJudge, BaseJudgeConfig

from panda_guard.utils import ComponentRegistry

judge_registry = ComponentRegistry[BaseJudge](
    "judge",
    BaseJudge,
    "panda_guard.judges"
)

config_registry = ComponentRegistry[BaseJudgeConfig](
    "judge_config",
    BaseJudgeConfig,
    "panda_guard.judge_configs"
)

def create_judge(config: BaseJudgeConfig) -> BaseJudge:
    return judge_registry.create_component(config)

def __getattr__(name):
    try:
        if name.endswith("Config"):
            return config_registry.get_component_class(name)
        else:
            return judge_registry.get_component_class(name)
    except ValueError:
        raise AttributeError(f"module 'panda_guard.role.judges' has no attribute '{name}'")