# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 20:57
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : __init__.py.py
# explain   :

from panda_guard.role.attacks.base import BaseAttacker, BaseAttackerConfig

from panda_guard.utils import ComponentRegistry

attacker_registry = ComponentRegistry[BaseAttacker](
    "attacker",
    BaseAttacker,
    "panda_guard.attackers"
)

config_registry = ComponentRegistry[BaseAttackerConfig](
    "attacker_config",
    BaseAttackerConfig,
    "panda_guard.attacker_configs"
)

def create_attacker(config: BaseAttackerConfig) -> BaseAttacker:
    return attacker_registry.create_component(config)


def __getattr__(name):
    try:
        if name.endswith("Config"):
            return config_registry.get_component_class(name)
        else:
            return attacker_registry.get_component_class(name)
    except ValueError:
        raise AttributeError(f"module 'panda_guard.role.attacks' has no attribute '{name}'")