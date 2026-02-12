# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 15:04
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : attacker_registry.py
# explain   :

from typing import TypeVar
from panda_guard.role.judges import BaseJudge, BaseJudgeConfig

JUDGES: dict[str, type[BaseJudge]] = {}

T = TypeVar("T", bound=BaseJudge)


def register_judge(judge: type[T]) -> type[T]:
    """
    Register a judge subclass with the registry.

    :param judge: The judge class to be registered.
    :return: The registered judge class.
    """
    JUDGES[judge.__name__] = judge
    return judge


def create_judge(config: BaseJudgeConfig) -> BaseJudge:
    """
    Create an instance of the registered judge class based on the given configuration.

    :param config: Configuration for the judge.
    :return: An instance of the judge.
    """
    return JUDGES[config.judge_cls](config)
