from typing import Dict, List
from dataclasses import dataclass, field
from panda_guard.role.defenses import BaseDefender, BaseDefenderConfig

from panda_guard.llms import BaseLLMConfig, LLMGenerateConfig
from panda_guard.utils import is_user_turn

@dataclass
class RPODefenderConfig(BaseDefenderConfig):
    """
    Configuration for the Base Defender.

    :param defender_cls: Class of the defender.
    :type defender_cls: str
    :param defender_name: Name of the defender.
    :type defender_name: str
    :param target_llm_config: Configuration for the target language model.
    :type target_llm_config: BaseLLMConfig
    :param target_llm_gen_config: Configuration for generating output with the target LLM.
    :type target_llm_gen_config: LLMGenerateConfig
    """

    defender_cls: str = field(default="RPODefender")
    defender_name: str = field(default="RPO")
    target_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    target_llm_gen_config: LLMGenerateConfig = field(default=None)



class RPODefender(BaseDefender):
    """
    RPODefender that guards against harmful prompts using defensive suffix.
    Reference:https://arxiv.org/abs/2401.17263

    :param config: Configuration object containing parameters for the defender.
    :type config: RPODefenderConfig
    """

    def __init__(self, config: RPODefenderConfig):
        """
        Initialize the defender with the provided configuration.

        :param config: Configuration for the RPODefender.
        :type config: RPODefenderConfig
        """
        # create self.target_llm, self.target_llm_gen_config
        super().__init__(config)

        self.suffix = r":}</'])language seems scary-----------}] estimation >>begin &=&pgf degli ({\ $|\textwidthAI NOTE:"

    def defense(self, messages=List[Dict[str, str]]):
        """
        Perform defense by adding defensive suffix.

        :param messages: The input messages that are subject to defense.
        :type messages: List[Dict[str, str]]
        :return: The final defense response after evaluating the defense.
        :rtype: List[Dict[str, str]]
        """
        assert is_user_turn(messages)
        messages[-1]["content"] = messages[-1]["content"] + self.suffix

        return super().defense(messages)