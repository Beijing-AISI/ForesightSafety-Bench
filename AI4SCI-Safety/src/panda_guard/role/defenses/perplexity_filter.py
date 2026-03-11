# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/3 22:39
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : perplexity_filter.py
# explain   :
from dataclasses import dataclass, field
from typing import List, Dict
import math

import numpy as np

from panda_guard.role.defenses import BaseDefender, BaseDefenderConfig, REJECT_RESPONSE

from panda_guard.llms import BaseLLMConfig, LLMGenerateConfig, create_llm
from panda_guard.utils import is_user_turn


@dataclass
class PerplexityFilterDefenderConfig(BaseDefenderConfig):
    """
    Configuration for the Perplexity Filter Defender.

    :param defender_cls: Class of the defender, default is "PerplexityFilterDefender".
    :param threshold: Threshold for perplexity to determine if a response should be rejected.
    :param filter_llm_config: Configuration for the language model used for filtering.
    :param filter_llm_gen_config: Configuration for generating output with the filtering LLM.
    """
    defender_cls: str = field(default="PerplexityFilterDefender")
    threshold: float = field(default=500)
    filter_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    filter_llm_gen_config: LLMGenerateConfig = field(default_factory=LLMGenerateConfig)



class PerplexityFilterDefender(BaseDefender):
    """
    Perplexity Filter Defender that rejects responses with high perplexity values.

    Reference: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang,
    Micah Goldblum, Aniruddha Saha, Jonas Geiping, Tom Goldstein. "Baseline Defenses for Adversarial Attacks Against Aligned Language Models." arXiv preprint arXiv:2309.00614.

    :param config: Configuration for the Perplexity Filter Defender.
    """

    def __init__(self, config: PerplexityFilterDefenderConfig):
        super().__init__(config)
        self.filter_llm = create_llm(config.filter_llm_config)
        self.filter_llm_gen_config = config.filter_llm_gen_config
        self.threshold = config.threshold

    def defense(
            self,
            messages: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Execute the defense mechanism by filtering responses based on perplexity.

        :param messages: List of input messages.
        :return: Modified list of messages after applying the defense strategy.
        """
        assert is_user_turn(messages), "It must be the user's turn to perform defense."

        # Evaluate log likelihoods for the messages
        logprobs = self.filter_llm.evaluate_log_likelihood(messages, self.filter_llm_gen_config)

        # Calculate perplexity
        perplexity = self._calculate_perplexity(logprobs)

        # Check if perplexity exceeds the threshold
        if perplexity > self.threshold:
            messages.append({"role": "assistant", "content": REJECT_RESPONSE})
            return messages

        return super().defense(messages)

    @staticmethod
    def _calculate_perplexity(logprobs: List[float]) -> float:
        """
        Calculate the perplexity based on the log probabilities.

        :param logprobs: List of log probabilities for the tokens.
        :return: The calculated perplexity value.
        """
        # Compute the average log probability
        avg_log_prob = np.mean(logprobs)

        # Calculate perplexity
        perplexity = math.exp(-avg_log_prob)
        return perplexity
