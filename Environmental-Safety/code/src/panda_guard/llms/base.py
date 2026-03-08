# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/8/31 22:00
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : base.py
# explain   :

import abc

from typing import Dict, List, Union, Any, Tuple
from dataclasses import dataclass, field
import concurrent.futures


@dataclass
class LLMGenerateConfig:
    """
    Configuration for LLM generation.

    :param max_n_tokens: Maximum number of tokens to generate.
    :param temperature: Temperature for sampling randomness.
    :param logprobs: Whether to return log probabilities.
    :param seed: Seed for reproducibility.
    :param stream: Whether to use streaming generation.
    """

    max_n_tokens: int = field(default=None)
    temperature: float = field(default=None)
    logprobs: bool = field(default=False)
    seed: int = field(default=None)
    stream: bool = field(default=False)  # Default to non-streaming behavior


@dataclass
class BaseLLMConfig(abc.ABC):
    """
    Base configuration for LLM.

    :param llm_type: Type of the LLM.
    :param model_name: Name of the model.
    """

    llm_type: str = field(default=None)
    model_name: str = field(default=None)


class BaseLLM(abc.ABC):
    """
    Abstract base class for LLM.

    :param config: Configuration object for LLM.
    """

    def __init__(self, config: BaseLLMConfig):
        self._CLS = config.llm_type
        self._NAME = config.model_name

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_requires = 0

    @abc.abstractmethod
    def generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Abstract method for generating response from LLM.

        :param messages: List of messages for input.
        :param config: Configuration for generation.
        :return: Generated response or responses with log probabilities.
        """
        pass

    @abc.abstractmethod
    def evaluate_log_likelihood(
        self,
        messages: List[Dict[str, str]],
        config: LLMGenerateConfig,
        require_grad=False,
    ) -> List[float]:
        """
        Abstract method for evaluating log likelihood of messages.

        :param messages: List of messages to evaluate.
        :param config: Configuration for generation.
        :param require_grad: Whether grad information is needed.
        :return: List of log likelihoods.
        """
        pass

    @abc.abstractmethod
    def continual_generate(
        self, messages: List[Dict[str, str]], config: LLMGenerateConfig
    ) -> Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]:
        """
        Remove EOS token in formatted prompt. Manually add generation prompt.

        :param messages: List of messages for input.
        :param config: Configuration for generation.
        :return: Generated response or responses with log probabilities.
        """
        pass

    def batch_generate(
            self,
            batch_messages: List[List[Dict[str, str]]],
            config: LLMGenerateConfig,
    ) -> List[Union[List[Dict[str, str]], Tuple[List[Dict[str, str]], List[float]]]]:
        """
        Generate responses for a batch of messages concurrently.

        :param batch_messages: List of batches of messages.
        :param config: Configuration for generation.
        :return: List of generated responses.
        """

        configs = [config] * len(batch_messages)

        if config.seed is not None:
            seeds = list(range(config.seed, config.seed + len(batch_messages)))
            for i, c in enumerate(configs):
                c.seed = seeds[i]

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.generate, messages, config)
                for config, messages in zip(configs, batch_messages)
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]
        return results

    def reset(self):
        """
        Reset the token counts and the number of requests.
        """
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.num_requires = 0

    def update(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        num_requires: int,
    ):
        """
        Update the token counts and number of requests.

        :param prompt_tokens: Number of tokens in prompt.
        :param completion_tokens: Number of tokens in completion.
        :param num_requires: Number of requests made.
        """
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.num_requires += num_requires

    @property
    def total_tokens(self):
        """
        Get the total number of tokens used.

        :return: Total number of tokens.
        """
        return self.prompt_tokens + self.completion_tokens

    @property
    def avg_tokens(self):
        """
        Get the average number of tokens per request.

        :return: Average number of tokens.
        """
        return self.total_tokens / self.num_requires
