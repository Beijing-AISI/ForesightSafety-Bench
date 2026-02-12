# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/4 14:37
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : inference.py
# explain   : Inference pipeline for executing attack and defense phases.

import abc
from typing import Dict, List, Union, Any, Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from panda_guard.llms import BaseLLM
from panda_guard.role.attacks import create_attacker, BaseAttacker, BaseAttackerConfig
from panda_guard.role.attacks.transfer import TransferAttacker
from panda_guard.role.defenses import create_defender, BaseDefender, BaseDefenderConfig
from panda_guard.role.judges import create_judge, BaseJudge, BaseJudgeConfig


@dataclass
class InferPipelineConfig:
    """
    Configuration class for the inference pipeline.

    :param attacker_config: Configuration for the attacker, detailing its parameters and behavior.
    :param defender_config: Configuration for the defender, detailing its parameters and behavior.
    :param judge_configs: List of configurations for any judges used in evaluating the effectiveness of the defense.
    """

    attacker_config: BaseAttackerConfig = field(default=None)
    defender_config: BaseDefenderConfig = field(default=None)
    judge_configs: List[BaseJudgeConfig] = field(default_factory=list)


def llm_register(attr: Union[BaseAttacker, BaseDefender, BaseJudge]) -> List[BaseLLM]:
    """
    Register any LLM instances associated with the given attribute (attacker, defender, or judge).

    :param attr: An instance of an attacker, defender, or judge.
    :return: List of registered LLM instances.
    """
    results = []
    for name, value in vars(attr).items():
        if isinstance(value, BaseLLM):
            results.append(value)
    return results


class InferPipeline:
    """
    Inference pipeline for managing and executing attacks and defenses.

    :param config: Configuration for the inference pipeline, including attacker, defender, and judges.
    :param verbose: Whether to enable verbose logging for debugging and transparency.
    """

    def __init__(self, config: InferPipelineConfig, verbose: bool = False):
        # Create instances of the attacker, defender, and any judges based on the provided configurations.
        # 根据提供的配置创建攻击者、防御者和评审者的实例。
        self.attacker = create_attacker(config.attacker_config)
        self.defender = create_defender(config.defender_config)
        self.judges = [
            create_judge(judge_config) for judge_config in config.judge_configs
        ]
        # Register LLMs used by the attacker, defender, and judges.
        # 注册攻击者、防御者和评审者所使用的LLM。
        self.attack_llms = llm_register(self.attacker)
        self.defense_llms = llm_register(self.defender)
        self.judges_llms = [llm_register(judge) for judge in self.judges]
        self.verbose = verbose

    def __call__(self, messages: List[Dict[str, str]], **kwargs) -> Union[Dict[str, Any], Iterator[Dict[str, Any]]]:
        """
        Execute the full inference process, including attack and defense phases.

        :param messages: Input messages to be processed through the pipeline.
        :param kwargs: Additional keyword arguments, such as reformulated requests for transfer attacks.
        :return: A dictionary containing the final defended messages and the token usage details,
                 or an iterator yielding streaming chunks.
        """
        # Check if streaming is enabled in the defender's config
        is_streaming = getattr(self.defender.target_llm_gen_config, "stream", False)

        # Check for TransferAttacker and validate the required argument
        if isinstance(self.attacker, TransferAttacker):
            assert (
                    "request_reformulated" in kwargs
            ), "TransferAttacker requires a 'request_reformulated' parameter."

        self.log(messages, "[INPUT]")

        # Perform the attack phase using the attacker instance
        attack = self.attacker.attack(
            messages, request_reformulated=kwargs.get("request_reformulated", None)
        )
        self.log(attack, "[ATTACK]")

        # Perform the defense phase
        defense_result = self.defender.defense(attack)

        # Check if the result is an iterator (streaming) or regular messages
        if is_streaming and hasattr(defense_result, "__iter__") and not isinstance(defense_result, list):
            # It's a streaming iterator, yield chunks as they come
            def stream_generator():
                final_messages = None

                for chunk, current_messages in defense_result:
                    # Yield each chunk with the current state of messages
                    yield {
                        "chunk": chunk,
                        "messages": current_messages,
                        "complete": False
                    }
                    final_messages = current_messages

                # After all chunks are processed, yield a final complete message with usage
                if final_messages:
                    self.log(final_messages, "[DEFENSE]")
                    yield {
                        "messages": final_messages,
                        "complete": True,
                        "usage": self.calc_tokens()
                    }

            return stream_generator()

        else:
            # It's a regular result (non-streaming)
            self.log(defense_result, "[DEFENSE]")
            usage = self.calc_tokens()

            return {"messages": defense_result, "usage": usage}


    def parallel_judging(
        self, defense: List[Dict[str, str]], request: str
    ) -> Dict[str, Any]:
        """
        Perform parallel evaluation of the defense using multiple judges.

        :param defense: Messages that have been defended and need evaluation.
        :param request: The original request for context in evaluation.
        :return: A dictionary containing the results from each judge.
        """
        judge_results = {}

        # Use ThreadPoolExecutor for parallel judge evaluations.
        # 使用ThreadPoolExecutor进行评审者的并行评估。
        with ThreadPoolExecutor() as executor:
            future_to_judge = {
                executor.submit(judge.judge, defense, request): judge
                for judge in self.judges
            }
            for future in as_completed(future_to_judge):
                judge = future_to_judge[future]
                try:
                    result = future.result()
                    judge_results[judge._NAME] = result
                except Exception as exc:
                    print(f"Judge {judge._NAME} generated an exception: {exc}")

        return judge_results

    def calc_tokens(self) -> Dict[str, Dict[str, int]]:
        """
        Calculate the token usage for both the attacker and defender, summarizing prompt and completion tokens.

        :return: Dictionary containing detailed token usage for both roles.
        """
        # Calculate token usage for the attacker.
        # 计算攻击者的Token使用量。
        attacker_prompt_tokens = sum(llm.prompt_tokens for llm in self.attack_llms)
        attacker_completion_tokens = sum(
            llm.completion_tokens for llm in self.attack_llms
        )

        # Calculate token usage for the defender.
        # 计算防御者的Token使用量。
        defender_prompt_tokens = sum(llm.prompt_tokens for llm in self.defense_llms)
        defender_completion_tokens = sum(
            llm.completion_tokens for llm in self.defense_llms
        )

        return {
            "attacker": {
                "prompt_tokens": attacker_prompt_tokens,
                "completion_tokens": attacker_completion_tokens,
            },
            "defender": {
                "prompt_tokens": defender_prompt_tokens,
                "completion_tokens": defender_completion_tokens,
            },
        }

    def reset(self):
        """
        Reset the state of all LLMs in the pipeline, preparing them for a new inference run.
        """
        for llm in self.attack_llms + self.defense_llms:
            llm.reset()

    def log(self, messages: List[Dict[str, str]], stage: str = None):
        """
        Log messages for each stage of the pipeline if verbose logging is enabled.

        :param messages: The messages to be logged at the current stage.
        :param stage: A label for the current stage of the pipeline.
        """
        if self.verbose:
            pass
            # print(stage)
            # print(messages)
            # print()
