# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/3 12:19
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : paraphrase.py
# explain   :
from dataclasses import dataclass, field
from typing import List, Dict
import warnings

from panda_guard.role.defenses import BaseDefender, BaseDefenderConfig

from panda_guard.llms import BaseLLMConfig, LLMGenerateConfig, create_llm


@dataclass
class ParaphraseDefenderConfig(BaseDefenderConfig):
    """
    Configuration for the Paraphrase Defender.

    :param defender_cls: Class of the defender, default is "ParaphraseDefender".
    :param paraphrase_llm_config: Configuration for the paraphrasing language model.
    :param paraphrase_llm_gen_config: Configuration for generating output with the paraphrasing LLM.
    :param paraphrase_prompt: Template for generating paraphrasing prompts.
    """
    defender_cls: str = field(default="ParaphraseDefender")
    paraphrase_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    paraphrase_llm_gen_config: LLMGenerateConfig = field(default_factory=lambda: LLMGenerateConfig(max_n_tokens=512))
    paraphrase_prompt: str = field(default='paraphrase the following paragraph: \n"{prompt}"\n\n')



class ParaphraseDefender(BaseDefender):
    """
    Paraphrase Defender that reformulates user inputs to mitigate harmful content.

    Reference: Neel Jain, Avi Schwarzschild, Yuxin Wen, Gowthami Somepalli, John Kirchenbauer, Ping-yeh Chiang, Micah Goldblum, Aniruddha Saha, Jonas Geiping, and Tom Goldstein. 2023. Baseline defenses for adversarial attacks against aligned language models. arXiv preprint arXiv:2309.00614.

    :param config: Configuration for the Paraphrase Defender.
    """

    def __init__(self, config: ParaphraseDefenderConfig):
        super().__init__(config)
        self.paraphrase_llm = create_llm(config.paraphrase_llm_config)
        self.paraphrase_llm_gen_config = config.paraphrase_llm_gen_config
        self.paraphrase_prompt = config.paraphrase_prompt

    def defense(
            self,
            messages: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Execute the defense mechanism by paraphrasing the latest user input.

        :param messages: List of input messages.
        :return: Modified list of messages after applying the defense strategy.
        """
        assert messages, "Messages cannot be empty."

        prompt = messages[-1]['content']
        paraphrase_prompt = self._paraphrase(prompt)

        if "\n" in paraphrase_prompt:
            warnings.warn(
                """A \n character is found in the output of paraphrase model and the content after \n is removed.""")
            paraphrase_prompt = "\n".join(paraphrase_prompt.split('\n')[1:])

        messages[-1]['content'] = paraphrase_prompt

        return super().defense(messages)

    def _paraphrase(self, prompt: str) -> str:
        """
        Generate a paraphrased version of the given prompt.

        :param prompt: The original user prompt.
        :return: The paraphrased prompt.
        """
        paraphrase_prompt = self.paraphrase_prompt.format(prompt=prompt)

        output = self.paraphrase_llm.generate(
            [{"role": "user", "content": paraphrase_prompt}],
            self.paraphrase_llm_gen_config
        )[-1]['content']

        paraphrase_prompt = output.strip().strip(']').strip('[')

        return paraphrase_prompt
