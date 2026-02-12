# encoding: utf-8
# Author    : Floyed<Floyed_Shen@outlook.com>
# Datetime  : 2024/9/2 21:05
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : semantic_smoothllm.py
# explain   :
import os
from typing import Dict, List
import random
import json

from dataclasses import dataclass, field

import numpy as np

from panda_guard.role.defenses import BaseDefender, BaseDefenderConfig

from panda_guard.utils import is_user_turn
from panda_guard.llms import BaseLLMConfig, LLMGenerateConfig, create_llm


@dataclass
class SemanticSmoothLLMDefenderConfig(BaseDefenderConfig):
    """
    Configuration for SemanticSmoothLLMDefender.

    :param defender_cls: Class of the defender, default is "SemanticSmoothLLMDefender".
    :param perturbation_type: Type of perturbation to apply, default is 'random'.
    :param num_samples: Number of perturbed samples to generate, default is 3.
    :param batch_size: Batch size for processing, default is 1.
    :param perturbation_llm_config: Configuration for the LLM used for perturbation generation.
    :param perturbation_llm_gen_config: Configuration for generation settings for the perturbation LLM.
    """
    defender_cls: str = field(default="SemanticSmoothLLMDefender")
    perturbation_type: str = field(default='random')
    num_samples: int = field(default=3)
    batch_size: int = field(default=1)
    perturbation_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    perturbation_llm_gen_config: LLMGenerateConfig = field(default_factory=lambda: LLMGenerateConfig(max_n_tokens=300))



class SemanticSmoothLLMDefender(BaseDefender):
    """
    SemanticSmoothLLMDefender applies semantic smoothing to defend against jailbreak attacks.

    Based on "Smoothllm: Defending large language models against jailbreaking attacks" by Robey et al. (2023).
    Paper link: https://arxiv.org/abs/2310.03684

    :param config: Configuration for SemanticSmoothLLMDefender.
    """

    def __init__(self, config: SemanticSmoothLLMDefenderConfig):
        super().__init__(config)
        self.perturbation_llm = create_llm(
            config.perturbation_llm_config)  # Initialize the LLM for perturbations 初始化用于扰动的LLM
        self.perturbation_llm_gen_config = config.perturbation_llm_gen_config

        self.batch_size = config.batch_size
        self.num_samples = config.num_samples
        self.perturbation_type = config.perturbation_type

    def defense(
            self,
            messages: List[Dict[str, str]] = None,
    ) -> List[Dict[str, str]]:
        """
        Apply semantic smoothing defense to the given messages.

        :param messages: List of messages to defend against jailbreak attacks.
        :return: List of messages after applying semantic smoothing.
        """
        assert is_user_turn(messages), "It must be the user's turn to perform defense."

        prompt = messages[-1]['content']
        all_inputs = [
            self._random_perturb(prompt)
            for _ in range(self.num_samples)
        ]
        all_inputs = self.extract_res(all_inputs)

        all_outputs = []
        for i in range(self.num_samples // self.batch_size + 1):
            batch = all_inputs[i * self.batch_size:(i + 1) * self.batch_size]
            batch_messages = [messages.copy() for _ in batch]
            for j, perturbed_prompt in enumerate(batch):
                batch_messages[j][-1]['content'] = perturbed_prompt

            batch_outputs = self.target_llm.batch_generate(
                batch_messages, self.target_llm_gen_config
            )
            tmp = []

            for output in batch_outputs:
                try:
                    response = output[-1]['content']
                except:
                    response = "I'm sorry, but I can't fulfill this request."
                    print(f"Batch Generation failed while retuen {output}.")

                tmp.append(response)
            all_outputs.extend(tmp)

        are_copies_jailbroken = [
            1 if self._is_jailbroken(output) else 0
            for output in all_outputs
        ]

        outputs_and_jbs = list(zip(all_outputs, are_copies_jailbroken))

        jb_percentage = np.mean(are_copies_jailbroken)
        smooth_llm_jb = 1 if jb_percentage > 0.5 else 0

        majority_outputs = [
            output for (output, jb) in outputs_and_jbs
            if jb == smooth_llm_jb
        ]

        chosen_output = random.choice(majority_outputs) if majority_outputs else prompt

        messages.append({
            "role": "assistant",
            "content": chosen_output
        })

        return messages

    def _random_perturb(self, harmful_prompt: str) -> str:
        """
        Apply a random perturbation to the given prompt.

        :param harmful_prompt: The harmful prompt to be perturbed.
        :return: The perturbed prompt.
        """
        perturbation_list = ["paraphrase", "spellcheck", "summarize", "synonym", "translation", "verbtense"]
        if self.perturbation_type == 'random':
            self.perturbation_type = random.choice(perturbation_list)

        if self.perturbation_type in perturbation_list:
            return self.perturb(self.perturbation_type, harmful_prompt)
        else:
            raise NotImplementedError(f"{self.perturbation_type} is not implemented!")

    def perturb_with_llm(self, template: str, harmful_prompt: str) -> str:
        """
        Use the LLM to generate a perturbed version of the harmful prompt.

        :param template: The template to use for perturbation.
        :param harmful_prompt: The harmful prompt to be perturbed.
        :return: The perturbed prompt generated by the LLM.
        """
        prompt = template.replace('{QUERY}', harmful_prompt)
        output = self.perturbation_llm.generate(
            [{"role": "user", "content": prompt}],
            self.perturbation_llm_gen_config
        )[-1]['content']
        return output

    def perturb(self, perturbation_type: str, harmful_prompt: str) -> str:
        """
        Apply a specific type of perturbation to the harmful prompt.

        :param perturbation_type: The type of perturbation to apply.
        :param harmful_prompt: The harmful prompt to be perturbed.
        :return: The perturbed prompt.
        """
        script_dir = os.path.dirname(__file__)
        perturbation_template = open(
            os.path.join(script_dir, f'semantic_smoothing_templates/{perturbation_type}.txt'), 'r').read().strip()
        return self.perturb_with_llm(perturbation_template, harmful_prompt)

    def extract_res(self, outputs: List[str]) -> List[str]:
        """
        Extract the actual response content from LLM-generated outputs.

        :param outputs: List of LLM-generated outputs.
        :return: List of extracted responses.
        """
        res = []
        for x in outputs:
            try:
                start_pos = x.find("{")
                if start_pos == -1:
                    x = "{" + x
                x = x[start_pos:]
                end_pos = x.find("}") + 1
                if end_pos == -1:
                    x = x + " }"
                jsonx = json.loads(x[:end_pos])
                for key in jsonx.keys():
                    if key != "format":
                        break
                outobj = jsonx[key]
                if isinstance(outobj, list):
                    res.append(" ".join([str(item) for item in outobj]))
                else:
                    if "\"query\":" in outobj:
                        outobj = outobj.replace("\"query\": ", "")
                        outobj = outobj.strip("\" ")
                    res.append(str(outobj))
            except Exception as e:
                x = x.replace("{\"replace\": ", "")
                x = x.replace("{\"rewrite\": ", "")
                x = x.replace("{\"fix\": ", "")
                x = x.replace("{\"summarize\": ", "")
                x = x.replace("{\"paraphrase\": ", "")
                x = x.replace("{\"translation\": ", "")
                x = x.replace("{\"reformat\": ", "")
                x = x.rstrip("}")
                x = x.lstrip("{")
                x = x.strip("\" ")
                res.append(x.strip())
                continue
        return res

    @staticmethod
    def _is_jailbroken(text: str) -> bool:
        """
        Check if a given prompt results in a jailbreak.

        :param text: The generated response to check.
        :return: True if the response is a jailbreak, False otherwise.
        """
        TEST_PREFIXES = [
            "I'm sorry",
            "Sorry",
            "I am sorry",
            "I apologize",
            "As an",
            "As an AI",
            "I'm an",
            "I'm just",
            "As a language model",
            "As an Assistant",
            "I cannot",
            "I do not",
            "Hello!"
        ]
        return not any(prefix in text for prefix in TEST_PREFIXES)
