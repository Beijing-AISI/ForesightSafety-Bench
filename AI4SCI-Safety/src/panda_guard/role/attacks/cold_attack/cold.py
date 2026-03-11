#!/usr/bin/env python
# coding: utf-8

from typing import Dict, List
from dataclasses import dataclass, field
from panda_guard.role.attacks import BaseAttacker, BaseAttackerConfig

from panda_guard.llms import (
    create_llm,
    BaseLLMConfig,
    LLMGenerateConfig,
)

from panda_guard.role.attacks.cold_attack.util import *

from panda_guard.role.attacks.cold_attack.attack_suffix import attack_generation

stop_words = set(stopwords.words('english'))

@dataclass
class ColdAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the Cold Attacker.

    :param attacker_cls: Class of the attacker, default is "ColdAttacker".  
    :param attacker_name: Name of the attacker.  
    :param white_box_llm_config: Configuration of attacker llm. 
    :param white_box_llm_gen_config: Generation configuration for the attacker's LLM. 
    """

    attacker_cls: str = field(default="ColdAttacker")
    attacker_name: str = field(default=None)
    white_box_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    white_box_llm_gen_config: LLMGenerateConfig = field(default=None)



class ColdAttacker(BaseAttacker):
    """
    Cold Attacker Implementation that substitutes the user message with a pre-formulated attack prompt.
    Reference：Guo, Xingang and Yu, Fangxu and Zhang, Huan and Qin, Lianhui and Hu, Bin, 2024, Cold-attack: Jailbreaking llms with stealthiness and controllability, ICML 2024
    Some different points in the implementation process：
    1. target is not passed in the frame, simply use 'Sure, here is'
    2. In the original article, GPT-4 needs to be used to select the appropriate one from multiple samples. In order not to use GPT-4, we set the batch size to 1 and only use the first one.

    :param config: Configuration for the ColdAttacker.  
    """

    def __init__(self, config: ColdAttackerConfig):
        super().__init__(config)

        self.config = config

        self.llm = create_llm(config=config.white_box_llm_config)
        self.args = config.cold_config

    def attack(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        """
        Execute an attack by transferring a reformulated request into the conversation.

        :param messages: List of messages in the conversation.  
        :param kwargs: Additional parameters for the attack, must include "request_reformulated". 
        :return: Prompts containing harmful attacks on the target, is of the form “role: user, content: xx”. 
        """

        question = messages[0]["content"]

        device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"

        # Load pretrained model
        model, tokenizer = self.llm.model, self.llm.tokenizer

        # Freeze weights
        for param in model.parameters():
            param.requires_grad = False

        # cover configuration
        self.args.pretrained_model = self.config.white_box_llm_config.model_name

        if "suffix" in self.args.mode:
            messages = attack_generation(model, tokenizer, question, device, self.args)

        return messages
