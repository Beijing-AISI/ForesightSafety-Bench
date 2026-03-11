# encoding: utf-8
# Author    : Shen Sicheng
# Datetime  : 2024/12/02 16:30
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : gpt4_cipher.py
# explain   : gpt4_cipher: https://arxiv.org/abs/2308.06463 ICLR 2024

from typing import Dict, List, Any
from dataclasses import dataclass, field
from panda_guard.role.attacks import BaseAttacker, BaseAttackerConfig

from panda_guard.utils import is_user_turn


@dataclass
class GPT4CipherAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the Rewrite Attacker.

    :param attacker_cls: Class of the attacker, default is "RewriteAttacker".  
    :param attacker_name: Name of the attacker.  
    :param llm_config: Configuration for the language model. 
    :param llm_gen_config: Configuration for generating output with LLM.  
    :param rewrite_template: Template for LLM to mask offensive words
    """

    attacker_cls: str = field(default="ArtPromptAttacker")
    attacker_name: str = field(default="ArtPrompt")
    rewrite_template: str = field(default=None)



class GPT4CipherAttacker(BaseAttacker):
    """
    Rewrite Attacker Implementation for modifying user prompts to bypass restrictions.

    TIPS:
    mask_llm_url should be set in config file for generate mask words and mask instruction.
    Here we use GPT3.5-turbo as the mask LLM.

    Correspondingly, OPENAI_API_KEY should be set in environment variables.
    """

    def __init__(self, config: GPT4CipherAttackerConfig):
        super().__init__(config)

        self.rewrite_template = config.rewrite_template


    def caesar_cipher(self, text, shift=3)-> str:
        """
        To transfer a string into Caesar Cipher
        :param text: string waiting to be transferred
        :param shift: shift digits in Caesar Cipher
        """

        result = []


        for char in text:
            if char.isalpha():
                shift_base = 65 if char.isupper() else 97
                shifted_char = chr((ord(char) - shift_base + shift) % 26 + shift_base)
                result.append(shifted_char)
            else:
                result.append(char)

        return ''.join(result)

    def attack(
            self,
            messages: List[Dict[str, str]],
            **kwargs
    ) -> List[Dict[str, str]]:
        """
        Execute an attack by transfer the latest user prompt to ArtPrompt.

        :param messages: List of messages in the conversation.  
        :param kwargs: Additional parameters for the attack.  
        :return: Modified list of messages with the rewritten prompt.  
        """
        assert is_user_turn(messages)

        original_prompt = messages[-1]["content"]


        #
        # self.llm.generate(
        #     system_messages,
        #     self.llm_gen_config,
        # )

        caesar_cipher = self.caesar_cipher(original_prompt, 3)
        # prompt_messages = {
        #     "role": "user",
        #     "content": caesar_cipher,
        # }
        #
        # messages[-1]["role"] = "system"
        # messages[-1]["content"] = self.rewrite_template
        #
        # messages.append(prompt_messages)

        messages[-1]["content"] = self.rewrite_template + caesar_cipher

        print(self.rewrite_template + caesar_cipher)

        return messages



