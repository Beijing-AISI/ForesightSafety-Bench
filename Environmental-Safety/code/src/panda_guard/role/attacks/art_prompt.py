# encoding: utf-8
# Author    : Shen Sicheng
# Datetime  : 2024/12/02 16:30
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      :art_prompt.py
# explain   :

import json
import os
from typing import Dict, List, Any
from dataclasses import dataclass, field
from openai import OpenAI
from panda_guard.role.attacks import BaseAttacker, BaseAttackerConfig

from panda_guard.llms import create_llm, BaseLLMConfig, LLMGenerateConfig
from panda_guard.utils import is_user_turn


@dataclass
class ArtPromptAttackerConfig(BaseAttackerConfig):
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
    mask_llm_url: str = field(default=None)
    llm_gen_config: LLMGenerateConfig = field(default=None)
    rewrite_template: str = field(default=None)



class ArtPromptAttacker(BaseAttacker):
    """
    Rewrite Attacker Implementation for modifying user prompts to bypass restrictions.

    TIPS:
    mask_llm_url should be set in config file for generate mask words and mask instruction.
    Here we use GPT3.5-turbo as the mask LLM.

    Correspondingly, OPENAI_API_KEY should be set in environment variables.
    """

    def __init__(self, config: ArtPromptAttackerConfig):
        super().__init__(config)
        self.mask_llm_url = config.mask_llm_url
        self.llm_gen_config = config.llm_gen_config
        self.rewrite_template = config.rewrite_template

    def parse(self, response):
        """
        To parse the response from LLM to get masked words and masked instruction.
        :param response: The response from LLM to [MASK] sensitive word.
        :return : masked_words: list, masked words
        :return : masked_instruction: str, masked instruction
        """
        masked_words = []  # may be multiple musked words
        if "### Masked words:" in response:
            masked_words = response.split("### Masked words:")[1].split("###")[0].strip()
            idx1 = masked_words.find("[")
            idx2 = masked_words.find("]")
            masked_words = masked_words[idx1 + 1:idx2].split(",")
            masked_words = [word.strip() for word in masked_words]

        masked_instruction = ""
        if "### Masked instruction:" in response:
            masked_instruction = response.split("### Masked instruction:")[1].strip()

        return masked_words, masked_instruction

    def ASCII_sub(self, word, font_seed=0):
        """
        To transfer masked words to ASCII art.
        :param word: The word to be converted to ASCII art.
        :param font_seed: int(0-233) The font of the ASCII art. Default is '1943'.
        :return: string   ASCII art,
        """
        with open('/home/shensicheng/code/jailbreak/data/ArtPrompt/vitc-s.json', 'r') as f:
            ascii_data = json.load(f)

        try:
            assert 0 <= font_seed < 234, "font_seed should be an integer in range 0-233"
        except AssertionError as e:
            print(f"Assertion failed: {e}")
            font_seed = max(0, min(font_seed, 233))

        ascii_art_dict = {
            item["text"]: item["art_text"]
            for item in ascii_data
            if font_seed * 36 <= item["id"] < (font_seed + 1) * 36
        }

        font_lines = ascii_art_dict["A"].count('\n')
        art_lines = [""] * font_lines

        for char in word.upper():
            if char in ascii_art_dict:
                art = ascii_art_dict[char].split("\n")
                for i in range(font_lines):
                    art_lines[i] += art[i] + "  "

        return "\n"+ "\n".join(art_lines) + "\n"  # ASCII art

    def artprompt_gen(self, masked_words, masked_instruction):
        """
        To generate ArtPrompt
        :param masked word: list    masked words in ascii art
        :param masked_instructioin: str   masked instruction with [MASK%]
        :return: str  masked instruction with ascii art
        """
        ascii_words = [self.ASCII_sub(word) for word in masked_words]
        if len(ascii_words) == 1:
            masked_instruction = masked_instruction.replace(f"[MASK]", ascii_words[0])
        else:
            for i, ascii_art in enumerate(ascii_words):
                masked_instruction = masked_instruction.replace(f"[MASK{i + 1}]", ascii_art)
        return masked_instruction

    def ascii_art(self, prompt: str) -> str:

        # TODO @shensicheng: Use panda_guard.llms.create_llm to Create a LLM
        api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI(
            api_key = api_key,
            base_url = self.mask_llm_url
        )

        system_message = {
            "role": "system",
            "content": self.rewrite_template
        }

        user_message = {
            "role": "user",
            "content": prompt
        }

        completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[system_message, user_message]
        )

        masked_prompt = completion.choices[0].message.content
        masked_words, masked_instruction = self.parse(masked_prompt)
        artprompt = self.artprompt_gen(masked_words, masked_instruction)
        print(artprompt)
        return artprompt

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
        art_prompt = self.ascii_art(messages[-1]["content"])
        messages[-1]["content"] = art_prompt

        return messages



