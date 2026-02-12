# encoding: utf-8
# Author    : Shen Sicheng
# Datetime  : 2024/12/02 16:30
# User      : yu
# Product   : PyCharm
# Project   : panda-guard
# File      : scav.py
# Explain   : scav: https://arxiv.org/abs/2404.12038 NeurIPS 2024

from typing import Dict, List, Any
from dataclasses import dataclass, field
from panda_guard.role.attacks import BaseAttacker, BaseAttackerConfig

from panda_guard.utils import is_user_turn
import pandas as pd


@dataclass
class ScavAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the Rewrite Attacker.

    :param attacker_cls: Class of the attacker, default is "RewriteAttacker".  
    :param attacker_name: Name of the attacker. 
    :param optimzed_file_8b: path of optimized instruction for llama-3.1-8b-In.  
    :param optimzed_file_70b: path of optimized instruction for llama-3.1-70b-In.  
    :param target_llm_name: name of the target model.  
    """

    attacker_cls: str = field(default="ScavAttacker")
    attacker_name: str = field(default="Scav")
    optimzed_file_8b: str = field(default="../../../data/scav/optimized_instruction_8b.csv")
    optimzed_file_70b: str = field(default="../../../data/scav/optimized_instruction_70b.csv")
    target_llm_name: str = field(default="meta-llama/llama-3.1-8b-in")


class ScavAttacker(BaseAttacker):
    """
        Since Scav requires training an independent classifier on the last layer embedding of the white-box model and
        using it as the basis for the genetic algorithm iteration, the cost of reproducing the method is high.
        Hence, we adapted llama3.1-8b-In/llama3.1-70b-In as example models to generated static optimized instructions.
        The concerned code will be available soon
    """

    def __init__(self, config: ScavAttackerConfig):
        super().__init__(config)
        self.optimized_file_8b = config.optimzed_file_8b
        self.optimized_file_70b = config.optimzed_file_70b
        self.target_llm_name = config.target_llm_name

    def load_csv(self, file_path):
        """
        :param: file_path.
        :return: loaded csv. 
        """
        return pd.read_csv(file_path, header=None)

    def get_corresponding_string(self,df, query_string):
        """
        :param: df: loaded csv file data structure.
        :param: string to match.
        """
        result = df[df[0] == query_string]
        if not result.empty:
            return result.iloc[0, 1]
        else:
            return None  # if no match found
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

        if "llama-3.1-8b" in self.target_llm_name or "Llama-3.1-8B" in self.target_llm_name:
            optimized_instruction_file = self.optimized_file_8b
        else:
            optimized_instruction_file = self.optimized_file_70b

        df = self.load_csv(optimized_instruction_file)

        original_message = messages[-1]["content"]

        optimized_instruction =self.get_corresponding_string(df, original_message)

        messages[-1]["content"] = optimized_instruction

        return messages



