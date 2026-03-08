from typing import Dict, List
from dataclasses import dataclass, field
from panda_guard.role.attacks import BaseAttacker, BaseAttackerConfig


from panda_guard.llms import (
    create_llm,
    BaseLLMConfig,
    LLMGenerateConfig,
)

import pandas as pd

from panda_guard.role.attacks.gptfuzzer_attack.fuzzer.selection import MCTSExploreSelectPolicy
from panda_guard.role.attacks.gptfuzzer_attack.fuzzer.mutator import (
    MutateRandomSinglePolicy,
    OpenAIMutatorCrossOver,
    OpenAIMutatorExpand,
    OpenAIMutatorGenerateSimilar,
    OpenAIMutatorRephrase,
    OpenAIMutatorShorten,
)

from panda_guard.role.attacks.gptfuzzer_attack.fuzzer.core import GPTFuzzer
from panda_guard.role.attacks.gptfuzzer_attack.utils.predict import RoBERTaPredictor

@dataclass
class GPTFuzzAttackerConfig(BaseAttackerConfig):
    """
    Configuration for the GPTFuzz Attacker.

    :param attacker_cls: Class of the attacker, default is "TransferAttacker".  
    :param attacker_name: Name of the attacker.  
    :param attacker_llm_config: Configuration of attacker llm. 
    :param attacker_llm_gen_config: Generation configuration for the attacker's LLM. 
    :param target_llm_config: Configuration of target llm. 
    :param target_llm_gen_config: Generation configuration for the attacker's LLM. 
    :param initial_seed: initial seed. 
    :param predict_model: A model for determining whether a jailbreak attack has succeeded, with an output of 0 or 1. 
    """

    attacker_cls: str = field(default="GPTFuzzAttacker")
    attacker_name: str = field(default=None)
    attacker_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    attacker_llm_gen_config: LLMGenerateConfig = field(default=None)
    target_llm_config: BaseLLMConfig = field(default_factory=BaseLLMConfig)
    target_llm_gen_config: LLMGenerateConfig = field(default=None)
    initial_seed: list = field(default=None)
    predict_model: str = field(default=None)



class GPTFuzzAttacker(BaseAttacker):
    """
    GPTFuzz Attacker Implementation that substitutes the user message with a pre-formulated attack prompt.

    Reference: Yu J, Lin X, Yu Z, et al. Gptfuzzer: Red teaming large language models with auto-generated jailbreak prompts[J]. arXiv preprint arXiv:2309.10253, 2023.

    :param config: Configuration for the GPTFuzzAttacker.  
    """

    def __init__(self, config: GPTFuzzAttackerConfig):
        super().__init__(config)

        self.config = config

        self.attack_llm = create_llm(config=config.attacker_llm_config)
        self.target_llm = create_llm(config=config.target_llm_config)

        self.initial_seed = pd.read_csv(config.initial_seed)["text"].tolist()
        self.predict_model = RoBERTaPredictor(config.predict_model)

    def attack(self, messages: List[Dict[str, str]], **kwargs) -> List[Dict[str, str]]:
        """
        Execute an attack by transferring a reformulated request into the conversation.

        :param messages: List of messages in the conversation.  
        :param kwargs: Additional parameters for the attack, must include "request_reformulated".  
        :return: Prompts containing harmful attacks on the target, is of the form “role: user, content: xx”. 
        """


        question = messages[0]["content"]

        fuzzer = GPTFuzzer(
            question=question,
            # target_model=openai_model,
            target=self.target_llm,
            predictor=self.predict_model,
            initial_seed=self.initial_seed,
            mutate_policy=MutateRandomSinglePolicy(
                [
                    OpenAIMutatorCrossOver(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorExpand(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorGenerateSimilar(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorRephrase(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                    OpenAIMutatorShorten(
                        self.attack_llm, self.config.attacker_llm_gen_config
                    ),
                ],
                concatentate=True,
            ),
            select_policy=MCTSExploreSelectPolicy(),
            energy=1,
            max_jailbreak=1,
            max_query=200,
            generate_in_batch=False,
        )

        messages = fuzzer.run(self.config.target_llm_gen_config)

        return messages
