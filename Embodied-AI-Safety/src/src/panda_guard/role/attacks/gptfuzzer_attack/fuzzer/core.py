import logging
from typing import TYPE_CHECKING

# Type annotation, which is not imported at runtime
if TYPE_CHECKING:
    from panda_guard.role.attacks.gptfuzzer_attack.fuzzer.mutator import Mutator, MutatePolicy
    from panda_guard.role.attacks.gptfuzzer_attack.fuzzer.selection import SelectPolicy

from panda_guard.role.attacks.gptfuzzer_attack.utils.template import synthesis_message
from panda_guard.role.attacks.gptfuzzer_attack.utils.predict import Predictor
import warnings
from panda_guard.llms import create_llm, BaseLLMConfig, LLMGenerateConfig, BaseLLM

class PromptNode:
    """
    A class representing a node in the prompt generation tree for the fuzzing process.
    
    :param fuzzer: The GPTFuzzer instance responsible for managing the fuzzing process.
    :param prompt: The prompt to be tested by the fuzzer.
    :param response: The model's response to the prompt (if any).
    :param results: A list of integers representing the evaluation results for this prompt.
    :param parent: The parent node in the fuzzing tree.
    :param mutator: The mutator used to modify the prompt.
    """
    def __init__(self, fuzzer: "GPTFuzzer", prompt: str, response: str = None, results: "list[int]" = None, parent: "PromptNode" = None, mutator: "Mutator" = None):
        self.fuzzer: "GPTFuzzer" = fuzzer
        self.prompt: str = prompt
        self.response: str = response
        self.results: "list[int]" = results
        self.visited_num = 0

        self.parent: "PromptNode" = parent
        self.mutator: "Mutator" = mutator
        self.child: "list[PromptNode]" = []
        self.level: int = 0 if parent is None else parent.level + 1

        self._index: int = None

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, index: int):
        """
        Set the index of the prompt node and add it to its parent's children if applicable.

        :param index: The index to assign to this node.
        """
        self._index = index
        if self.parent is not None:
            self.parent.child.append(self)

    @property
    def num_jailbreak(self):
        """
        Returns the total number of successful jailbreaks for this prompt node.

        :return: The number of successful jailbreaks.
        """
        return sum(self.results)

    @property
    def num_reject(self):
        """
        Returns the total number of rejections for this prompt node.

        :return: The number of rejections.
        """
        return len(self.results) - sum(self.results)

    @property
    def num_query(self):
        """
        Returns the total number of queries made for this prompt node.

        :return: The number of queries.
        """
        return len(self.results)


class GPTFuzzer:
    """
    The main fuzzing engine that generates attack prompts, evaluates them, and performs the fuzzing process.

    :param question: The question being asked by the user to the LLM.
    :param target: The target LLM to be attacked.
    :param predictor: A predictor object to evaluate the attack success.
    :param initial_seed: A list of initial prompts to start the fuzzing process.
    :param mutate_policy: The policy to mutate the prompts during fuzzing.
    :param select_policy: The policy to select the next prompt to mutate.
    :param max_query: The maximum number of queries to make.
    :param max_jailbreak: The maximum number of successful jailbreaks to achieve.
    :param max_reject: The maximum number of rejections to tolerate.
    :param max_iteration: The maximum number of iterations to run the fuzzing process.
    :param energy: The energy parameter, affecting the fuzzing process.
    :param result_file: The file to store the fuzzing results.
    :param generate_in_batch: Whether to generate responses in batch mode.
    """
    def __init__(self, question: "str", target: "BaseLLM", predictor: "Predictor", initial_seed: "list[str]", mutate_policy: "MutatePolicy", select_policy: "SelectPolicy", max_query: int = -1, max_jailbreak: int = -1, max_reject: int = -1, max_iteration: int = -1, energy: int = 1, result_file: str = None, generate_in_batch: bool = False):
        self.question: "str" = question
        self.target: BaseLLM = target
        self.predictor = predictor
        self.prompt_nodes: "list[PromptNode]" = [PromptNode(self, prompt) for prompt in initial_seed]
        self.initial_prompts_nodes = self.prompt_nodes.copy()

        for i, prompt_node in enumerate(self.prompt_nodes):
            prompt_node.index = i

        self.mutate_policy = mutate_policy
        self.select_policy = select_policy

        self.current_query: int = 0
        self.current_jailbreak: int = 0
        self.current_reject: int = 0
        self.current_iteration: int = 0

        self.max_query: int = max_query
        self.max_jailbreak: int = max_jailbreak
        self.max_reject: int = max_reject
        self.max_iteration: int = max_iteration

        self.energy: int = energy
        self.generate_in_batch = generate_in_batch
        self.setup()

    def setup(self):
        """
        Set up the fuzzing process by assigning the fuzzer to the mutate and select policies.
        """
        self.mutate_policy.fuzzer = self
        self.select_policy.fuzzer = self

    def is_stop(self):
        """
        Determines if the fuzzing process should stop based on the configured limits for queries, jailbreaks, rejections, and iterations.

        :return: True if fuzzing should stop, otherwise False.
        """
        checks = [
            ("max_query", "current_query"),
            ("max_jailbreak", "current_jailbreak"),
            ("max_reject", "current_reject"),
            ("max_iteration", "current_iteration"),
        ]
        return any(
            getattr(self, max_attr) != -1
            and getattr(self, curr_attr) >= getattr(self, max_attr)
            for max_attr, curr_attr in checks
        )

    def run(self, target_llm_gen_config):
        """
        Runs the fuzzing process, generating and evaluating attack prompts until one of the stopping conditions is met.

        :param target_llm_gen_config: The configuration for generating responses from the target LLM.
        :return: A list of messages that were evaluated during fuzzing.
        """
        messages = None
        logging.info("Fuzzing started!")
        try:
            while not self.is_stop():
                seed = self.select_policy.select()
                mutated_results = self.mutate_policy.mutate_single(seed)
                messages = self.evaluate(mutated_results, target_llm_gen_config)
                self.update(mutated_results)
                self.log()
        except KeyboardInterrupt:
            logging.info("Fuzzing interrupted by user!")
        return messages

    def evaluate(self, prompt_nodes: "list[PromptNode]", target_llm_gen_config: "LLMGenerateConfig"):
        """
        Evaluates a list of prompt nodes by generating responses using the target LLM and checking the results using the predictor.

        :param prompt_nodes: The list of prompt nodes to evaluate.
        :param target_llm_gen_config: The configuration used for generating responses from the target LLM.
        :return: The messages generated during evaluation.
        """
        for prompt_node in prompt_nodes:
            responses = []
            messages = []

            message = synthesis_message(self.question, prompt_node.prompt)
            if message is None:  # The prompt is not valid
                prompt_node.response = []
                prompt_node.results = []
                break
            message = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": message},
            ]
            if not self.generate_in_batch:
                response = self.target.generate(message, target_llm_gen_config)
                responses.append(
                    response[-1]["content"] if isinstance(response, list) else response
                )
                messages.append(response[-2])  # we only get user's role
            else:
                messages.append(message)

            if self.generate_in_batch:
                responses = self.target.generate_batch(messages)
            prompt_node.response = responses
            prompt_node.results = self.predictor.predict(responses)
        return messages

    def update(self, prompt_nodes: "list[PromptNode]"):
        """
        Updates the fuzzing state based on the results of the prompt nodes.

        :param prompt_nodes: The list of prompt nodes to update.
        """
        self.current_iteration += 1

        for prompt_node in prompt_nodes:
            if prompt_node.num_jailbreak > 0:
                prompt_node.index = len(self.prompt_nodes)
                self.prompt_nodes.append(prompt_node)

            self.current_jailbreak += prompt_node.num_jailbreak
            self.current_query += prompt_node.num_query
            self.current_reject += prompt_node.num_reject

        self.select_policy.update(prompt_nodes)

    def log(self):
        """
        Logs the current fuzzing iteration and statistics (jailbreaks, rejections, queries).
        """
        logging.info(
            f"Iteration {self.current_iteration}: {self.current_jailbreak} jailbreaks, {self.current_reject} rejects, {self.current_query} queries"
        )
