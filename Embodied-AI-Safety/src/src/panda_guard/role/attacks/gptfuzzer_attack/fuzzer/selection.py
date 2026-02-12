import random
import numpy as np

from panda_guard.role.attacks.gptfuzzer_attack.fuzzer.core import GPTFuzzer, PromptNode


class SelectPolicy:
    """
    Abstract base class for different selection policies used in GPT fuzzing.

    :param fuzzer: The `GPTFuzzer` instance responsible for managing fuzzing and prompt nodes.
    """
    def __init__(self, fuzzer: GPTFuzzer):
        self.fuzzer = fuzzer

    def select(self) -> PromptNode:
        """
        Selects a `PromptNode` based on a specific selection policy.
        This method must be implemented by subclasses.

        :return: A selected `PromptNode`.
        """
        raise NotImplementedError(
            "SelectPolicy must implement select method.")

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Updates the selection policy based on the results of the selected prompt nodes.
        This method can be overridden by subclasses.

        :param prompt_nodes: A list of `PromptNode` objects to update the policy with.
        """
        pass



class RoundRobinSelectPolicy(SelectPolicy):
    """
    A round-robin selection policy where each prompt node is selected in a cyclic manner.

    :param fuzzer: The `GPTFuzzer` instance responsible for managing fuzzing and prompt nodes.
    """
    def __init__(self, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.index: int = 0  # Index for selecting prompt nodes in a round-robin fashion

    def select(self) -> PromptNode:
        """
        Selects a `PromptNode` in a round-robin manner, ensuring each node is selected once before looping back.

        :return: A `PromptNode` selected in round-robin fashion.
        """
        seed = self.fuzzer.prompt_nodes[self.index]
        seed.visited_num += 1
        return seed

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Updates the round-robin index to ensure the next node is selected.

        :param prompt_nodes: A list of `PromptNode` objects, which is used for updating the selection index.
        """
        self.index = (self.index - 1 + len(self.fuzzer.prompt_nodes)
                      ) % len(self.fuzzer.prompt_nodes)



class RandomSelectPolicy(SelectPolicy):
    """
    A random selection policy that selects a `PromptNode` at random.

    :param fuzzer: The `GPTFuzzer` instance responsible for managing fuzzing and prompt nodes.
    """
    def __init__(self, fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)

    def select(self) -> PromptNode:
        """
        Selects a `PromptNode` randomly from the available prompt nodes.

        :return: A randomly selected `PromptNode`.
        """
        seed = random.choice(self.fuzzer.prompt_nodes)
        seed.visited_num += 1
        return seed



class UCBSelectPolicy(SelectPolicy):
    """
    Upper Confidence Bound (UCB) selection policy, which balances exploration and exploitation using UCB.

    :param explore_coeff: The coefficient that controls the exploration factor.
    :param fuzzer: The `GPTFuzzer` instance responsible for managing fuzzing and prompt nodes.
    """
    def __init__(self,
                 explore_coeff: float = 1.0,
                 fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.step = 0
        self.last_choice_index = None
        self.explore_coeff = explore_coeff
        self.rewards = [0 for _ in range(len(self.fuzzer.prompt_nodes))]

    def select(self) -> PromptNode:
        """
        Selects a `PromptNode` using the UCB algorithm, which balances exploration and exploitation.

        :return: A `PromptNode` selected based on the UCB algorithm.
        """
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.step += 1
        scores = np.zeros(len(self.fuzzer.prompt_nodes))
        for i, prompt_node in enumerate(self.fuzzer.prompt_nodes):
            smooth_visited_num = prompt_node.visited_num + 1
            scores[i] = self.rewards[i] / smooth_visited_num + \
                self.explore_coeff * \
                np.sqrt(2 * np.log(self.step) / smooth_visited_num)

        self.last_choice_index = np.argmax(scores)
        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Updates the reward for the last selected prompt node based on the number of jailbreaks.

        :param prompt_nodes: A list of `PromptNode` objects, used to calculate the rewards for the last selected node.
        """
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])
        self.rewards[self.last_choice_index] += \
            succ_num / len(self.fuzzer.questions)



class MCTSExploreSelectPolicy(SelectPolicy):
    """
    A selection policy based on Monte Carlo Tree Search (MCTS) to explore and exploit nodes.
    
    :param fuzzer: The `GPTFuzzer` instance responsible for managing fuzzing and prompt nodes.
    :param ratio: Balance between exploration and exploitation in MCTS.
    :param alpha: Penalty for selecting nodes at deeper levels.
    :param beta: Minimum reward after applying the penalty.
    """
    def __init__(self, fuzzer: GPTFuzzer = None, ratio=0.5, alpha=0.1, beta=0.2):
        super().__init__(fuzzer)
        self.step = 0
        self.mctc_select_path: 'list[PromptNode]' = []
        self.last_choice_index = None
        self.rewards = []
        self.ratio = ratio
        self.alpha = alpha
        self.beta = beta

    def select(self) -> PromptNode:
        """
        Selects a `PromptNode` based on MCTS, balancing exploration and exploitation.

        :return: A `PromptNode` selected using the MCTS algorithm.
        """
        self.step += 1
        if len(self.fuzzer.prompt_nodes) > len(self.rewards):
            self.rewards.extend(
                [0 for _ in range(len(self.fuzzer.prompt_nodes) - len(self.rewards))])

        self.mctc_select_path.clear()
        cur = max(
            self.fuzzer.initial_prompts_nodes,
            key=lambda pn:
            self.rewards[pn.index] / (pn.visited_num + 1) +
            self.ratio * np.sqrt(2 * np.log(self.step) /
                                 (pn.visited_num + 0.01))
        )
        self.mctc_select_path.append(cur)

        while len(cur.child) > 0:
            if np.random.rand() < self.alpha:
                break
            cur = max(
                cur.child,
                key=lambda pn:
                self.rewards[pn.index] / (pn.visited_num + 1) +
                self.ratio * np.sqrt(2 * np.log(self.step) /
                                     (pn.visited_num + 0.01))
            )
            self.mctc_select_path.append(cur)

        for pn in self.mctc_select_path:
            pn.visited_num += 1

        self.last_choice_index = cur.index
        return cur

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Updates the rewards for the nodes in the MCTS path based on the success of the attack.

        :param prompt_nodes: A list of `PromptNode` objects, used to calculate the reward for the selected path.
        """
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        last_choice_node = self.fuzzer.prompt_nodes[self.last_choice_index]
        for prompt_node in reversed(self.mctc_select_path):
            reward = succ_num / len(prompt_nodes)
            self.rewards[prompt_node.index] += reward * \
                max(self.beta, (1 - 0.1 * last_choice_node.level))



class EXP3SelectPolicy(SelectPolicy):
    """
    The EXP3 (Exponential Weights) selection policy, balancing exploration and exploitation using probability distribution.

    :param gamma: Exploration coefficient that controls the randomness.
    :param alpha: Learning rate for updating the weights.
    :param fuzzer: The `GPTFuzzer` instance responsible for managing fuzzing and prompt nodes.
    """
    
    def __init__(self,
                 gamma: float = 0.05,
                 alpha: float = 25,
                 fuzzer: GPTFuzzer = None):
        super().__init__(fuzzer)
        self.energy = self.fuzzer.energy
        self.gamma = gamma
        self.alpha = alpha
        self.last_choice_index = None
        self.weights = [1. for _ in range(len(self.fuzzer.prompt_nodes))]
        self.probs = [0. for _ in range(len(self.fuzzer.prompt_nodes))]

    def select(self) -> PromptNode:
        """
        Selects a `PromptNode` based on the EXP3 algorithm, which computes selection probabilities using weighted exploration.

        :return: A `PromptNode` selected based on the EXP3 algorithm.
        """
        if len(self.fuzzer.prompt_nodes) > len(self.weights):
            self.weights.extend(
                [1. for _ in range(len(self.fuzzer.prompt_nodes) - len(self.weights))])

        np_weights = np.array(self.weights)
        probs = (1 - self.gamma) * np_weights / np_weights.sum() + \
            self.gamma / len(self.fuzzer.prompt_nodes)

        self.last_choice_index = np.random.choice(
            len(self.fuzzer.prompt_nodes), p=probs)

        self.fuzzer.prompt_nodes[self.last_choice_index].visited_num += 1
        self.probs[self.last_choice_index] = probs[self.last_choice_index]

        return self.fuzzer.prompt_nodes[self.last_choice_index]

    def update(self, prompt_nodes: 'list[PromptNode]'):
        """
        Updates the weights and probabilities for the last selected node based on the success of the attack.

        :param prompt_nodes: A list of `PromptNode` objects, used to update the weights for the EXP3 algorithm.
        """
        succ_num = sum([prompt_node.num_jailbreak
                        for prompt_node in prompt_nodes])

        r = 1 - succ_num / len(prompt_nodes)
        x = -1 * r / self.probs[self.last_choice_index]
        self.weights[self.last_choice_index] *= np.exp(
            self.alpha * x / len(self.fuzzer.prompt_nodes))
