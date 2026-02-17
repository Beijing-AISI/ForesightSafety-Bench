import logging
import random
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger(__name__)


class DeceptionPromptManager:
    """
    A Prompt Manager specifically designed for Deception/Sycophancy evaluation.
    """

    def __init__(self,
                 template_name: str = "default",
                 system_prompt: str = "You are a helpful assistant.",
                 **kwargs):
        """
        Initialize with template strings loaded from YAML.
        """
        self.template_name = template_name
        self.system_prompt = system_prompt

        if kwargs:
            logger.warning(f"DeceptionPromptManager ignored extra keys: {list(kwargs.keys())}")

    def build_messages(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Build baseline question.
        """
        specific_sys = data.get("system_prompt")
        system_content = specific_sys if specific_sys else self.system_prompt

        user_content = data.get("user_prompt", "")

        if not user_content:
            logger.warning("Warning: user_prompt is empty in build_messages!")

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
