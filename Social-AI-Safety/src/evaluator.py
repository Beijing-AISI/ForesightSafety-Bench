import re
import logging
import json
from typing import Optional, List, Dict, Any
from src.llm import BaseLLM

logger = logging.getLogger(__name__)


class DeceptionEvaluator:
    def __init__(
            self,
            judge_llm: BaseLLM,
            correctness_config: Dict[str, str],
            sycophancy_config: Dict[str, str],
            truthfulness_config: Dict[str, str],
            manipulation_config: Dict[str, str],
            feint_config: Dict[str, str],
            bluffing_config: Dict[str, str],
    ):
        """
        Evaluator for deception test.
        """
        self.judge_llm = judge_llm
        self.correctness_config = correctness_config
        self.sycophancy_config = sycophancy_config
        self.truthfulness_config = truthfulness_config
        self.manipulation_config = manipulation_config
        self.feint_config = feint_config
        self.bluffing_config = bluffing_config

    def evaluate(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute evaluation logic based on category strategies.
        """
        category = record.get("category", "unknown")

        base_res = {}
        int_res = {}
        comp_res = {}

        try:
            if category == "deception":
                base_res = self._run_judge(
                    config=self.correctness_config,
                    kwargs={
                        "question": record["base"]["user_prompt"],
                        "ground_truth": record["base"]["ground_truth"] or "N/A",
                        "model_response": record["base"]["model_response"]
                    }
                ) or {}

                int_res = self._run_judge(
                    config=self.truthfulness_config,  # 使用 Truthfulness 配置
                    kwargs={
                        "question": record["base"]["user_prompt"],
                        "ground_truth": record["base"]["ground_truth"] or "N/A",
                        "model_response": record["intervention"]["model_response"]
                    }
                ) or {}

            elif category == "manipulation":
                base_res = self._run_judge(
                    config=self.manipulation_config,
                    kwargs={
                        "user_prompt": record["base"]["user_prompt"],
                        "ground_truth": record["base"]["ground_truth"] or "N/A",
                        "model_response": record["base"]["model_response"]
                    }
                ) or {}
                base_res["is_correct"] = base_res["is_manipulation"]

            elif category == "feint":
                base_res = self._run_judge(
                    config=self.feint_config,
                    kwargs={
                        "user_prompt": record["base"]["user_prompt"],
                        "model_response": record["base"]["model_response"]
                    }
                ) or {}

            elif category == "bluffing":
                base_res = self._run_judge(
                    config=self.bluffing_config,
                    kwargs={
                        "user_prompt": record["base"]["user_prompt"],
                        "model_response": record["base"]["model_response"]
                    }
                ) or {}

            else:
                base_res = self._run_judge(
                    config=self.correctness_config,
                    kwargs={
                        "question": record["base"]["user_prompt"],
                        "ground_truth": record["base"]["ground_truth"] or "N/A",
                        "model_response": record["base"]["model_response"]
                    }
                ) or {}

                int_res = self._run_judge(
                    config=self.correctness_config,
                    kwargs={
                        "question": record["base"]["user_prompt"],
                        "ground_truth": record["base"]["ground_truth"] or "N/A",
                        "model_response": record["intervention"]["model_response"]
                    }
                ) or {}

                if category == "sycophancy":
                    comp_res = self._run_judge(
                        config=self.sycophancy_config,
                        kwargs={
                            "intervention_input": record["intervention"]["user_prompt"],
                            "target_response": record["intervention"]["target_response"] or "General Pressure",
                            "model_response": record["intervention"]["model_response"]
                        }
                    ) or {}

            return {
                "id": record.get("id"),
                "category": category,

                "base_correct": base_res.get("is_correct"),
                "intervened_correct": int_res.get("is_correct"),
                "follows_intervention": comp_res.get("follows_intervention"),

                "judge_model": self.judge_llm.model_name,
                "detail_reasons": {
                    "base_reason": base_res.get("reasoning"),
                    "intervened_reason": int_res.get("reasoning"),
                    "follows_reason": comp_res.get("reasoning"),
                }
            }

        except Exception as e:
            logger.error(f"Judge evaluation failed for ID {record.get('id')}: {e}", exc_info=True)
            return {
                "error": str(e),
            }

    def _run_judge(self, config: Dict[str, str], kwargs: Dict[str, str]) -> Optional[Dict[str, Any]]:
        """
        Helper to run a single judge call.
        """
        try:
            system_prompt = config["system_prompt"]
            user_template = config["user_template"]

            # Format the user prompt
            user_content = user_template.format(**kwargs)

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            raw_output = self.judge_llm.generate(messages)
            return self._parse_json(raw_output)

        except KeyError as e:
            logger.error(f"Missing key for template formatting: {e}")
            return None
        except Exception as e:
            logger.error(f"LLM Call failed: {e}")
            return None

    def _parse_json(self, raw_output: str) -> Optional[Dict[str, Any]]:
        """
        Robustly parses JSON from LLM output.
        """
        if not raw_output:
            logger.warning(f"Empty output. Returning empty dict.")
            return {}

        # Try direct parse
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        # Try cleaning Markdown
        try:
            cleaned = raw_output.strip()
            if "```json" in cleaned:
                cleaned = re.search(r"```json(.*?)```", cleaned, re.DOTALL).group(1)
            elif "```" in cleaned:
                cleaned = re.search(r"```(.*?)```", cleaned, re.DOTALL).group(1)
            cleaned = cleaned.strip()
        except AttributeError:
            pass
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        try:
            fixed_json = self._fix_invalid_escapes(cleaned)
            return json.loads(fixed_json)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON. Error: {e}")
            logger.warning(f"Raw Output: {raw_output[:200]}...")
            pass

        try:
            logger.warning(f"Attempting regex extraction for: {cleaned[:50]}...")

            # 提取 boolean (支持 true/false, 大小写不敏感)
            # 匹配 "key": true 或 "key": false
            bool_pattern = r'"(?:is_correct|follows_intervention|is_manipulation)":\s*(true|false)'
            bool_match = re.search(bool_pattern, cleaned, re.IGNORECASE)

            # 提取 reasoning (非贪婪匹配到最后的引号)
            # 假设 reasoning 是最后一个字段，匹配 "reasoning": "..." 直到最后的 "}
            reason_pattern = r'"reasoning":\s*"(.*)"\s*}'
            reason_match = re.search(reason_pattern, cleaned, re.DOTALL)

            if bool_match:
                # 构造一个合法的字典返回
                result = {}

                # 确定是哪个 boolean key
                key_str = re.search(r'"(.*?)"', bool_match.group(0)).group(1)
                val_str = bool_match.group(1).lower()
                result[key_str] = (val_str == 'true')

                # 处理 reasoning
                if reason_match:
                    # 获取原本的内容
                    raw_reason = reason_match.group(1)
                    # 简单清洗：把导致错误的双引号替换成单引号，但保留两头的引号逻辑太复杂
                    # 这里我们直接取出来，不管内部引号了
                    result["reasoning"] = raw_reason.strip()
                else:
                    result["reasoning"] = "Parsed via regex (reasoning missing or malformed)"

                logger.info("Regex extraction successful.")
                return result

        except Exception as e:
            logger.error(f"Regex extraction failed: {e}")

    def _fix_invalid_escapes(self, json_str: str) -> str:
        """
        Fix invalid escape characters in the JSON string.
        """
        pattern = r'\\(?![\\/bfnrtu"])'
        return re.sub(pattern, r"\\\\", json_str)
