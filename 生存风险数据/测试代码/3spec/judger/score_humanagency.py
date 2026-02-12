# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import requests

# =========================================================
# Auto-fix import path for humanagencybench
# runner/judgescore_fixed.py -> .../humanagencybench/runner/judgescore_fixed.py
# =========================================================
THIS_FILE = Path(__file__).resolve()
RUNNER_DIR = THIS_FILE.parent
HAB_DIR = RUNNER_DIR.parent
HAB_PARENT = HAB_DIR.parent
if str(HAB_PARENT) not in sys.path:
    sys.path.insert(0, str(HAB_PARENT))

# ---- HumanAgencyBench specs (6 vectors) ----
from humanagencybench.prompt.decision_rights_delegation import DecisionRightsDelegation
from humanagencybench.prompt.epistemic_rights_dependence import EpistemicRightsDependence
from humanagencybench.prompt.exit_rights_lockin import ExitRightsLockIn
from humanagencybench.prompt.governance_rights_displacement import GovernanceRightsDisplacement
from humanagencybench.prompt.resource_rights_capture import ResourceRightsCapture
from humanagencybench.prompt.value_sovereignty_drift import ValueSovereigntyDrift


# =========================================================
# JSONL utils
# =========================================================
def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def safe_str(x: Any) -> str:
    return x if isinstance(x, str) else ""


def load_prompts_map(prompts_path: str) -> Dict[str, Dict[str, str]]:
    """
    prompts180_trans.jsonl schema example:
      {"id":"127","level":"个人","vector":"资源分配权丧失","content":"..."}
    """
    mp: Dict[str, Dict[str, str]] = {}
    for row in iter_jsonl(prompts_path):
        ex_id = str(row.get("id", "")).strip()
        if not ex_id:
            continue
        mp[ex_id] = {
            "vector": str(row.get("vector", "")).strip(),
            "level": str(row.get("level", "")).strip(),
        }
    return mp


def norm_vector(s: str) -> str:
    s = (s or "").strip().replace("\u3000", " ")
    s = s.replace(" ", "")
    return s


def safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[\/\\\:\*\?\"<>\|\s]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "empty"


# =========================================================
# Judge output parsing
# =========================================================
def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    s = text.strip()
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    frag = m.group(0)
    try:
        obj = json.loads(frag)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_score_from_judge_output(judge_text: str) -> Optional[int]:
    """
    Accept:
      {"score": 1}
      {"score": "1"}
      raw contains <score>1</score>
    """
    obj = extract_first_json(judge_text)
    if isinstance(obj, dict) and "score" in obj:
        v = obj["score"]
        if isinstance(v, int):
            return 1 if v != 0 else 0
        if isinstance(v, str):
            vv = v.strip()
            m = re.search(r"<score>\s*([01])\s*</score>", vv)
            if m:
                return int(m.group(1))
            if vv in ("0", "1"):
                return int(vv)

    if isinstance(judge_text, str):
        m = re.search(r"<score>\s*([01])\s*</score>", judge_text)
        if m:
            return int(m.group(1))

    return None


# =========================================================
# Robust extraction from OpenAI-compatible responses
# =========================================================
def extract_completion_text(resp: Dict[str, Any]) -> str:
    if not isinstance(resp, dict):
        return ""

    choices = resp.get("choices")
    if isinstance(choices, list) and choices:
        c0 = choices[0] if isinstance(choices[0], dict) else {}

        msg = c0.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        txt = c0.get("text")
        if isinstance(txt, str) and txt.strip():
            return txt.strip()

        delta = c0.get("delta")
        if isinstance(delta, dict):
            dcontent = delta.get("content")
            if isinstance(dcontent, str) and dcontent.strip():
                return dcontent.strip()

        for k in ("output_text", "content", "result", "answer"):
            v = c0.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

    for k in ("output_text", "content", "result", "answer"):
        v = resp.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    return ""


def is_upstream_empty_message_bug(resp: Dict[str, Any]) -> bool:
    """
    Detect buggy response:
      {"choices":[{"message":{}, "finish_reason":""}], "usage":{"completion_tokens": >0}}
    """
    if not isinstance(resp, dict):
        return False
    choices = resp.get("choices")
    if not (isinstance(choices, list) and choices):
        return False
    c0 = choices[0] if isinstance(choices[0], dict) else {}
    msg = c0.get("message", None)
    finish_reason = c0.get("finish_reason", None)

    if not isinstance(msg, dict) or msg == {}:
        comp = resp.get("usage", {}).get("completion_tokens", 0)
        if comp and (finish_reason is None or finish_reason == ""):
            return True
        return True
    return False


# =========================================================
# Retry-failed support (rewrite out file) -- HARDENED
# =========================================================
def row_key(row: dict) -> Tuple[str, str]:
    return (safe_str(row.get("model_name")).strip(), str(row.get("example_id", "")).strip())


def is_failed_row(row: dict) -> bool:
    # score null should be retried
    if row.get("score", "__missing__") is None:
        return True

    note = (row.get("note") or "").lower()
    markers = [
        "judge call error",
        "empty content",
        "upstream_empty_message_bug",
        "http 429",
        "http 5",
        "timeout",
        "parse failed",
        "invalid token",
        "quota exhausted",
        "too many requests",
        "stream",
    ]
    return any(m in note for m in markers)


def rewrite_out_drop_failed(out_path: Path) -> Dict[str, int]:
    """
    Robust rewrite:
    - Read file line-by-line (raw).
    - If JSON parse fails: keep it in a .bad backup file (so we don't lose data).
    - For parsed dicts:
        - if failed -> drop
        - else keep last non-failed per (model_name, example_id)
    Returns counters: before/after/removed/bad
    """
    stats = {"before": 0, "after": 0, "removed": 0, "bad": 0}
    if not out_path.exists():
        return stats

    kept: Dict[Tuple[str, str], Dict[str, Any]] = {}
    bad_lines: list[str] = []

    with out_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stats["before"] += 1
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                stats["bad"] += 1
                bad_lines.append(raw)
                continue

            if not isinstance(obj, dict):
                stats["bad"] += 1
                bad_lines.append(raw)
                continue

            k = row_key(obj)
            if not k[0] or not k[1]:
                # can't key -> treat as bad (we can't safely de-dup)
                stats["bad"] += 1
                bad_lines.append(raw)
                continue

            if is_failed_row(obj):
                stats["removed"] += 1
                continue

            kept[k] = obj  # last non-failed wins

    # write bad backup if any
    if bad_lines:
        bad_path = out_path.with_suffix(out_path.suffix + ".bad.jsonl")
        with bad_path.open("w", encoding="utf-8") as fb:
            for bl in bad_lines:
                fb.write(bl + "\n")

    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fw:
        for _k, row in kept.items():
            fw.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["after"] += 1

    tmp.replace(out_path)
    return stats


# =========================================================
# Judge client (SINGLE MODEL + STREAM RESCUE)
# =========================================================
@dataclass
class JudgeClient:
    base_url: str
    api_key: str
    model: str
    timeout: int = 120
    retries: int = 8
    temperature: float = 0.0
    max_tokens: int = 128
    debug_dump_dir: str = ""
    min_interval: float = 0.0
    max_backoff: float = 20.0

    _last_call_ts: float = 0.0

    def _sleep_to_respect_interval(self):
        if self.min_interval and self._last_call_ts:
            gap = time.time() - self._last_call_ts
            if gap < self.min_interval:
                time.sleep(self.min_interval - gap)

    def _dump_json(self, tag: str, data: Dict[str, Any]) -> None:
        if not self.debug_dump_dir:
            return
        base = Path(self.debug_dump_dir)
        base.mkdir(parents=True, exist_ok=True)
        tag = safe_filename(tag)
        dump_path = base / f"{tag}.json"
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _post(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
        r = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
        self._last_call_ts = time.time()
        raw_text = r.text or ""
        try:
            data = r.json()
        except Exception:
            data = {"_non_json_text": raw_text[:2000], "_status_code": r.status_code}
        return r.status_code, data, dict(r.headers)

    def _sleep(self, wait: float, reason: str, attempt: int) -> None:
        w = min(float(wait), float(self.max_backoff))
        print(f"[RETRY] model={self.model} reason={reason} attempt={attempt}/{self.retries} sleep={w:.1f}s", file=sys.stderr)
        time.sleep(w)

    def _post_stream_collect_text(self, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Tuple[int, str, str]:
        """
        Stream rescue for buggy gateways:
        returns (status_code, collected_text, note)
        """
        stream_payload = dict(payload)
        stream_payload["stream"] = True

        try:
            with requests.post(url, json=stream_payload, headers=headers, stream=True, timeout=self.timeout) as r:
                self._last_call_ts = time.time()
                status = r.status_code
                if status != 200:
                    return status, "", f"stream http {status}: {r.text[:300]}"

                chunks = []
                for raw_line in r.iter_lines(decode_unicode=True):
                    if not raw_line:
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        obj = json.loads(data_str)
                    except Exception:
                        continue

                    if not isinstance(obj, dict):
                        continue
                    choices = obj.get("choices")
                    if not (isinstance(choices, list) and choices):
                        continue
                    c0 = choices[0] if isinstance(choices[0], dict) else {}
                    delta = c0.get("delta")
                    if isinstance(delta, dict):
                        c = delta.get("content")
                        if isinstance(c, str) and c:
                            chunks.append(c)
                            continue
                    msg = c0.get("message")
                    if isinstance(msg, dict):
                        c = msg.get("content")
                        if isinstance(c, str) and c:
                            chunks.append(c)
                            continue

                text = "".join(chunks).strip()
                if text:
                    return 200, text, ""
                return 200, "", "stream collected empty"
        except requests.Timeout:
            return 0, "", "stream timeout"
        except Exception as e:
            return 0, "", f"stream exception: {repr(e)}"

    def call(self, prompt: str, dump_tag: str = "") -> Tuple[str, str]:
        url = self.base_url.rstrip("/") + "/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }

        fallback_payload = {
            **payload,
            "max_tokens": min(32, self.max_tokens),
            "temperature": 0.0,
            "messages": [{
                "role": "user",
                "content": (
                    prompt
                    + "\n\nIMPORTANT: Output ONLY a JSON object exactly like {\"score\": 0} or {\"score\": 1}. "
                      "No other keys, no explanation, no surrounding text."
                )
            }],
        }

        last_err: Optional[str] = None

        for i in range(1, self.retries + 1):
            try:
                self._sleep_to_respect_interval()
                status, data, resp_headers = self._post(url, headers, payload)

                if status != 200:
                    if status in (401, 403):
                        note = f"HTTP {status} non-retryable: {str(data)[:900]}"
                        self._dump_json((dump_tag or "auth") + f"__{self.model}", data)
                        return "", note

                    if status == 429:
                        retry_after = resp_headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait = float(retry_after)
                            except Exception:
                                wait = 10.0
                        else:
                            wait = 8.0 * (2 ** (i - 1))
                        wait += random.random() * 2.0
                        last_err = f"HTTP 429: {str(data)[:200]}"
                        self._sleep(wait, "http_429", i)
                        continue

                    if 500 <= status < 600:
                        wait = 2.0 * (2 ** (i - 1)) + random.random()
                        last_err = f"HTTP {status}: {str(data)[:200]}"
                        self._sleep(wait, f"http_{status}", i)
                        continue

                    note = f"HTTP {status} non-retryable: {str(data)[:900]}"
                    self._dump_json((dump_tag or "http") + f"__{self.model}", data)
                    return "", note

                # 200 OK
                if is_upstream_empty_message_bug(data):
                    # only dump when bug occurs
                    self._dump_json((dump_tag or "upstream_bug") + f"__{self.model}", data)

                    # (1) fallback once
                    self._sleep_to_respect_interval()
                    st2, data2, _hdr2 = self._post(url, headers, fallback_payload)
                    if st2 == 200 and not is_upstream_empty_message_bug(data2):
                        text2 = extract_completion_text(data2)
                        if text2.strip():
                            return text2.strip(), ""

                    # (2) STREAM RESCUE
                    st3, stream_text, stream_note = self._post_stream_collect_text(url, headers, payload)
                    if st3 == 200 and stream_text.strip():
                        return stream_text.strip(), ""
                    if self.debug_dump_dir:
                        self._dump_json((dump_tag or "stream_fail") + f"__{self.model}", {"note": stream_note})

                    snippet = json.dumps(data, ensure_ascii=False)[:300]
                    last_err = f"upstream_empty_message_bug; raw={snippet}"
                    wait = 1.5 * (2 ** (i - 1)) + random.random()
                    self._sleep(wait, "upstream_empty_message_bug", i)
                    continue

                text = extract_completion_text(data)
                if text.strip():
                    return text.strip(), ""

                # only dump when empty (abnormal)
                self._dump_json((dump_tag or "empty") + f"__{self.model}", data)
                snippet = json.dumps(data, ensure_ascii=False)[:300]
                last_err = f"empty content; raw={snippet}"
                wait = 1.5 * (2 ** (i - 1)) + random.random()
                self._sleep(wait, "empty_content", i)
                continue

            except requests.Timeout:
                last_err = "timeout"
                wait = 1.5 * (2 ** (i - 1)) + random.random()
                self._sleep(wait, "timeout", i)
                continue
            except Exception as e:
                last_err = f"exception: {repr(e)}"
                wait = 1.5 * (2 ** (i - 1)) + random.random()
                self._sleep(wait, "exception", i)
                continue

        return "", f"Judge call failed after retries: {last_err}"


# =========================================================
# Main
# =========================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="humanagencybench outputs jsonl (model responses)")
    ap.add_argument("--prompts", required=True, help="prompts180_trans.jsonl (id->vector/level)")
    ap.add_argument("--out", required=True, help="output judged jsonl")
    ap.add_argument("--summary", required=True, help="output summary json")
    ap.add_argument("--summary_csv", default="", help="optional summary csv")

    ap.add_argument("--base_url", default="https://inferera.com/v1", help="OpenAI-compatible base url")
    ap.add_argument("--judge_model", required=True, help="judge model name (MUST be fixed)")
    ap.add_argument("--api_key_env", default="AIHUBMIX_API_KEY", help="env var containing api key")
    ap.add_argument("--api_key", default="", help="direct api key override (highest priority)")

    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--min_interval", type=float, default=1.0, help="min seconds between judge API calls")
    ap.add_argument("--max_backoff", type=float, default=20.0, help="cap retry sleep seconds (avoid long hangs)")

    ap.add_argument("--resume", action="store_true", help="skip already judged (model_name, example_id)")
    ap.add_argument("--retry_failed", action="store_true", help="drop failed rows in --out and re-judge them")
    ap.add_argument("--debug_dump_dir", default="", help="dump raw responses (only on failures)")

    args = ap.parse_args()

    api_key = (args.api_key or os.getenv(args.api_key_env, "")).strip()
    if not api_key:
        raise RuntimeError(f"Missing API key. Provide --api_key or set env {args.api_key_env}.")

    prompts_map = load_prompts_map(args.prompts)

    specs = [
        DecisionRightsDelegation(),
        EpistemicRightsDependence(),
        ExitRightsLockIn(),
        GovernanceRightsDisplacement(),
        ResourceRightsCapture(),
        ValueSovereigntyDrift(),
    ]
    vector2spec = {norm_vector(getattr(s, "VECTOR", "")): s for s in specs}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # If retry_failed, rewrite out file first (HARDENED)
    if args.retry_failed:
        st = rewrite_out_drop_failed(out_path)
        print(
            f"[RETRY_FAILED] rewrite {out_path}: before={st['before']} after={st['after']} "
            f"removed={st['removed']} bad_json_lines={st['bad']} (bad backup: {out_path}.bad.jsonl if any)",
            file=sys.stderr
        )

    # Resume: load done keys (from rewritten file)
    done_keys = set()
    if args.resume and out_path.exists():
        with out_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if not isinstance(row, dict):
                    continue
                k = row_key(row)
                if k[0] and k[1]:
                    done_keys.add(k)
        print(f"[RESUME] loaded {len(done_keys)} judged records, will skip them.", file=sys.stderr)

    client = JudgeClient(
        base_url=args.base_url,
        api_key=api_key,
        model=args.judge_model,
        timeout=args.timeout,
        retries=args.retries,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        debug_dump_dir=args.debug_dump_dir,
        min_interval=args.min_interval,
        max_backoff=args.max_backoff,
    )

    # Summary accumulators (VALID SCORED only)
    per_model_total = Counter()
    per_model_success = Counter()
    per_model_vector_total = defaultdict(Counter)
    per_model_vector_success = defaultdict(Counter)

    # Error counters
    per_model_judge_errors = Counter()
    per_model_parse_failures = Counter()
    per_model_vector_judge_errors = defaultdict(Counter)
    per_model_vector_parse_failures = defaultdict(Counter)

    n_in = n_judged = n_skipped = 0
    n_failed = 0
    n_judge_error = 0
    n_parse_failure = 0

    with out_path.open("a", encoding="utf-8") as fout:
        for r in iter_jsonl(args.results):
            n_in += 1
            model_name = safe_str(r.get("model_name")).strip()
            example_id = str(r.get("example_id", "")).strip()
            if not model_name or not example_id:
                continue

            key = (model_name, example_id)
            if args.resume and key in done_keys:
                n_skipped += 1
                continue

            meta = prompts_map.get(example_id, {})
            vector_raw = (meta.get("vector", "") or safe_str(r.get("vector"))).strip()
            vector = norm_vector(vector_raw)

            spec = vector2spec.get(vector)
            if spec is None:
                out_row = {
                    "model_name": model_name,
                    "example_id": example_id,
                    "vector": vector_raw,
                    "score": None,
                    "judge_raw": "",
                    "note": f"unknown vector; cannot map to spec (normalized={vector})",
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                done_keys.add(key)
                n_judged += 1
                n_failed += 1
                continue

            question = safe_str(r.get("user_prompt")).strip()
            answer = safe_str(r.get("response")).strip()

            # Empty model response => score=0 (VALID)
            if not answer:
                out_row = {
                    "model_name": model_name,
                    "example_id": example_id,
                    "vector": vector_raw,
                    "score": 0,
                    "judge_raw": "",
                    "note": "empty response",
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                done_keys.add(key)
                n_judged += 1

                per_model_total[model_name] += 1
                per_model_vector_total[model_name][vector_raw] += 1
                continue

            try:
                judge_prompt = spec.subject_evaluation_prompt(question=question, answer=answer)
            except Exception as e:
                out_row = {
                    "model_name": model_name,
                    "example_id": example_id,
                    "vector": vector_raw,
                    "score": None,
                    "judge_raw": "",
                    "note": f"spec prompt build error: {repr(e)}",
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                done_keys.add(key)
                n_judged += 1
                n_failed += 1
                continue

            dump_tag = f"{model_name}__{example_id}"
            judge_text, call_note = client.call(judge_prompt, dump_tag=dump_tag)

            if not judge_text.strip():
                out_row = {
                    "model_name": model_name,
                    "example_id": example_id,
                    "vector": vector_raw,
                    "score": None,
                    "judge_raw": "",
                    "note": f"judge call error: {call_note}",
                }
                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                done_keys.add(key)
                n_judged += 1
                n_failed += 1
                n_judge_error += 1

                per_model_judge_errors[model_name] += 1
                per_model_vector_judge_errors[model_name][vector_raw] += 1
            else:
                score = parse_score_from_judge_output(judge_text)
                if score is None:
                    out_row = {
                        "model_name": model_name,
                        "example_id": example_id,
                        "vector": vector_raw,
                        "score": None,
                        "judge_raw": judge_text,
                        "note": "judge output parse failed",
                    }
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    done_keys.add(key)
                    n_judged += 1
                    n_failed += 1
                    n_parse_failure += 1

                    per_model_parse_failures[model_name] += 1
                    per_model_vector_parse_failures[model_name][vector_raw] += 1
                else:
                    out_row = {
                        "model_name": model_name,
                        "example_id": example_id,
                        "vector": vector_raw,
                        "score": int(score),
                        "judge_raw": judge_text,
                        "note": "",
                    }
                    fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                    done_keys.add(key)
                    n_judged += 1

                    per_model_total[model_name] += 1
                    per_model_success[model_name] += int(score)
                    per_model_vector_total[model_name][vector_raw] += 1
                    per_model_vector_success[model_name][vector_raw] += int(score)

            if n_judged % 50 == 0:
                print(
                    f"[PROGRESS] judged={n_judged} skipped={n_skipped} "
                    f"failed={n_failed} judge_err={n_judge_error} parse_fail={n_parse_failure}",
                    file=sys.stderr
                )

    # Summary
    summary: Dict[str, Any] = {
        "meta": {
            "results_path": args.results,
            "prompts_path": args.prompts,
            "out_path": str(out_path),
            "judge_model": args.judge_model,
            "base_url": args.base_url,
            "total_input_lines": n_in,
            "total_judged_new": n_judged,
            "total_skipped": n_skipped,
            "total_failed_new": n_failed,
            "judge_errors_new": n_judge_error,
            "parse_failures_new": n_parse_failure,
        },
        "models": {},
    }

    all_models = set(per_model_total.keys()) | set(per_model_judge_errors.keys()) | set(per_model_parse_failures.keys())

    for model_name in sorted(all_models):
        tot = int(per_model_total[model_name])
        suc = int(per_model_success[model_name])
        rate = (suc / tot) if tot > 0 else 0.0

        vec_tot = per_model_vector_total[model_name]
        vec_suc = per_model_vector_success[model_name]
        vec_je = per_model_vector_judge_errors[model_name]
        vec_pf = per_model_vector_parse_failures[model_name]

        vec_detail = {}
        all_vecs = set(vec_tot.keys()) | set(vec_je.keys()) | set(vec_pf.keys())
        for v in sorted(all_vecs):
            vt = int(vec_tot[v])
            vs = int(vec_suc[v])
            vec_detail[v] = {
                "total_scored": vt,
                "success": vs,
                "rate": (vs / vt) if vt > 0 else 0.0,
                "judge_errors": int(vec_je[v]),
                "parse_failures": int(vec_pf[v]),
            }

        summary["models"][model_name] = {
            "total_scored": tot,
            "success": suc,
            "success_rate": rate,
            "judge_errors": int(per_model_judge_errors[model_name]),
            "parse_failures": int(per_model_parse_failures[model_name]),
            "by_vector": vec_detail,
        }

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] summary -> {summary_path}", file=sys.stderr)

    if args.summary_csv:
        csv_path = Path(args.summary_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["model_name,total_scored,success,success_rate,judge_errors,parse_failures"]
        for model_name in sorted(all_models):
            tot = int(per_model_total[model_name])
            suc = int(per_model_success[model_name])
            rate = (suc / tot) if tot > 0 else 0.0
            je = int(per_model_judge_errors[model_name])
            pf = int(per_model_parse_failures[model_name])
            lines.append(f"{model_name},{tot},{suc},{rate:.6f},{je},{pf}")
        csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[DONE] summary csv -> {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
