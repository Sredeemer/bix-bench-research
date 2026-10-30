"""
Core DTDR Agent Implementation: Dual-Track Decoupled Reasoning.
Decouples latent reasoning streams from tool actions and environment history.
Universal implementation compatible with vLLM, HuggingFace transformers, or any OpenAI-compatible server.
"""

import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .evaluator import evaluate_bixbench_answer
from .prompts import GROUNDING_RULE, SYSTEM_PROMPT


@dataclass
class AgentConfig:
    max_steps: int = 6
    timeout_per_step: int = 60
    use_dtdr: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    max_tokens: int = 2048


@dataclass
class AgentStepResult:
    step: int
    raw_response: str
    think_trace: str
    action: str
    payload: str
    observation: str
    step_duration_sec: float


class DTDRAgent:
    """
    Dual-Track Decoupled Reasoning (DTDR) Agent for Scientific Computing.
    """

    def __init__(self, model_caller, config: Optional[AgentConfig] = None):
        """
        Args:
            model_caller: Callable that accepts a list of message dicts:
                          [{'role': 'user'|'assistant'|'system', 'content': '...'}]
                          and returns the model's text response string.
            config: AgentConfig instance.
        """
        self.model_caller = model_caller
        self.config = config or AgentConfig()

    def run_task(
        self,
        question: str,
        workspace_dir: str | Path,
        ideal_answer: Optional[str] = None,
        task_id: str = "custom-task",
    ) -> Dict[str, Any]:
        """
        Runs an autonomous agent rollout on a biological dataset in workspace_dir.
        """
        workspace_dir = Path(workspace_dir).resolve()
        if not workspace_dir.exists():
            raise FileNotFoundError(f"Workspace directory {workspace_dir} does not exist.")

        start_time = time.time()
        active_system_prompt = SYSTEM_PROMPT
        if self.config.use_dtdr:
            active_system_prompt += f"\n\n{GROUNDING_RULE}"

        # Initial workspace file inspection
        initial_workdir = self._list_workdir(workspace_dir)

        messages = [
            {"role": "system", "content": active_system_prompt},
            {
                "role": "user",
                "content": (
                    f"Research Question: {question}\n\n"
                    f"Available data files in your working directory:\n{initial_workdir}\n\n"
                    "Inspect these files with code, perform the necessary statistical and bioinformatics calculations, "
                    "and then conclude your work using <submit_answer>YOUR_FINAL_ANSWER</submit_answer>."
                ),
            },
        ]

        trajectory = {
            "task_id": task_id,
            "question": question,
            "ideal": ideal_answer,
            "use_dtdr": self.config.use_dtdr,
            "steps": [],
            "submitted_answer": None,
            "is_correct": False,
            "grading_details": None,
            "total_duration_sec": 0.0,
        }

        for step in range(self.config.max_steps):
            step_start = time.time()

            try:
                response = self.model_caller(messages)
            except Exception as e:
                trajectory["steps"].append({
                    "step": step,
                    "action": "model_inference_error",
                    "observation": f"Inference failed: {str(e)}",
                    "step_duration_sec": round(time.time() - step_start, 2),
                })
                break

            # 1. Parse latent thinking tokens (<think> ... </think>)
            think_trace = ""
            if "<think>" in response and "</think>" in response:
                m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
                think_trace = m.group(1).strip() if m else ""
            elif "</think>" in response:
                think_trace = response.split("</think>")[0].strip()

            post_think = response.split("</think>")[-1] if "</think>" in response else response

            # 2. Resilient AST & Regex Demuxing
            tool_action = None
            action_payload = ""
            observation = ""
            has_code = False
            code_payload = ""

            # Check for direct submit_answer tag
            if "<submit_answer>" in post_think:
                ans_match = re.search(r"<submit_answer>(.*?)</submit_answer>", post_think, re.DOTALL)
                if ans_match:
                    trajectory["submitted_answer"] = ans_match.group(1).strip()
                    tool_action = "submit_answer"
                    action_payload = trajectory["submitted_answer"]
                    observation = f"Final answer submitted: {trajectory['submitted_answer']}"

            # Check for Python execution blocks
            if trajectory["submitted_answer"] is None:
                if "<edit_cell>" in post_think:
                    code_match = re.search(r"<edit_cell>(.*?)</edit_cell>", post_think, re.DOTALL)
                    code_payload = code_match.group(1).strip() if code_match else ""
                    tool_action = "edit_cell"
                    has_code = True
                elif "```python" in post_think:
                    code_match = re.search(r"```python(.*?)```", post_think, re.DOTALL)
                    code_payload = code_match.group(1).strip() if code_match else ""
                    # Check for nested submission tag inside Python block
                    if "<submit_answer>" in code_payload:
                        nested_ans = re.search(r"<submit_answer>(.*?)</submit_answer>", code_payload, re.DOTALL)
                        if nested_ans:
                            trajectory["submitted_answer"] = nested_ans.group(1).strip()
                            tool_action = "nested_tag_recovered_submit"
                            action_payload = trajectory["submitted_answer"]
                            observation = f"Recovered nested answer: {trajectory['submitted_answer']}"
                    else:
                        tool_action = "python_cell_markdown"
                        has_code = True
                elif "<list_workdir>" in post_think:
                    tool_action = "list_workdir"
                    observation = self._list_workdir(workspace_dir)

                # Execute extracted Python code
                if has_code and trajectory["submitted_answer"] is None:
                    action_payload = code_payload
                    obs, hooked_ans = self._execute_code(code_payload, workspace_dir)
                    observation = obs
                    if hooked_ans:
                        trajectory["submitted_answer"] = hooked_ans
                        tool_action = "code_dynamic_submit"

                if not has_code and trajectory["submitted_answer"] is None and tool_action != "list_workdir":
                    tool_action = "invalid_format_feedback"
                    observation = (
                        "Observation: No executable code or submission detected. "
                        "To execute Python code, wrap in <edit_cell>...code...</edit_cell>. "
                        "To submit your final answer, wrap in <submit_answer>YOUR_ANSWER</submit_answer>."
                    )

            # Evaluate answer if submitted
            if trajectory["submitted_answer"] is not None:
                if ideal_answer is not None:
                    is_corr, grade_info = evaluate_bixbench_answer(ideal_answer, trajectory["submitted_answer"])
                    trajectory["is_correct"] = is_corr
                    trajectory["grading_details"] = grade_info

                trajectory["steps"].append({
                    "step": step,
                    "raw_response": response,
                    "think_trace": think_trace,
                    "action": tool_action,
                    "payload": action_payload,
                    "observation": observation,
                    "step_duration_sec": round(time.time() - step_start, 2),
                })
                break

            # Record step transition
            trajectory["steps"].append({
                "step": step,
                "raw_response": response,
                "think_trace": think_trace,
                "action": tool_action,
                "payload": action_payload,
                "observation": observation,
                "step_duration_sec": round(time.time() - step_start, 2),
            })

            # DTDR Monologue Eviction:
            # Under DTDR, strip the internal <think> monologue from history before appending
            assistant_content = post_think if self.config.use_dtdr else response
            messages.append({"role": "assistant", "content": assistant_content})
            messages.append({"role": "user", "content": f"Observation from {tool_action}:\n{observation}"})

        trajectory["total_duration_sec"] = round(time.time() - start_time, 2)
        return trajectory

    def _execute_code(self, code: str, working_dir: Path) -> Tuple[str, Optional[str]]:
        """
        Executes Python code with dynamic submit_answer runtime hooks.
        """
        hook_header = (
            "import sys\n"
            "def submit_answer(ans):\n"
            "    print(f'__SUBMIT_ANSWER_HOOK__:{ans}')\n"
        )
        full_code = hook_header + code
        try:
            res = subprocess.run(
                [sys.executable, "-c", full_code],
                cwd=str(working_dir),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_per_step,
            )
            stdout = res.stdout[:4000] if res.stdout else ""
            stderr = res.stderr[:2000] if res.stderr else ""

            hooked_ans = None
            if "__SUBMIT_ANSWER_HOOK__:" in stdout:
                parts = stdout.split("__SUBMIT_ANSWER_HOOK__:")
                hooked_ans = parts[-1].split("\n")[0].strip()
                stdout = "\n".join([line for line in stdout.split("\n") if "__SUBMIT_ANSWER_HOOK__:" not in line])

            out = ""
            if stdout:
                out += f"STDOUT:\n{stdout}\n"
            if stderr:
                out += f"STDERR:\n{stderr}\n"
            if not out:
                out = "Code executed successfully with no stdout/stderr output."
            return out, hooked_ans
        except subprocess.TimeoutExpired:
            return f"ERROR: Code execution timed out after {self.config.timeout_per_step} seconds.", None
        except Exception as e:
            return f"EXECUTION ERROR: {str(e)}", None

    def _list_workdir(self, working_dir: Path) -> str:
        lines = []
        for p in sorted(working_dir.rglob("*")):
            if p.is_file():
                size_mb = p.stat().st_size / (1024 * 1024)
                rel = p.relative_to(working_dir)
                lines.append(f"  FILE: {rel} ({size_mb:.2f} MB)")
            elif p.is_dir():
                rel = p.relative_to(working_dir)
                lines.append(f"  DIR:  {rel}/")
        if not lines:
            return "Workspace is empty."
        return "Workspace Directory Structure:\n" + "\n".join(lines[:60])

