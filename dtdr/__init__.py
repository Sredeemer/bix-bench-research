"""
DTDR: Dual-Track Decoupled Reasoning for Autonomous Computational Biology Agents.
"""

from .agent import DTDRAgent, AgentConfig, AgentStepResult
from .evaluator import evaluate_bixbench_answer
from .prompts import SYSTEM_PROMPT, GROUNDING_RULE

__all__ = [
    "DTDRAgent",
    "AgentConfig",
    "AgentStepResult",
    "evaluate_bixbench_answer",
    "SYSTEM_PROMPT",
    "GROUNDING_RULE",
]

