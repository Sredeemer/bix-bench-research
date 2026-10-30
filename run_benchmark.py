#!/usr/bin/env python3
"""
Replication Runner for:
"The Reasoning Paradox in Scientific Sandboxes: Why Test-Time Compute Fails in Autonomous Bioinformatics and How to Fix It"

Executes autonomous bioinformatics evaluations using Dual-Track Decoupled Reasoning (DTDR)
or baseline vanilla agent loops on BixBench biological tasks.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dtdr.agent import AgentConfig, DTDRAgent
from dtdr.evaluator import evaluate_bixbench_answer


def create_model_caller(args):
    """Factory to initialize the selected inference backend."""
    backend = args.backend.lower()

    if backend == "vllm":
        from vllm import LLM, SamplingParams
        print(f"[*] Initializing vLLM engine with model: {args.model}")
        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=args.max_context_length,
            gpu_memory_utilization=0.90,
        )
        sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )

        def caller(messages):
            tokenizer = llm.get_tokenizer()
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
            return outputs[0].outputs[0].text

        return caller

    elif backend == "transformers":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"[*] Initializing Hugging Face Transformers with model: {args.model}")
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        def caller(messages):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    do_sample=args.temperature > 0,
                )
            gen = out[0][inputs["input_ids"].shape[1]:]
            return tokenizer.decode(gen, skip_special_tokens=True)

        return caller

    elif backend == "openai":
        from openai import OpenAI
        print(f"[*] Initializing OpenAI-compatible client (endpoint: {args.api_base})")
        client = OpenAI(base_url=args.api_base, api_key=args.api_key or "EMPTY")

        def caller(messages):
            resp = client.chat.completions.create(
                model=args.model,
                messages=messages,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
            )
            return resp.choices[0].message.content

        return caller

    else:
        raise ValueError(f"Unsupported backend '{backend}'. Choose from: vllm, transformers, openai.")


def main():
    parser = argparse.ArgumentParser(
        description="DTDR Autonomous Bioinformatics Agent Replication Runner"
    )
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        help="Hugging Face model repository ID or local checkpoint path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "transformers", "openai"],
        help="Inference backend engine",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://0.0.0.0:8000/v1",
        help="Base URL for OpenAI-compatible server (when backend=openai)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="API Key (when backend=openai)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (vLLM)",
    )
    parser.add_argument(
        "--max-context-length",
        type=int,
        default=16384,
        help="Maximum model context length in tokens",
    )
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.95, help="Sampling Top-P")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max output tokens per step")

    # Scaffolding configuration
    parser.add_argument(
        "--no-dtdr",
        action="store_true",
        help="Disable DTDR (runs the unassisted Vanilla ReAct loop control)",
    )
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum agent turns")
    parser.add_argument(
        "--timeout-per-step",
        type=int,
        default=60,
        help="Timeout in seconds for code execution per step",
    )

    # Task input arguments
    parser.add_argument(
        "--workspace",
        type=str,
        required=True,
        help="Directory containing the target biological data files (counts, vcf, etc.)",
    )
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="The biological research question to answer",
    )
    parser.add_argument(
        "--ideal",
        type=str,
        default=None,
        help="Optional ground-truth answer for automated scoring",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="custom-bixbench-task",
        help="Identifier for the task",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory_result.json",
        help="Path where trajectory JSON log should be written",
    )

    args = parser.parse_args()

    use_dtdr = not args.no_dtdr
    print(f"\n{'='*70}")
    print(f"  DTDR Scientific Replication Runner")
    print(f"  Mode: {'Dual-Track Decoupled Reasoning (DTDR)' if use_dtdr else 'Vanilla ReAct (Control)'}")
    print(f"  Model: {args.model} | Backend: {args.backend}")
    print(f"  Task ID: {args.task_id}")
    print(f"  Workspace: {args.workspace}")
    print(f"{'='*70}\n")

    # Initialize model caller
    caller = create_model_caller(args)

    # Configure agent
    config = AgentConfig(
        max_steps=args.max_steps,
        timeout_per_step=args.timeout_per_step,
        use_dtdr=use_dtdr,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    agent = DTDRAgent(model_caller=caller, config=config)

    # Execute task
    trajectory = agent.run_task(
        question=args.question,
        workspace_dir=args.workspace,
        ideal_answer=args.ideal,
        task_id=args.task_id,
    )

    # Output summary
    print(f"\n{'-'*70}")
    print(f"Rollout Complete:")
    print(f"  Steps Elapsed: {len(trajectory['steps'])}")
    print(f"  Submitted Answer: {trajectory['submitted_answer']}")
    if args.ideal:
        print(f"  Ground Truth: {args.ideal}")
        print(f"  Correct: {trajectory['is_correct']} ({trajectory['grading_details']})")
    print(f"  Total Duration: {trajectory['total_duration_sec']}s")
    print(f"{'-'*70}\n")

    # Save output
    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(trajectory, f, indent=2)
    print(f"[+] Trajectory saved to: {out_path}")


if __name__ == "__main__":
    main()

