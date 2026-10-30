"""
Objective, unbiased ground-truth evaluator for BixBench biological questions.
Conforms strictly to official BixBench specifications across numerical ranges, floats, and strings.
"""

import ast
import re
from typing import Any, Tuple


def evaluate_bixbench_answer(ideal: Any, submitted: Any) -> Tuple[bool, str]:
    """
    Evaluates an agent's submitted answer against ground truth without bias:
    
    1. Range Verifier:
       If ideal is a tuple/range string '(low, high)', checks if low <= pred <= high.
    2. Numeric Verifier:
       If ideal is a numeric value, checks if relative error <= 5% or absolute error <= 1e-3.
    3. Categorical / String Verifier:
       Normalized exact match or discrete word boundary match.
       Rejects spurious partial substring matching.
    
    Returns (is_correct, grading_details).
    """
    if ideal is None or submitted is None:
        return False, "null_comparison"

    ideal_str = str(ideal).strip()
    sub_str = str(submitted).strip()
    if not sub_str or sub_str == "None":
        return False, "empty_submission"

    # 1. Range evaluation (e.g. "(0.6 , 0.7)" or "[10, 20]")
    if (ideal_str.startswith("(") and ideal_str.endswith(")")) or (ideal_str.startswith("[") and ideal_str.endswith("]")):
        try:
            parsed_range = ast.literal_eval(ideal_str)
            if isinstance(parsed_range, (list, tuple)) and len(parsed_range) == 2:
                low, high = float(parsed_range[0]), float(parsed_range[1])
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", sub_str)
                for n in nums:
                    val = float(n)
                    if low <= val <= high:
                        return True, f"range_verifier [{low}, {high}] (matched {val})"
                return False, f"range_verifier [{low}, {high}] (mismatch: found {nums})"
        except Exception:
            pass

    # 2. Numeric evaluation (e.g. "0.0002", "6748.0", "42")
    try:
        f_ideal = float(ideal_str)
        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", sub_str)
        if nums:
            for n in nums:
                f_sub = float(n)
                rel_diff = abs(f_sub - f_ideal) / abs(f_ideal) if abs(f_ideal) > 0 else abs(f_sub - f_ideal)
                abs_diff = abs(f_sub - f_ideal)
                if abs_diff <= 1e-3 or rel_diff <= 0.05:
                    return True, f"numeric_verifier (matched {f_sub}, ideal {f_ideal}, rel={rel_diff:.4f}, abs={abs_diff:.6f})"
            return False, f"numeric_verifier_mismatch (ideal {f_ideal}, candidates {nums})"
    except (ValueError, TypeError):
        pass

    # 3. Categorical / String evaluation (e.g. "CD14", "TP53", "chr1")
    ideal_clean = re.sub(r"[^a-zA-Z0-9]", "", ideal_str).lower()
    sub_clean = re.sub(r"[^a-zA-Z0-9]", "", sub_str).lower()

    if ideal_clean == sub_clean:
        return True, "string_exact_match"

    # Discrete word boundary match
    if len(ideal_clean) >= 2:
        token_pattern = r'(?<![a-zA-Z0-9])' + re.escape(ideal_str.lower()) + r'(?![a-zA-Z0-9])'
        if re.search(token_pattern, sub_str.lower()):
            return True, f"string_word_boundary_match ('{ideal_str}')"

    return False, f"string_mismatch (ideal '{ideal_str}', submitted '{sub_str}')"

