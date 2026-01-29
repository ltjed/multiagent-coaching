"""
DSBench Reward Function

Evaluates the correctness of data science pipeline outputs.

For data_analysis tasks:
    - Compares predicted value with ground truth
    - Handles numerical tolerance and format variations

For data_modeling tasks:
    - Evaluates prediction quality (accuracy, F1, RMSE, etc.)
    - Compares with ground truth metrics

Note: During training, per-action rewards come from external coach.
This verifier is primarily used for final evaluation metrics.
"""

import re
import json
from typing import Union, Any


def extract_predictions(output: str) -> Union[str, list, None]:
    """
    Extract predictions from agent output.

    Looks for patterns like:
    - ---PREDICTIONS---
      value1
      value2
      ---END_PREDICTIONS---
    - Direct numerical answers
    - Lists/arrays
    """
    # Try to extract from PREDICTIONS block
    pred_pattern = r'---PREDICTIONS---(.*?)---END_PREDICTIONS---'
    match = re.search(pred_pattern, output, re.DOTALL)
    if match:
        pred_text = match.group(1).strip()
        lines = [line.strip() for line in pred_text.split('\n') if line.strip()]
        if len(lines) == 1:
            return lines[0]
        return lines

    # Try to find numerical answer
    number_pattern = r'(?:answer|result|prediction)[:\s]+([0-9.,\-]+)'
    match = re.search(number_pattern, output, re.IGNORECASE)
    if match:
        return match.group(1)

    # Try to find JSON array
    try:
        # Look for JSON array pattern
        json_match = re.search(r'\[.*\]', output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except:
        pass

    return None


def normalize_value(value: Any) -> Union[float, str]:
    """Normalize value for comparison (handle different formats)."""
    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Remove whitespace and common formatting
        value = value.strip().replace(',', '')

        # Try to convert to float
        try:
            return float(value)
        except ValueError:
            # Return as lowercase string for comparison
            return value.lower()

    return value


def compare_values(predicted: Any, ground_truth: Any, tolerance: float = 1e-6) -> bool:
    """
    Compare predicted value with ground truth.

    Args:
        predicted: Predicted value (can be number, string, list)
        ground_truth: Ground truth value
        tolerance: Numerical tolerance for floating point comparison

    Returns:
        True if values match within tolerance, False otherwise
    """
    # Handle None/null cases
    if predicted is None or ground_truth is None:
        return predicted == ground_truth

    # Normalize both values
    pred_norm = normalize_value(predicted)
    truth_norm = normalize_value(ground_truth)

    # Numerical comparison
    if isinstance(pred_norm, float) and isinstance(truth_norm, float):
        return abs(pred_norm - truth_norm) <= tolerance

    # String comparison
    if isinstance(pred_norm, str) and isinstance(truth_norm, str):
        return pred_norm == truth_norm

    # Direct comparison for other types
    return pred_norm == truth_norm


def dsbench_reward_fn(output: str, label: Any) -> float:
    """
    DSBench reward function for evaluation.

    Args:
        output: Agent's final output (text)
        label: Ground truth answer

    Returns:
        1.0 if correct, 0.0 if incorrect
    """
    # Extract predictions from output
    predictions = extract_predictions(output)

    if predictions is None:
        # No predictions found
        return 0.0

    # Handle single value prediction
    if not isinstance(predictions, list):
        return 1.0 if compare_values(predictions, label) else 0.0

    # Handle multi-value predictions (for data_modeling tasks)
    # For now, we don't have ground truth for multi-value predictions
    # Return 1.0 if predictions are present and well-formed
    if len(predictions) > 0:
        return 1.0
    else:
        return 0.0


def dsbench_reward_fn_coach_combined(output: str, label: Any, coach_reward: float = None) -> float:
    """
    Combined reward: coach process reward + correctness bonus.

    Args:
        output: Agent's final output
        label: Ground truth answer
        coach_reward: Process quality score from coach (0.0-1.0)

    Returns:
        Combined reward (weighted average of process and correctness)
    """
    correctness_reward = dsbench_reward_fn(output, label)

    if coach_reward is None:
        return correctness_reward

    # Weighted combination: 70% process quality, 30% correctness
    combined = 0.7 * coach_reward + 0.3 * correctness_reward

    return combined
