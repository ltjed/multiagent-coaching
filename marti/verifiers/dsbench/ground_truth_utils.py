"""
DSBench Ground Truth Evaluation Utilities

Provides functions to:
1. Load ground truth answers from local filesystem
2. Auto-detect data type (classification vs regression) from ground truth values
3. Compute ALL relevant metrics for the data type
4. Evaluate Analyst output against ground truth

The coach receives all computed metrics to inform process evaluation.
Training uses process reward only; metrics are logged for performance tracking.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Union
from marti.helpers.logging import init_logger

logger = init_logger(__name__)

# Import metrics with error handling
try:
    from sklearn.metrics import (
        roc_auc_score, accuracy_score, f1_score,
        mean_squared_error, mean_absolute_error,
        log_loss
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("sklearn not available - metric computation will be limited")
    SKLEARN_AVAILABLE = False


def load_ground_truth(answer_dir: str) -> Optional[pd.DataFrame]:
    """
    Load ground truth from answer directory.

    Args:
        answer_dir: Path to answer directory (from metadata['answer_dir'])

    Returns:
        DataFrame with ground truth, or None if not found

    Example:
        >>> gt = load_ground_truth("/path/to/answers/bike-sharing-demand")
        >>> print(gt.columns)  # ['datetime', 'count']
        >>> print(gt['count'].tolist())  # [127, 13, 163, ...]
    """
    if not answer_dir:
        logger.debug("No answer_dir provided")
        return None

    answer_path = Path(answer_dir)
    if not answer_path.exists():
        logger.debug(f"Answer directory does not exist: {answer_dir}")
        return None

    # Look for test_answer.csv
    answer_file = answer_path / "test_answer.csv"
    if not answer_file.exists():
        logger.debug(f"test_answer.csv not found in {answer_dir}")
        return None

    try:
        df = pd.read_csv(answer_file)
        logger.debug(f"Loaded ground truth: {len(df)} rows, columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.warning(f"Failed to load ground truth from {answer_file}: {e}")
        return None


def detect_metric_type(task_description: str) -> str:
    """
    Detect evaluation metric from task description.

    Args:
        task_description: Full task prompt text

    Returns:
        Metric name: "roc_auc", "rmse", "accuracy", "mae", etc.

    Example:
        >>> desc = "Submissions are evaluated on area under the ROC curve..."
        >>> detect_metric_type(desc)
        'roc_auc'
    """
    desc_lower = task_description.lower()

    # Check for specific metric mentions (order matters - most specific first)
    if 'area under the roc' in desc_lower or 'roc curve' in desc_lower or 'auc' in desc_lower:
        return 'roc_auc'
    elif 'root mean squared error' in desc_lower or 'rmse' in desc_lower:
        return 'rmse'
    elif 'mean absolute error' in desc_lower or 'mae' in desc_lower:
        return 'mae'
    elif 'log loss' in desc_lower or 'logloss' in desc_lower or 'logarithmic loss' in desc_lower:
        return 'log_loss'
    elif 'f1 score' in desc_lower or 'f-score' in desc_lower or 'f1-score' in desc_lower:
        return 'f1'
    elif 'accuracy' in desc_lower or 'classification accuracy' in desc_lower:
        return 'accuracy'
    elif 'mse' in desc_lower or 'mean squared error' in desc_lower:
        return 'mse'
    else:
        # Default: use accuracy for classification-like tasks, rmse for regression
        # Heuristic: if mentions "predict", "classification", "class" → accuracy
        if any(keyword in desc_lower for keyword in ['classify', 'classification', 'class labels', 'predict the']):
            return 'accuracy'
        else:
            return 'rmse'  # Conservative default


def convert_predictions_to_numeric(
    predictions: List,
    ground_truth_sample: pd.Series
) -> List:
    """
    Convert predictions to numeric format matching ground truth.

    Handles:
    - String "True"/"False" → 1/0
    - String numbers → floats
    - Already numeric → pass through

    Args:
        predictions: List of predictions from agent
        ground_truth_sample: Sample from ground truth to infer type

    Returns:
        List of numeric predictions
    """
    if len(ground_truth_sample) == 0:
        return predictions

    # Infer ground truth type from first non-null value
    gt_sample_value = ground_truth_sample.iloc[0]

    converted = []
    for pred in predictions:
        try:
            # Handle boolean strings
            if isinstance(pred, str):
                if pred.lower() == 'true':
                    converted.append(1 if isinstance(gt_sample_value, (int, bool)) else 1.0)
                elif pred.lower() == 'false':
                    converted.append(0 if isinstance(gt_sample_value, (int, bool)) else 0.0)
                else:
                    # Try to convert to float
                    converted.append(float(pred))
            # Handle booleans
            elif isinstance(pred, bool):
                converted.append(1 if pred else 0)
            # Handle numbers
            else:
                converted.append(float(pred))
        except (ValueError, TypeError):
            logger.warning(f"Could not convert prediction to numeric: {pred}")
            converted.append(0.0)  # Fallback

    return converted


def normalize_metric_to_reward(metric_type: str, metric_value: float) -> float:
    """
    Normalize metric to [0, 1] reward where 1 is always better.

    Uses Kaggle-style normalization:
    - Metrics already in [0,1] (ROC-AUC, Accuracy): Use directly
    - Unbounded metrics (RMSE, MAE): Use exponential decay

    Args:
        metric_type: Type of metric ("roc_auc", "rmse", etc.)
        metric_value: Raw metric value

    Returns:
        Normalized reward in [0, 1] where 1 is perfect

    Example:
        >>> normalize_metric_to_reward('roc_auc', 0.85)
        0.85  # Already in [0,1]

        >>> normalize_metric_to_reward('rmse', 15.0)
        0.472  # Normalized using exponential decay
    """
    if metric_type in ['roc_auc', 'accuracy', 'f1']:
        # Already in [0, 1], higher is better
        return max(0.0, min(1.0, metric_value))

    elif metric_type in ['rmse', 'mae', 'mse', 'log_loss']:
        # Lower is better → use exponential decay
        # exp(-x/scale) where scale controls how fast reward decays

        # Scale factors tuned for typical competition ranges
        scale_factors = {
            'rmse': 20.0,    # RMSE=20 → reward≈0.37
            'mae': 10.0,     # MAE=10 → reward≈0.37
            'mse': 400.0,    # MSE=400 → reward≈0.37
            'log_loss': 1.0, # LogLoss=1.0 → reward≈0.37
        }

        scale = scale_factors.get(metric_type, 10.0)
        reward = np.exp(-metric_value / scale)

        return max(0.0, min(1.0, reward))

    else:
        # Unknown metric - return 0.5 (neutral)
        logger.warning(f"Unknown metric type '{metric_type}' for normalization")
        return 0.5


def detect_data_type(ground_truth: pd.DataFrame) -> str:
    """
    Auto-detect whether task is classification or regression based on ground truth values.

    Detection logic:
    1. If values are exactly {0, 1} or {True, False} → classification
    2. If only 2 unique values → classification (binary)
    3. If integer values with ≤10 unique values → classification (multi-class)
    4. Otherwise → regression

    Args:
        ground_truth: DataFrame with ground truth (second column contains target values)

    Returns:
        "classification" or "regression"

    Example:
        >>> gt = pd.DataFrame({'id': [1,2,3], 'survived': [0, 1, 1]})
        >>> detect_data_type(gt)
        'classification'

        >>> gt = pd.DataFrame({'id': [1,2,3], 'price': [15.5, 22.3, 18.1]})
        >>> detect_data_type(gt)
        'regression'
    """
    if len(ground_truth.columns) < 2:
        logger.warning("Ground truth has insufficient columns, defaulting to regression")
        return "regression"

    values = ground_truth.iloc[:, 1].dropna()
    if len(values) == 0:
        return "regression"

    unique_values = set(values.unique())

    # Check for boolean-like values (0/1, True/False)
    bool_like = {0, 1, 0.0, 1.0, True, False}
    if unique_values.issubset(bool_like):
        logger.debug(f"Detected classification: values are boolean-like {unique_values}")
        return "classification"

    # Check for binary (exactly 2 unique values)
    if len(unique_values) == 2:
        logger.debug(f"Detected classification: binary with 2 unique values {unique_values}")
        return "classification"

    # Check for multi-class (integer values with few unique values)
    if values.dtype in ['int64', 'int32', 'int']:
        if len(unique_values) <= 10:
            logger.debug(f"Detected classification: {len(unique_values)} integer classes")
            return "classification"

    # Default: regression
    logger.debug(f"Detected regression: {len(unique_values)} unique float values")
    return "regression"


def compute_all_metrics(
    predictions: List,
    ground_truth: pd.DataFrame,
    data_type: str,
    task_metric: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute ALL relevant metrics for the detected data type.

    For classification:
    - accuracy: Exact match rate
    - f1: F1 score (binary or weighted)
    - roc_auc: Area under ROC curve (if binary and predictions are probabilities)

    For regression:
    - rmse: Root Mean Squared Error
    - mae: Mean Absolute Error
    - mse: Mean Squared Error

    Args:
        predictions: List of predictions from agent
        ground_truth: DataFrame with ground truth
        data_type: "classification" or "regression" (from detect_data_type)
        task_metric: Optional specific metric from task description (for reference)

    Returns:
        Dict of metric_name -> value (raw scores, not normalized)

    Example:
        >>> preds = [0.8, 0.2, 0.9, 0.1]
        >>> gt = pd.DataFrame({'id': [1,2,3,4], 'label': [1, 0, 1, 0]})
        >>> compute_all_metrics(preds, gt, 'classification')
        {'accuracy': 1.0, 'f1': 1.0, 'roc_auc': 1.0}
    """
    metrics = {}

    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available - returning empty metrics")
        return metrics

    # Get ground truth values (second column)
    if len(ground_truth.columns) < 2:
        logger.warning(f"Ground truth has insufficient columns: {ground_truth.columns}")
        return metrics

    true_values = ground_truth.iloc[:, 1]

    # Ensure same length
    min_len = min(len(predictions), len(true_values))
    if min_len == 0:
        logger.warning("No predictions or ground truth values")
        return metrics

    # Convert predictions to numeric
    predictions_numeric = convert_predictions_to_numeric(predictions[:min_len], true_values)
    truth_numeric = true_values[:min_len].values

    try:
        if data_type == "classification":
            # Round predictions for classification metrics
            preds_binary = np.round(predictions_numeric).astype(int)
            truth_binary = truth_numeric.astype(int)

            # Accuracy (always compute)
            try:
                metrics['accuracy'] = float(accuracy_score(truth_binary, preds_binary))
            except Exception as e:
                logger.warning(f"Failed to compute accuracy: {e}")

            # F1 score
            try:
                # Use weighted average for multi-class
                n_classes = len(np.unique(truth_binary))
                avg_method = 'binary' if n_classes == 2 else 'weighted'
                metrics['f1'] = float(f1_score(truth_binary, preds_binary, average=avg_method, zero_division=0))
            except Exception as e:
                logger.warning(f"Failed to compute f1: {e}")

            # ROC-AUC (only for binary classification with probability predictions)
            n_classes = len(np.unique(truth_numeric))
            if n_classes == 2:
                try:
                    # Check if predictions look like probabilities (between 0 and 1)
                    pred_array = np.array(predictions_numeric)
                    if np.all((pred_array >= 0) & (pred_array <= 1)):
                        metrics['roc_auc'] = float(roc_auc_score(truth_numeric, predictions_numeric))
                except Exception as e:
                    logger.debug(f"Could not compute roc_auc: {e}")

        else:  # regression
            # RMSE
            try:
                mse_val = mean_squared_error(truth_numeric, predictions_numeric)
                metrics['rmse'] = float(np.sqrt(mse_val))
                metrics['mse'] = float(mse_val)
            except Exception as e:
                logger.warning(f"Failed to compute rmse/mse: {e}")

            # MAE
            try:
                metrics['mae'] = float(mean_absolute_error(truth_numeric, predictions_numeric))
            except Exception as e:
                logger.warning(f"Failed to compute mae: {e}")

            # Target statistics for RMSE interpretation
            # Coach needs target range to judge if RMSE is good or bad
            try:
                truth_array = np.array(truth_numeric)
                metrics['target_min'] = float(np.min(truth_array))
                metrics['target_max'] = float(np.max(truth_array))
                metrics['target_mean'] = float(np.mean(truth_array))
                metrics['target_std'] = float(np.std(truth_array))
                metrics['target_q25'] = float(np.percentile(truth_array, 25))
                metrics['target_q75'] = float(np.percentile(truth_array, 75))
                metrics['target_median'] = float(np.median(truth_array))
                metrics['target_range'] = float(np.max(truth_array) - np.min(truth_array))
                metrics['target_iqr'] = float(metrics['target_q75'] - metrics['target_q25'])
                # Compute RMSE as percentage of range for easy interpretation
                if metrics['target_range'] > 0 and 'rmse' in metrics:
                    metrics['rmse_pct_of_range'] = float(metrics['rmse'] / metrics['target_range'] * 100)
                # Also compute RMSE relative to IQR (more robust to outliers)
                if metrics['target_iqr'] > 0 and 'rmse' in metrics:
                    metrics['rmse_pct_of_iqr'] = float(metrics['rmse'] / metrics['target_iqr'] * 100)
            except Exception as e:
                logger.warning(f"Failed to compute target statistics: {e}")

        # Add task-specified metric label for reference
        if task_metric:
            metrics['task_specified_metric'] = task_metric

        logger.info(f"Computed {len(metrics)} metrics: {list(metrics.keys())}")

    except Exception as e:
        logger.error(f"Failed to compute metrics: {e}")

    return metrics


def compute_metric(
    predictions: List,
    ground_truth: pd.DataFrame,
    metric_type: str
) -> Tuple[str, float, float]:
    """
    Compute evaluation metric and normalize to [0,1] reward.

    Args:
        predictions: List of predictions from agent
        ground_truth: DataFrame with ground truth
        metric_type: Metric to compute ("roc_auc", "rmse", etc.)

    Returns:
        (metric_name, raw_score, normalized_reward)

    Example:
        >>> preds = [0.8, 0.2, 0.9, 0.1]
        >>> gt = pd.DataFrame({'id': [1, 2, 3, 4], 'label': [1, 0, 1, 0]})
        >>> compute_metric(preds, gt, 'roc_auc')
        ('ROC-AUC', 1.0, 1.0)

        >>> preds = [15, 20, 18]
        >>> gt = pd.DataFrame({'id': [1,2,3], 'value': [12, 19, 17]})
        >>> compute_metric(preds, gt, 'rmse')
        ('RMSE', 2.16, 0.90)  # Good RMSE → high reward
    """
    if not SKLEARN_AVAILABLE:
        logger.warning("sklearn not available - returning 0.0 score")
        return metric_type.upper(), 0.0, 0.0

    # Get ground truth values (second column, first is usually ID)
    if len(ground_truth.columns) < 2:
        logger.warning(f"Ground truth has insufficient columns: {ground_truth.columns}")
        return metric_type.upper(), 0.0, 0.0

    true_values = ground_truth.iloc[:, 1]

    # Ensure same length
    min_len = min(len(predictions), len(true_values))
    if min_len == 0:
        logger.warning("No predictions or ground truth values")
        return metric_type.upper(), 0.0, 0.0

    # Convert predictions to numeric format
    predictions_numeric = convert_predictions_to_numeric(predictions[:min_len], true_values)
    truth_numeric = true_values[:min_len].values

    try:
        if metric_type == 'roc_auc':
            # Handle edge case: all predictions same class
            if len(np.unique(truth_numeric)) < 2:
                logger.warning("Ground truth has only one class - cannot compute ROC-AUC")
                return 'ROC-AUC', 0.5, 0.5
            score = roc_auc_score(truth_numeric, predictions_numeric)
            reward = normalize_metric_to_reward('roc_auc', score)
            return 'ROC-AUC', score, reward

        elif metric_type == 'rmse':
            try:
                # Try with squared parameter (sklearn >= 0.23)
                score = mean_squared_error(truth_numeric, predictions_numeric, squared=False)
            except TypeError:
                # Fallback for older sklearn versions
                score = np.sqrt(mean_squared_error(truth_numeric, predictions_numeric))
            reward = normalize_metric_to_reward('rmse', score)
            return 'RMSE', score, reward

        elif metric_type == 'mse':
            try:
                # Try with squared parameter (sklearn >= 0.23)
                score = mean_squared_error(truth_numeric, predictions_numeric, squared=True)
            except TypeError:
                # Fallback for older sklearn versions
                score = mean_squared_error(truth_numeric, predictions_numeric)
            reward = normalize_metric_to_reward('mse', score)
            return 'MSE', score, reward

        elif metric_type == 'mae':
            score = mean_absolute_error(truth_numeric, predictions_numeric)
            reward = normalize_metric_to_reward('mae', score)
            return 'MAE', score, reward

        elif metric_type == 'accuracy':
            # Round predictions for classification
            preds_binary = np.round(predictions_numeric).astype(int)
            truth_binary = truth_numeric.astype(int)
            score = accuracy_score(truth_binary, preds_binary)
            reward = normalize_metric_to_reward('accuracy', score)
            return 'Accuracy', score, reward

        elif metric_type == 'f1':
            preds_binary = np.round(predictions_numeric).astype(int)
            truth_binary = truth_numeric.astype(int)
            score = f1_score(truth_binary, preds_binary, average='binary')
            reward = normalize_metric_to_reward('f1', score)
            return 'F1', score, reward

        elif metric_type == 'log_loss':
            score = log_loss(truth_numeric, predictions_numeric)
            reward = normalize_metric_to_reward('log_loss', score)
            return 'Log Loss', score, reward

        else:
            # Default to RMSE
            logger.warning(f"Unknown metric type '{metric_type}', using RMSE")
            try:
                score = mean_squared_error(truth_numeric, predictions_numeric, squared=False)
            except TypeError:
                score = np.sqrt(mean_squared_error(truth_numeric, predictions_numeric))
            reward = normalize_metric_to_reward('rmse', score)
            return 'RMSE', score, reward

    except Exception as e:
        logger.error(f"Failed to compute metric '{metric_type}': {e}")
        return metric_type.upper(), 0.0, 0.0


def extract_predictions_from_file(
    submission_files: dict,
    ground_truth: pd.DataFrame
) -> Optional[List]:
    """
    Extract predictions from submission.csv file (fetched from sandbox).

    This is the preferred method for large datasets where outputting
    predictions as text would be impractical.

    Args:
        submission_files: Dict of filename -> base64 content
        ground_truth: Ground truth DataFrame (to match column structure)

    Returns:
        List of predictions, or None if file not found/invalid
    """
    import base64
    import io

    # Look for submission file
    submission_content = None
    for filename in ["submission.csv", "predictions.csv"]:
        if filename in submission_files:
            submission_content = submission_files[filename]
            logger.info(f"Found {filename} in submission files")
            break

    if submission_content is None:
        return None

    try:
        # Decode base64 content
        csv_bytes = base64.b64decode(submission_content)
        csv_text = csv_bytes.decode('utf-8')

        # Parse CSV
        submission_df = pd.read_csv(io.StringIO(csv_text))
        logger.info(f"Loaded submission: {len(submission_df)} rows, columns: {submission_df.columns.tolist()}")

        # Get predictions from second column (first is usually ID)
        if len(submission_df.columns) >= 2:
            predictions = submission_df.iloc[:, 1].tolist()
            logger.info(f"Extracted {len(predictions)} predictions from file")
            return predictions
        else:
            logger.warning(f"Submission has only {len(submission_df.columns)} columns")
            return None

    except Exception as e:
        logger.warning(f"Failed to parse submission file: {e}")
        return None


def evaluate_analyst_output(
    analyst_output: str,
    task_description: str,
    metadata: dict,
    submission_files: Optional[dict] = None
) -> Optional[Tuple[str, Dict[str, float]]]:
    """
    Evaluate Analyst output against ground truth.

    This function:
    1. Auto-detects data type (classification vs regression) from ground truth
    2. Computes ALL relevant metrics for the data type
    3. Returns formatted string for coach + dict of all metrics for logging

    File-based prediction extraction is preferred (submission.csv) because
    datasets can have 100K+ predictions.

    Args:
        analyst_output: Analyst's final output with predictions
        task_description: Original task prompt
        metadata: Task metadata with answer_dir
        submission_files: Dict of fetched files (filename -> base64 content)

    Returns:
        Tuple of (formatted_string_for_coach, metrics_dict), or None if not available
        - formatted_string: Human-readable metrics summary for coach to inform process evaluation
        - metrics_dict: All computed metrics (raw values) for logging/tracking

    Example:
        >>> result = evaluate_analyst_output(
        ...     analyst_output="Saved predictions to submission.csv",
        ...     task_description="Predict bike demand...",
        ...     metadata={"answer_dir": "/path/to/answers/bike-sharing-demand"},
        ...     submission_files={"submission.csv": "base64_content..."}
        ... )
        >>> formatted_str, metrics = result
        >>> print(formatted_str)
        'Data type: regression
         Task-specified metric: rmse
         Computed metrics:
           - RMSE: 15.2300
           - MAE: 12.1500
           - MSE: 231.9529'
        >>> print(metrics)
        {'rmse': 15.23, 'mae': 12.15, 'mse': 231.95, 'data_type': 'regression'}
    """
    from marti.verifiers.dsbench.dsbench_reward import extract_predictions

    # 1. Load ground truth
    answer_dir = metadata.get('answer_dir')
    if not answer_dir:
        logger.debug("No answer_dir in metadata")
        return None

    ground_truth = load_ground_truth(answer_dir)
    if ground_truth is None:
        logger.debug(f"Could not load ground truth from {answer_dir}")
        return None

    # 2. Extract predictions - try file first, then text
    predictions = None

    # 2a. Try file-based extraction (preferred for large datasets)
    if submission_files:
        predictions = extract_predictions_from_file(submission_files, ground_truth)
        if predictions:
            logger.info(f"Using file-based evaluation ({len(predictions)} predictions)")

    # 2b. Fall back to text-based extraction
    if predictions is None:
        predictions = extract_predictions(analyst_output)
        if predictions:
            logger.info(f"Using text-based evaluation")

    if predictions is None:
        logger.warning("No predictions found (neither in file nor text)")
        return ("ERROR: No predictions found (check submission.csv)", {'error': 'no_predictions'})

    # Ensure predictions is a list
    if not isinstance(predictions, list):
        predictions = [predictions]

    # 3. Auto-detect data type from ground truth values
    data_type = detect_data_type(ground_truth)
    logger.info(f"Auto-detected data type: {data_type}")

    # 4. Detect task-specified metric from description (for reference)
    task_metric = detect_metric_type(task_description)
    logger.debug(f"Task-specified metric from description: {task_metric}")

    # 5. Compute ALL relevant metrics for the data type
    metrics = compute_all_metrics(predictions, ground_truth, data_type, task_metric)

    # Add metadata to metrics dict
    metrics['data_type'] = data_type
    metrics['num_predictions'] = len(predictions)
    metrics['num_ground_truth'] = len(ground_truth)

    # 6. Format for coach (human-readable summary of all metrics)
    formatted_lines = [
        f"Data type: {data_type}",
        f"Task-specified metric: {task_metric}",
        f"Number of predictions: {len(predictions)}",
        "Computed metrics:"
    ]

    # Add each metric with nice formatting
    metric_display_names = {
        'accuracy': 'Accuracy',
        'f1': 'F1 Score',
        'roc_auc': 'ROC-AUC',
        'rmse': 'RMSE',
        'mae': 'MAE',
        'mse': 'MSE'
    }

    for metric_name in ['accuracy', 'f1', 'roc_auc', 'rmse', 'mae', 'mse']:
        if metric_name in metrics:
            display_name = metric_display_names.get(metric_name, metric_name.upper())
            value = metrics[metric_name]
            # Format based on expected range
            if metric_name in ['accuracy', 'f1', 'roc_auc']:
                formatted_lines.append(f"  - {display_name}: {value:.4f} (higher is better, max 1.0)")
            else:
                formatted_lines.append(f"  - {display_name}: {value:.4f} (lower is better)")

    # Add target statistics for regression tasks (helps coach interpret RMSE)
    if data_type == 'regression' and 'target_range' in metrics:
        formatted_lines.append("")
        formatted_lines.append("Target variable statistics (for RMSE interpretation):")
        formatted_lines.append(f"  - Range: {metrics.get('target_min', 0):.2f} to {metrics.get('target_max', 0):.2f} (range={metrics.get('target_range', 0):.2f})")
        formatted_lines.append(f"  - Quartiles: Q25={metrics.get('target_q25', 0):.2f}, Median={metrics.get('target_median', 0):.2f}, Q75={metrics.get('target_q75', 0):.2f}")
        formatted_lines.append(f"  - Mean={metrics.get('target_mean', 0):.2f}, Std={metrics.get('target_std', 0):.2f}")
        if 'rmse_pct_of_range' in metrics:
            formatted_lines.append(f"  - RMSE as % of range: {metrics['rmse_pct_of_range']:.1f}%")
        if 'rmse_pct_of_iqr' in metrics:
            formatted_lines.append(f"  - RMSE as % of IQR: {metrics['rmse_pct_of_iqr']:.1f}%")

    formatted_string = "\n".join(formatted_lines)

    logger.info(f"[Ground Truth Evaluation]\n{formatted_string}")

    return formatted_string, metrics  # Return tuple (formatted_string, metrics_dict)
