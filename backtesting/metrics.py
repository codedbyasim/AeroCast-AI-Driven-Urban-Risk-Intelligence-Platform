"""
AeroCast Statistical Evaluation & Backtesting Metrics.
"""

from typing import Dict, Any, List, Tuple
import numpy as np


def calculate_continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression metrics with numerical stability safeguards.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            "sample_count": 0,
            "mae": 0.0,
            "rmse": 0.0,
            "r2": 0.0,
            "mape": 0.0,
            "median_ae": 0.0,
            "max_error": 0.0,
        }

    residuals = y_true - y_pred
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 1e-9 else 0.0

    # MAPE with epsilon protection
    eps = 1e-3
    mape = float(np.mean(np.abs(residuals) / np.maximum(np.abs(y_true), eps)) * 100.0)
    median_ae = float(np.median(np.abs(residuals)))
    max_error = float(np.max(np.abs(residuals)))

    return {
        "sample_count": int(len(y_true)),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "r2": round(r2, 4),
        "mape_percent": round(mape, 2),
        "median_ae": round(median_ae, 3),
        "max_error": round(max_error, 3),
    }


def calculate_extreme_event_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 100.0
) -> Dict[str, Any]:
    """
    Computes binary confusion matrix and classification metrics for extreme hazard spike events.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    actual_positives = y_true >= threshold
    predicted_positives = y_pred >= threshold

    tp = int(np.sum(actual_positives & predicted_positives))
    fp = int(np.sum((~actual_positives) & predicted_positives))
    fn = int(np.sum(actual_positives & (~predicted_positives)))
    tn = int(np.sum((~actual_positives) & (~predicted_positives)))

    total_actual_pos = tp + fn
    total_pred_pos = tp + fp

    recall = float(tp / total_actual_pos) if total_actual_pos > 0 else 1.0
    precision = float(tp / total_pred_pos) if total_pred_pos > 0 else 1.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 1.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    false_negative_rate = float(fn / total_actual_pos) if total_actual_pos > 0 else 0.0

    return {
        "threshold_ug_m3": threshold,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "actual_event_count": total_actual_pos,
        "recall_sensitivity": round(recall, 4),
        "precision": round(precision, 4),
        "specificity": round(specificity, 4),
        "f1_score": round(f1, 4),
        "false_negative_miss_rate": round(false_negative_rate, 4),
    }


def calculate_directional_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_current: np.ndarray
) -> float:
    """
    Calculates the proportion of times the model correctly predicts the trajectory sign (rising vs falling).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_current = np.asarray(y_current, dtype=float)

    actual_dir = np.sign(y_true - y_current)
    pred_dir = np.sign(y_pred - y_current)

    correct = np.sum(actual_dir == pred_dir)
    total = len(actual_dir)

    return round(float(correct / total) if total > 0 else 1.0, 4)


def detect_model_drift(
    current_mae: float,
    current_r2: float,
    baseline_mae: float = 20.93,
    max_allowed_mae: float = 28.0,
    min_allowed_r2: float = 0.25,
) -> Dict[str, Any]:
    """
    Evaluates whether recent error metrics indicate statistical drift or degradation.
    """
    mae_degradation = current_mae - baseline_mae
    is_mae_breach = current_mae > max_allowed_mae
    is_r2_breach = current_r2 < min_allowed_r2

    if is_mae_breach or is_r2_breach:
        status = "CRITICAL_DRIFT"
        recommendation = "Immediate retraining recommended. Model error exceeds operational tolerance."
    elif mae_degradation > 4.0:
        status = "DEGRADED"
        recommendation = "Performance is slightly degraded. Monitor incoming telemetry."
    else:
        status = "HEALTHY"
        recommendation = "Model accuracy is well within validated baseline parameters."

    return {
        "drift_status": status,
        "current_mae": round(current_mae, 3),
        "baseline_mae": round(baseline_mae, 3),
        "current_r2": round(current_r2, 4),
        "mae_degradation": round(mae_degradation, 3),
        "is_retraining_required": (status == "CRITICAL_DRIFT"),
        "recommendation": recommendation,
    }
