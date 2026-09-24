"""C题问题2第一阶段：左右视觉提示的多通道特征表示与跨被试验证。

把本文件放在 C:\\C\\src\\06_q2_feature_model.py 后直接运行。

前置条件：已经运行 03_preprocess_erp.py，并生成
    C:\\C\\data\\processed\\erp_epochs\\*_erp_epochs.npz

本程序沿用问题一的356个清洁ERP试验，不重新滤波，也不使用机器给出的
FzDecon、F3Decon、F4Decon。特征只来自原始Fz、F3、F4经问题一预处理后的
提示锁定片段。

特征表示：
1. 时域形态：早期250-500 ms和晚期550-900 ms的均值、标准差、峰峰值、
   面积、正/负峰、峰潜伏期、斜率和线长；
2. 频域调制：delta、theta、alpha、beta功率相对提示前基线的dB变化，
   以及响应期相对功率；
3. 固定侧化：所有试验统一计算F4-F3，不随真实标签翻转符号；
4. 通道协同：Fz-F3、Fz-F4、F3-F4在三个时窗内的零时滞相关、
   ±80 ms内最大互相关及对应时滞。

验证原则：
- 主验证严格按被试外推：A训练/B测试与B训练/A测试；
- Task-1、Task-2分别验证，并给出两任务合并验证；
- 标准化、单变量特征选择和正则化参数选择全部只在训练被试内部完成；
- 被试、任务、文件名、试验编号、行为正确性和标签均不进入特征矩阵；
- 比较时域、时域+频域、完整多通道三组特征，避免只报告最好结果。

输出：
    data/processed/q2_trial_features.csv
    results/06_q2_feature_model/cross_subject_results.csv
    results/06_q2_feature_model/cross_subject_predictions.csv
    results/06_q2_feature_model/bidirectional_summary.csv
    results/06_q2_feature_model/selected_coefficients.csv
    results/06_q2_feature_model/feature_effects.csv
    results/06_q2_feature_model/stable_feature_effects.csv
    results/06_q2_feature_model/feature_group_performance.png
    results/06_q2_feature_model/primary_confusion_matrices.png
    results/06_q2_feature_model/stable_feature_heatmap.png
    results/06_q2_feature_model/q2_feature_model_report.txt

说明：只有两名被试，因此本阶段结果属于“跨被试可迁移性初检”，不能替代
更大样本的群体外部验证，也不能由三个额区电极反推出唯一LGN或皮层源。
"""

from __future__ import annotations

import warnings
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import correlate, correlation_lags, welch
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FILE_TO_EPOCH = {
    "VisualCogA_Task-1.mat": "A_Task-1_erp_epochs.npz",
    "VisualCogA_Task-2.mat": "A_Task-2_erp_epochs.npz",
    "VisualCogB_Task-1.mat": "B_Task-1_erp_epochs.npz",
    "VisualCogB_Task-2.mat": "B_Task-2_erp_epochs.npz",
}
SUBJECT_BY_FILE = {
    "VisualCogA_Task-1.mat": "A",
    "VisualCogA_Task-2.mat": "A",
    "VisualCogB_Task-1.mat": "B",
    "VisualCogB_Task-2.mat": "B",
}
TASK_BY_FILE = {
    "VisualCogA_Task-1.mat": "Task-1",
    "VisualCogA_Task-2.mat": "Task-2",
    "VisualCogB_Task-1.mat": "Task-1",
    "VisualCogB_Task-2.mat": "Task-2",
}
EXPECTED_FILES = tuple(FILE_TO_EPOCH)
EEG_LABELS = ("Fz", "F3", "F4")

TIME_WINDOWS = {
    "early_250_500ms": (0.25, 0.50),
    "late_550_900ms": (0.55, 0.90),
}
COUPLING_WINDOWS = {
    "early_250_500ms": (0.25, 0.50),
    "late_550_900ms": (0.55, 0.90),
    "full_0_900ms": (0.00, 0.90),
}
BASELINE_WINDOW = (-0.50, 0.00)
SPECTRAL_RESPONSE_WINDOW = (0.00, 0.90)
FREQUENCY_BANDS = {
    "delta_1_4Hz": (1.0, 4.0),
    "theta_4_8Hz": (4.0, 8.0),
    "alpha_8_13Hz": (8.0, 13.0),
    "beta_13_30Hz": (13.0, 30.0),
}
CHANNEL_PAIRS = (("Fz", "F3"), ("Fz", "F4"), ("F3", "F4"))

FEATURE_GROUP_PREFIXES = {
    "time_only": ("time__", "laterality_time__"),
    "time_plus_spectral": (
        "time__",
        "laterality_time__",
        "spectral__",
        "laterality_spectral__",
    ),
    "full_multichannel": (
        "time__",
        "laterality_time__",
        "spectral__",
        "laterality_spectral__",
        "coupling__",
    ),
}
SCOPE_ORDER = ("Task-1", "Task-2", "Pooled")
FEATURE_GROUP_ORDER = ("time_only", "time_plus_spectral", "full_multichannel")

RANDOM_SEED = 20260924
INNER_FOLDS = 5
REGULARIZATION_GRID = (0.01, 0.1, 1.0, 10.0)
SELECT_K_CANDIDATES = (10, 20, 30)
N_PERMUTATIONS = 2000
N_BOOTSTRAP = 2000
MAX_LAG_S = 0.080
EPSILON = 1e-12


def deterministic_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return (RANDOM_SEED + zlib.crc32(text)) % (2**32 - 1)


def window_mask(time_s: np.ndarray, limits: tuple[float, float], right_closed: bool = True) -> np.ndarray:
    if right_closed:
        return (time_s >= limits[0]) & (time_s <= limits[1])
    return (time_s >= limits[0]) & (time_s < limits[1])


def safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size < 3 or np.std(first) < EPSILON or np.std(second) < EPSILON:
        return 0.0
    value = float(np.corrcoef(first, second)[0, 1])
    return value if np.isfinite(value) else 0.0


def time_statistics(signal: np.ndarray, local_time: np.ndarray, sample_rate: float) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    local_time = np.asarray(local_time, dtype=np.float64)
    if signal.size < 3:
        raise ValueError("时间窗内采样点不足。")

    centered_time = local_time - np.mean(local_time)
    slope_denominator = float(np.sum(centered_time**2))
    slope = float(np.sum(centered_time * (signal - np.mean(signal))) / max(slope_denominator, EPSILON))
    positive_index = int(np.argmax(signal))
    negative_index = int(np.argmin(signal))

    return {
        "mean": float(np.mean(signal)),
        "std": float(np.std(signal, ddof=0)),
        "peak_to_peak": float(np.ptp(signal)),
        "area": float(np.trapezoid(signal, local_time)),
        "positive_peak": float(signal[positive_index]),
        "positive_peak_latency_s": float(local_time[positive_index]),
        "negative_peak": float(signal[negative_index]),
        "negative_peak_latency_s": float(local_time[negative_index]),
        "slope_per_s": slope,
        "line_length_per_s": float(np.mean(np.abs(np.diff(signal))) * sample_rate),
    }


def spectral_summary(signal: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.float64)
    nperseg = min(128, signal.size)
    if nperseg < 16:
        raise ValueError("频谱时间窗过短。")
    noverlap = nperseg // 2 if signal.size > nperseg else 0
    frequency, psd = welch(
        signal,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=512,
        detrend="constant",
        scaling="density",
    )
    return frequency, psd


def integrate_band(frequency: np.ndarray, psd: np.ndarray, limits: tuple[float, float]) -> float:
    mask = (frequency >= limits[0]) & (frequency < limits[1])
    if np.count_nonzero(mask) < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], frequency[mask]))


def spectral_features(
    baseline: np.ndarray,
    response: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    base_frequency, base_psd = spectral_summary(baseline, sample_rate)
    response_frequency, response_psd = spectral_summary(response, sample_rate)
    total_response = integrate_band(response_frequency, response_psd, (1.0, 30.0))

    output: dict[str, float] = {}
    for band_name, limits in FREQUENCY_BANDS.items():
        base_power = integrate_band(base_frequency, base_psd, limits)
        response_power = integrate_band(response_frequency, response_psd, limits)
        output[f"{band_name}__change_db"] = float(
            10.0 * np.log10((response_power + EPSILON) / (base_power + EPSILON))
        )
        output[f"{band_name}__response_relative"] = float(
            response_power / (total_response + EPSILON)
        )
    return output


def coupling_features(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate: float,
) -> dict[str, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_centered = first - np.mean(first)
    second_centered = second - np.mean(second)
    denominator = float(np.sqrt(np.sum(first_centered**2) * np.sum(second_centered**2)))

    zero_lag = safe_correlation(first, second)
    if denominator <= EPSILON:
        return {
            "zero_lag_correlation": zero_lag,
            "max_abs_cross_correlation": 0.0,
            "lag_at_max_ms": 0.0,
        }

    cross = correlate(first_centered, second_centered, mode="full", method="auto") / denominator
    lags = correlation_lags(first.size, second.size, mode="full")
    max_lag_samples = int(round(MAX_LAG_S * sample_rate))
    allowed = np.abs(lags) <= max_lag_samples
    local_cross = cross[allowed]
    local_lags = lags[allowed]
    best_index = int(np.argmax(np.abs(local_cross)))
    return {
        "zero_lag_correlation": zero_lag,
        "max_abs_cross_correlation": float(local_cross[best_index]),
        "lag_at_max_ms": float(local_lags[best_index] / sample_rate * 1000.0),
    }


def load_epoch_file(epoch_path: Path, file_name: str) -> dict[str, object]:
    with np.load(epoch_path, allow_pickle=False) as archive:
        epochs = np.asarray(archive["epochs_accepted"], dtype=np.float64)
        trial_ids = np.asarray(archive["accepted_trial_id"], dtype=np.int32)
        directions = np.asarray(archive["cue_direction_accepted"]).astype(str)
        behavior = np.asarray(archive["behavior_correct_inferred_accepted"], dtype=np.int8)
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        channel_labels = tuple(np.asarray(archive["channel_labels"]).astype(str).tolist())
        sample_rate = float(np.asarray(archive["sample_rate_hz"]).squeeze())
        filter_band = tuple(np.asarray(archive["filter_band_hz"], dtype=np.float64).tolist())

    if epochs.ndim != 3 or epochs.shape[1] != 3:
        raise ValueError(f"{epoch_path.name} 的epochs_accepted形状异常：{epochs.shape}")
    if channel_labels != EEG_LABELS:
        raise ValueError(f"{epoch_path.name} 的通道顺序为{channel_labels}，预期为{EEG_LABELS}。")
    if epochs.shape[0] != trial_ids.size or epochs.shape[0] != directions.size:
        raise ValueError(f"{epoch_path.name} 的试验索引与片段数量不一致。")
    if not np.isfinite(epochs).all():
        raise ValueError(f"{epoch_path.name} 的清洁片段仍含NaN或Inf。")
    if not np.isclose(sample_rate, 256.0):
        raise ValueError(f"{epoch_path.name} 采样率不是256 Hz。")
    if not np.allclose(filter_band, (0.5, 30.0)):
        raise ValueError(f"{epoch_path.name} 滤波频带不是问题一的0.5-30 Hz。")
    if not set(np.unique(directions)).issubset({"L", "R"}):
        raise ValueError(f"{epoch_path.name} 含未知方向标签。")
    if behavior.size != epochs.shape[0]:
        raise ValueError(f"{epoch_path.name} 的行为标记与片段数量不一致。")

    return {
        "file_name": file_name,
        "subject": SUBJECT_BY_FILE[file_name],
        "task": TASK_BY_FILE[file_name],
        "epochs": epochs,
        "trial_ids": trial_ids,
        "directions": directions,
        "behavior": behavior,
        "time_s": time_s,
        "sample_rate": sample_rate,
    }


def extract_trial_features(file_data: dict[str, object]) -> list[dict[str, object]]:
    epochs = np.asarray(file_data["epochs"], dtype=np.float64)
    time_s = np.asarray(file_data["time_s"], dtype=np.float64)
    sample_rate = float(file_data["sample_rate"])
    channel_index = {label: index for index, label in enumerate(EEG_LABELS)}

    baseline_mask = window_mask(time_s, BASELINE_WINDOW, right_closed=False)
    response_mask = window_mask(time_s, SPECTRAL_RESPONSE_WINDOW)
    if np.count_nonzero(baseline_mask) < 100 or np.count_nonzero(response_mask) < 200:
        raise ValueError("ERP时间轴不能覆盖频谱基线或响应窗。")

    rows: list[dict[str, object]] = []
    for epoch_index in range(epochs.shape[0]):
        epoch = epochs[epoch_index]
        row: dict[str, object] = {
            "file_name": file_data["file_name"],
            "subject": file_data["subject"],
            "task": file_data["task"],
            "trial_id": int(np.asarray(file_data["trial_ids"])[epoch_index]),
            "cue_direction": str(np.asarray(file_data["directions"])[epoch_index]),
            "target_R": int(str(np.asarray(file_data["directions"])[epoch_index]) == "R"),
            "behavior_correct_inferred": int(np.asarray(file_data["behavior"])[epoch_index]),
        }

        # 原始三个额区通道的时域形态。
        for window_name, limits in TIME_WINDOWS.items():
            mask = window_mask(time_s, limits)
            local_time = time_s[mask]
            for channel_name, index in channel_index.items():
                statistics = time_statistics(epoch[index, mask], local_time, sample_rate)
                for statistic_name, value in statistics.items():
                    row[f"time__{channel_name}__{window_name}__{statistic_name}"] = value

            # 固定F4-F3，不依据左右标签改变符号。
            lateral_signal = epoch[channel_index["F4"], mask] - epoch[channel_index["F3"], mask]
            lateral_statistics = time_statistics(lateral_signal, local_time, sample_rate)
            for statistic_name, value in lateral_statistics.items():
                row[f"laterality_time__F4_minus_F3__{window_name}__{statistic_name}"] = value

        # 频带功率相对提示前基线的变化及响应期相对功率。
        channel_spectral: dict[str, dict[str, float]] = {}
        for channel_name, index in channel_index.items():
            values = spectral_features(
                epoch[index, baseline_mask],
                epoch[index, response_mask],
                sample_rate,
            )
            channel_spectral[channel_name] = values
            for feature_name, value in values.items():
                row[f"spectral__{channel_name}__{feature_name}"] = value

        for band_name in FREQUENCY_BANDS:
            for measure in ("change_db", "response_relative"):
                key = f"{band_name}__{measure}"
                row[f"laterality_spectral__F4_minus_F3__{key}"] = (
                    channel_spectral["F4"][key] - channel_spectral["F3"][key]
                )

        # 通道协同特征。
        for window_name, limits in COUPLING_WINDOWS.items():
            mask = window_mask(time_s, limits)
            for first_name, second_name in CHANNEL_PAIRS:
                values = coupling_features(
                    epoch[channel_index[first_name], mask],
                    epoch[channel_index[second_name], mask],
                    sample_rate,
                )
                for feature_name, value in values.items():
                    row[f"coupling__{first_name}_{second_name}__{window_name}__{feature_name}"] = value

        rows.append(row)
    return rows


def get_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    prefixes = tuple({prefix for values in FEATURE_GROUP_PREFIXES.values() for prefix in values})
    return [column for column in feature_table.columns if column.startswith(prefixes)]


def columns_for_group(all_feature_columns: list[str], group_name: str) -> list[str]:
    prefixes = FEATURE_GROUP_PREFIXES[group_name]
    return [column for column in all_feature_columns if column.startswith(prefixes)]


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = matrix.ravel()
    sensitivity = true_positive / max(true_positive + false_negative, 1)
    specificity = true_negative / max(true_negative + false_positive, 1)
    auc = roc_auc_score(y_true, probability) if np.unique(y_true).size == 2 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_R": float(sensitivity),
        "specificity_L": float(specificity),
        "f1_R": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(auc),
        "tn": int(true_negative),
        "fp": int(false_positive),
        "fn": int(false_negative),
        "tp": int(true_positive),
    }


def stratified_permutation_p_value(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> float:
    observed = float(balanced_accuracy_score(y_true, y_pred))
    exceed = 0
    unique_strata = np.unique(strata)
    for _ in range(N_PERMUTATIONS):
        permuted = y_true.copy()
        for stratum in unique_strata:
            indices = np.flatnonzero(strata == stratum)
            permuted[indices] = rng.permutation(permuted[indices])
        null_score = float(balanced_accuracy_score(permuted, y_pred))
        exceed += int(null_score >= observed)
    return float((exceed + 1) / (N_PERMUTATIONS + 1))


def stratified_bootstrap_interval(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    unique_strata = np.unique(strata)
    scores = np.empty(N_BOOTSTRAP, dtype=np.float64)
    for bootstrap_index in range(N_BOOTSTRAP):
        selections: list[np.ndarray] = []
        for stratum in unique_strata:
            indices = np.flatnonzero(strata == stratum)
            selections.append(rng.choice(indices, size=indices.size, replace=True))
        selected = np.concatenate(selections)
        scores[bootstrap_index] = balanced_accuracy_score(y_true[selected], y_pred[selected])
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def scope_mask(feature_table: pd.DataFrame, scope: str) -> np.ndarray:
    if scope == "Pooled":
        return np.ones(len(feature_table), dtype=bool)
    return feature_table["task"].to_numpy(dtype=str) == scope


def run_one_split(
    feature_table: pd.DataFrame,
    all_feature_columns: list[str],
    scope: str,
    feature_group: str,
    train_subject: str,
    test_subject: str,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    in_scope = scope_mask(feature_table, scope)
    train_mask = in_scope & (feature_table["subject"].to_numpy(dtype=str) == train_subject)
    test_mask = in_scope & (feature_table["subject"].to_numpy(dtype=str) == test_subject)
    feature_columns = columns_for_group(all_feature_columns, feature_group)

    train = feature_table.loc[train_mask].reset_index(drop=True)
    test = feature_table.loc[test_mask].reset_index(drop=True)
    x_train = train[feature_columns].to_numpy(dtype=np.float64)
    x_test = test[feature_columns].to_numpy(dtype=np.float64)
    y_train = train["target_R"].to_numpy(dtype=np.int8)
    y_test = test["target_R"].to_numpy(dtype=np.int8)

    if train_subject == test_subject:
        raise AssertionError("训练和测试被试不能相同。")
    if np.unique(y_train).size != 2 or np.unique(y_test).size != 2:
        raise ValueError(f"{scope} {train_subject}->{test_subject} 的训练或测试集合缺少某一方向。")
    if not np.isfinite(x_train).all() or not np.isfinite(x_test).all():
        raise ValueError("特征矩阵含NaN或Inf。")

    nonconstant_count = int(np.count_nonzero(np.var(x_train, axis=0) > 1e-12))
    k_values = sorted({min(value, nonconstant_count) for value in SELECT_K_CANDIDATES})
    k_values = [value for value in k_values if value >= 2]
    if not k_values:
        raise ValueError("训练集合非恒定特征不足。")

    min_class_count = int(np.min(np.bincount(y_train, minlength=2)))
    n_splits = min(INNER_FOLDS, min_class_count)
    if n_splits < 3:
        raise ValueError("训练集合每类样本不足3个，无法完成内部参数选择。")

    seed = deterministic_seed(scope, feature_group, train_subject, test_subject)
    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipeline = Pipeline(
        steps=[
            ("variance", VarianceThreshold(threshold=1e-12)),
            ("scale", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=k_values[0])),
            (
                "model",
                LogisticRegression(
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=seed,
                ),
            ),
        ]
    )
    search = GridSearchCV(
        pipeline,
        param_grid={
            "select__k": k_values,
            "model__C": REGULARIZATION_GRID,
        },
        scoring="balanced_accuracy",
        cv=inner_cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
        return_train_score=False,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message="Features .* are constant")
        warnings.filterwarnings("ignore", message="invalid value encountered")
        search.fit(x_train, y_train)

    probability = search.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.5).astype(np.int8)
    metrics = metric_values(y_test, prediction, probability)
    task_strata = test["task"].astype(str).to_numpy()
    permutation_p = stratified_permutation_p_value(
        y_test,
        prediction,
        task_strata,
        np.random.default_rng(seed + 1),
    )

    result: dict[str, object] = {
        "scope": scope,
        "feature_group": feature_group,
        "train_subject": train_subject,
        "test_subject": test_subject,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_input_features": int(len(feature_columns)),
        "n_selected_features": int(search.best_params_["select__k"]),
        "best_C": float(search.best_params_["model__C"]),
        "inner_cv_balanced_accuracy": float(search.best_score_),
        "test_permutation_p": permutation_p,
        **metrics,
    }

    prediction_table = test[
        ["file_name", "subject", "task", "trial_id", "cue_direction", "target_R", "behavior_correct_inferred"]
    ].copy()
    prediction_table.insert(0, "scope", scope)
    prediction_table.insert(1, "feature_group", feature_group)
    prediction_table.insert(2, "train_subject", train_subject)
    prediction_table["predicted_R"] = prediction
    prediction_table["probability_R"] = probability
    prediction_table["correct_prediction"] = prediction == y_test

    fitted_pipeline = search.best_estimator_
    variance_mask = fitted_pipeline.named_steps["variance"].get_support()
    after_variance = np.asarray(feature_columns, dtype=object)[variance_mask]
    selection_mask = fitted_pipeline.named_steps["select"].get_support()
    selected_names = after_variance[selection_mask]
    coefficients = np.asarray(fitted_pipeline.named_steps["model"].coef_[0], dtype=np.float64)
    coefficient_table = pd.DataFrame(
        {
            "scope": scope,
            "feature_group": feature_group,
            "train_subject": train_subject,
            "test_subject": test_subject,
            "feature": selected_names,
            "standardized_coefficient": coefficients,
            "absolute_coefficient": np.abs(coefficients),
        }
    ).sort_values("absolute_coefficient", ascending=False)

    return result, prediction_table, coefficient_table


def build_bidirectional_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scope in SCOPE_ORDER:
        for feature_group in FEATURE_GROUP_ORDER:
            subset = predictions[
                (predictions["scope"] == scope)
                & (predictions["feature_group"] == feature_group)
            ].copy()
            y_true = subset["target_R"].to_numpy(dtype=np.int8)
            y_pred = subset["predicted_R"].to_numpy(dtype=np.int8)
            probability = subset["probability_R"].to_numpy(dtype=np.float64)
            strata = (subset["subject"].astype(str) + "_" + subset["task"].astype(str)).to_numpy()
            seed = deterministic_seed("summary", scope, feature_group)
            low, high = stratified_bootstrap_interval(
                y_true,
                y_pred,
                strata,
                np.random.default_rng(seed),
            )
            permutation_p = stratified_permutation_p_value(
                y_true,
                y_pred,
                strata,
                np.random.default_rng(seed + 1),
            )
            rows.append(
                {
                    "scope": scope,
                    "feature_group": feature_group,
                    "n_external_predictions": int(len(subset)),
                    "balanced_accuracy_ci_low": low,
                    "balanced_accuracy_ci_high": high,
                    "permutation_p": permutation_p,
                    **metric_values(y_true, y_pred, probability),
                }
            )
    return pd.DataFrame(rows)


def hedges_g(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    degrees = left.size + right.size - 2
    if degrees <= 0:
        return float("nan")
    pooled_variance = (
        (left.size - 1) * np.var(left, ddof=1)
        + (right.size - 1) * np.var(right, ddof=1)
    ) / degrees
    if pooled_variance <= EPSILON:
        return 0.0
    correction = 1.0 - 3.0 / (4.0 * (left.size + right.size) - 9.0)
    return float(correction * (np.mean(right) - np.mean(left)) / np.sqrt(pooled_variance))


def build_feature_effects(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows: list[dict[str, object]] = []
    for task in ("Task-1", "Task-2"):
        for subject in ("A", "B"):
            subset = feature_table[
                (feature_table["task"] == task) & (feature_table["subject"] == subject)
            ]
            left = subset[subset["target_R"] == 0]
            right = subset[subset["target_R"] == 1]
            for feature in feature_columns:
                effect_rows.append(
                    {
                        "task": task,
                        "subject": subject,
                        "feature": feature,
                        "n_left": int(len(left)),
                        "n_right": int(len(right)),
                        "hedges_g_R_minus_L": hedges_g(
                            left[feature].to_numpy(dtype=np.float64),
                            right[feature].to_numpy(dtype=np.float64),
                        ),
                    }
                )
    effects = pd.DataFrame(effect_rows)

    stable_rows: list[dict[str, object]] = []
    for task in ("Task-1", "Task-2"):
        task_effects = effects[effects["task"] == task]
        pivot = task_effects.pivot(index="feature", columns="subject", values="hedges_g_R_minus_L")
        for feature, values in pivot.iterrows():
            g_a = float(values["A"])
            g_b = float(values["B"])
            sign_consistent = bool(np.sign(g_a) == np.sign(g_b) and g_a != 0.0 and g_b != 0.0)
            stable_rows.append(
                {
                    "task": task,
                    "feature": feature,
                    "g_subject_A": g_a,
                    "g_subject_B": g_b,
                    "sign_consistent": sign_consistent,
                    "mean_abs_g": float((abs(g_a) + abs(g_b)) / 2.0),
                    "conservative_abs_g": float(min(abs(g_a), abs(g_b))) if sign_consistent else 0.0,
                }
            )
    stable = pd.DataFrame(stable_rows).sort_values(
        ["task", "conservative_abs_g", "mean_abs_g"],
        ascending=[True, False, False],
    )
    return effects, stable


def plot_feature_group_performance(
    results: pd.DataFrame,
    summary: pd.DataFrame,
    path: Path,
) -> None:
    colors = {
        "time_only": "#4C78A8",
        "time_plus_spectral": "#F2A541",
        "full_multichannel": "#2A9D8F",
    }
    labels = {
        "time_only": "Time",
        "time_plus_spectral": "Time + spectral",
        "full_multichannel": "Full multichannel",
    }
    x = np.arange(len(SCOPE_ORDER), dtype=np.float64)
    width = 0.24
    fig, axis = plt.subplots(figsize=(10.5, 5.8))

    for group_index, feature_group in enumerate(FEATURE_GROUP_ORDER):
        local = summary.set_index(["scope", "feature_group"])
        centers = x + (group_index - 1) * width
        values = np.asarray(
            [local.loc[(scope, feature_group), "balanced_accuracy"] for scope in SCOPE_ORDER],
            dtype=np.float64,
        )
        low = np.asarray(
            [local.loc[(scope, feature_group), "balanced_accuracy_ci_low"] for scope in SCOPE_ORDER]
        )
        high = np.asarray(
            [local.loc[(scope, feature_group), "balanced_accuracy_ci_high"] for scope in SCOPE_ORDER]
        )
        yerr = np.vstack([values - low, high - values])
        axis.bar(
            centers,
            values,
            width=width,
            color=colors[feature_group],
            label=labels[feature_group],
            yerr=yerr,
            capsize=3,
            alpha=0.9,
        )
        for scope_index, scope in enumerate(SCOPE_ORDER):
            directional = results[
                (results["scope"] == scope) & (results["feature_group"] == feature_group)
            ]["balanced_accuracy"].to_numpy(dtype=np.float64)
            offsets = np.linspace(-0.035, 0.035, directional.size)
            axis.scatter(
                centers[scope_index] + offsets,
                directional,
                s=28,
                facecolor="white",
                edgecolor="black",
                linewidth=0.7,
                zorder=4,
            )

    axis.axhline(0.5, color="#555555", linestyle="--", linewidth=1.2, label="Chance")
    axis.set_xticks(x)
    axis.set_xticklabels(SCOPE_ORDER)
    axis.set_ylim(0.25, 0.82)
    axis.set_ylabel("External balanced accuracy")
    axis.set_title("Bidirectional cross-subject validation")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_primary_confusion_matrices(predictions: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8))
    for axis, scope in zip(axes, SCOPE_ORDER):
        subset = predictions[
            (predictions["scope"] == scope)
            & (predictions["feature_group"] == "full_multichannel")
        ]
        matrix = confusion_matrix(
            subset["target_R"].to_numpy(dtype=np.int8),
            subset["predicted_R"].to_numpy(dtype=np.int8),
            labels=[0, 1],
        )
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=max(int(matrix.max()), 1))
        for row in range(2):
            for column in range(2):
                axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=13)
        axis.set_xticks([0, 1], ["Pred L", "Pred R"])
        axis.set_yticks([0, 1], ["True L", "True R"])
        axis.set_title(scope)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Full multichannel features: external predictions", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def short_feature_name(feature: str) -> str:
    replacements = {
        "laterality_time__F4_minus_F3__": "LatTime/",
        "laterality_spectral__F4_minus_F3__": "LatSpec/",
        "time__": "Time/",
        "spectral__": "Spec/",
        "coupling__": "Coupling/",
        "early_250_500ms": "early",
        "late_550_900ms": "late",
        "full_0_900ms": "full",
    }
    output = feature
    for old, new in replacements.items():
        output = output.replace(old, new)
    return output.replace("__", "/")


def plot_stable_feature_heatmap(
    effects: pd.DataFrame,
    stable: pd.DataFrame,
    path: Path,
) -> None:
    selected: list[str] = []
    for task in ("Task-1", "Task-2"):
        local = stable[(stable["task"] == task) & stable["sign_consistent"]].head(6)
        selected.extend(local["feature"].tolist())
    selected = list(dict.fromkeys(selected))
    if not selected:
        selected = stable.sort_values("mean_abs_g", ascending=False).head(10)["feature"].tolist()

    columns = (("Task-1", "A"), ("Task-1", "B"), ("Task-2", "A"), ("Task-2", "B"))
    matrix = np.full((len(selected), len(columns)), np.nan, dtype=np.float64)
    for row_index, feature in enumerate(selected):
        for column_index, (task, subject) in enumerate(columns):
            local = effects[
                (effects["feature"] == feature)
                & (effects["task"] == task)
                & (effects["subject"] == subject)
            ]
            if not local.empty:
                matrix[row_index, column_index] = float(local.iloc[0]["hedges_g_R_minus_L"])

    finite = np.abs(matrix[np.isfinite(matrix)])
    limit = max(float(np.percentile(finite, 95)) if finite.size else 0.5, 0.3)
    fig_height = max(5.0, 0.45 * len(selected) + 1.8)
    fig, axis = plt.subplots(figsize=(10.5, fig_height))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    axis.set_xticks(np.arange(4), ["A Task-1", "B Task-1", "A Task-2", "B Task-2"])
    axis.set_yticks(np.arange(len(selected)), [short_feature_name(value) for value in selected])
    axis.set_title("Descriptive feature effects (Hedges g, R - L)")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(column, row, f"{value:+.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="Hedges g")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def build_report(
    feature_table: pd.DataFrame,
    feature_columns: list[str],
    results: pd.DataFrame,
    summary: pd.DataFrame,
    stable: pd.DataFrame,
) -> str:
    primary = summary[
        (summary["scope"] == "Pooled")
        & (summary["feature_group"] == "full_multichannel")
    ].iloc[0]
    best = summary.sort_values(["balanced_accuracy", "roc_auc"], ascending=False).iloc[0]
    minimum_class = int(
        feature_table.groupby(["subject", "task", "cue_direction"], observed=True).size().min()
    )
    split_disjoint = bool(
        np.all(results["train_subject"].to_numpy() != results["test_subject"].to_numpy())
    )
    all_finite = bool(np.isfinite(feature_table[feature_columns].to_numpy(dtype=np.float64)).all())
    forbidden_in_features = any(
        token in column.lower()
        for column in feature_columns
        for token in ("direction", "target_r", "subject", "task", "file_name", "behavior", "decon")
    )

    lines = [
        "C题问题2第一阶段：多通道特征表示与跨被试验证报告",
        "=" * 68,
        "",
        "一、数据与特征",
        f"清洁试验数：{len(feature_table)}",
        f"候选特征数：{len(feature_columns)}",
        f"每个被试-任务-方向的最少试验数：{minimum_class}",
        "输入通道：仅Fz、F3、F4；未使用FzDecon、F3Decon、F4Decon。",
        "标签方向：L=0，R=1；标签只用于训练和评估，不参与特征构造。",
        "",
        "二、验证设计",
        "外部验证：A训练/B测试与B训练/A测试。",
        "内部选择：仅在训练被试内用5折分层交叉验证选择特征数和L2正则化强度。",
        "比较口径：time_only、time_plus_spectral、full_multichannel三组均完整报告。",
        "置换检验：在被试-任务层内置换真实标签2000次。",
        "",
        "三、双向跨被试汇总",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"[{row.scope} | {row.feature_group}] n={row.n_external_predictions}, "
            f"BAC={row.balanced_accuracy:.4f} "
            f"(95% CI {row.balanced_accuracy_ci_low:.4f}-{row.balanced_accuracy_ci_high:.4f}), "
            f"AUC={row.roc_auc:.4f}, F1={row.f1_R:.4f}, MCC={row.mcc:.4f}, "
            f"permutation p={row.permutation_p:.4f}"
        )

    primary_significant = bool(primary["balanced_accuracy"] > 0.5 and primary["permutation_p"] < 0.05)
    lines.extend(
        [
            "",
            "四、主结果判读",
            f"主模型（Pooled | full_multichannel）BAC={primary['balanced_accuracy']:.4f}, "
            f"95% CI {primary['balanced_accuracy_ci_low']:.4f}-{primary['balanced_accuracy_ci_high']:.4f}, "
            f"AUC={primary['roc_auc']:.4f}, permutation p={primary['permutation_p']:.4f}。",
            (
                "主模型在双向跨被试检验中高于机会水平，可作为当前数据下的初步可迁移证据。"
                if primary_significant
                else "主模型尚未形成显著的跨被试区分证据；应如实报告，并把特征表示解释为候选机制特征，而非已验证生物标志物。"
            ),
            f"本轮最高BAC来自 {best['scope']} | {best['feature_group']}："
            f"BAC={best['balanced_accuracy']:.4f}, p={best['permutation_p']:.4f}。",
            "",
            "五、描述性稳定特征（不用于替代外部验证）",
        ]
    )
    for task in ("Task-1", "Task-2"):
        top = stable[(stable["task"] == task) & stable["sign_consistent"]].head(5)
        lines.append(f"{task}：")
        if top.empty:
            lines.append("  没有在A、B两名被试中方向一致的候选特征。")
        else:
            for row in top.itertuples(index=False):
                lines.append(
                    f"  {row.feature}: g_A={row.g_subject_A:+.3f}, "
                    f"g_B={row.g_subject_B:+.3f}, conservative |g|={row.conservative_abs_g:.3f}"
                )

    lines.extend(
        [
            "",
            "自动验收",
            "-" * 68,
            f"{pass_fail(len(feature_table) == 356)}  应读取问题一的356个清洁试验",
            f"{pass_fail(feature_table[['file_name', 'trial_id']].drop_duplicates().shape[0] == len(feature_table))}  文件内试验编号应唯一",
            f"{pass_fail(all_finite)}  全部候选特征应为有限数",
            f"{pass_fail(not forbidden_in_features)}  特征列不得包含标签/被试/任务/行为/Decon信息",
            f"{pass_fail(split_disjoint)}  所有外部验证的训练与测试被试应完全分离",
            f"{pass_fail(set(feature_table['cue_direction']) == {'L', 'R'})}  方向标签应仅包含L和R",
            f"{pass_fail(minimum_class >= 40)}  每个被试-任务-方向至少应有40个清洁试验",
            f"{pass_fail(len(summary) == 9)}  应得到3个任务范围乘3组特征的9项双向汇总",
            "",
            "边界说明",
            "1. 频带特征来自短时ERP片段，只用于区分性表征，不作稳态节律诊断。",
            "2. 三个额区电极不足以唯一定位LGN、V1或左右半球源；机制模型必须表述为受约束的现象学模型。",
            "3. 仅有两名被试，A<->B验证是严格但小样本的外推初检，后续仍需更多被试验证。",
            "4. 若分类未显著高于机会水平，这也是有效结果：说明当前额区数据不能支持可靠左右解码，不能选择性汇报内部折数成绩。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    epoch_dir = project_root / "data" / "processed" / "erp_epochs"
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "results" / "06_q2_feature_model"
    processed_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        str(epoch_dir / epoch_name)
        for epoch_name in FILE_TO_EPOCH.values()
        if not (epoch_dir / epoch_name).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "缺少问题一生成的ERP分段文件，请先运行03_preprocess_erp.py：\n"
            + "\n".join(missing)
        )

    feature_rows: list[dict[str, object]] = []
    print("正在提取逐试验多通道特征……")
    for file_name in EXPECTED_FILES:
        epoch_path = epoch_dir / FILE_TO_EPOCH[file_name]
        file_data = load_epoch_file(epoch_path, file_name)
        rows = extract_trial_features(file_data)
        feature_rows.extend(rows)
        print(f"  {file_name}: {len(rows)}个清洁试验")

    feature_table = pd.DataFrame(feature_rows)
    feature_columns = get_feature_columns(feature_table)
    if len(feature_columns) < 80:
        raise ValueError(f"候选特征数异常偏少：{len(feature_columns)}")
    if not np.isfinite(feature_table[feature_columns].to_numpy(dtype=np.float64)).all():
        raise ValueError("特征提取后仍含NaN或Inf。")

    feature_path = processed_dir / "q2_trial_features.csv"
    feature_table.to_csv(feature_path, index=False, encoding="utf-8-sig", float_format="%.9g")

    result_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    coefficient_frames: list[pd.DataFrame] = []
    print("正在进行严格的双向跨被试验证……")
    for scope in SCOPE_ORDER:
        for feature_group in FEATURE_GROUP_ORDER:
            for train_subject, test_subject in (("A", "B"), ("B", "A")):
                result, predictions, coefficients = run_one_split(
                    feature_table,
                    feature_columns,
                    scope,
                    feature_group,
                    train_subject,
                    test_subject,
                )
                result_rows.append(result)
                prediction_frames.append(predictions)
                coefficient_frames.append(coefficients)
                print(
                    f"  {scope:6s} | {feature_group:18s} | "
                    f"{train_subject}->{test_subject}: "
                    f"BAC={result['balanced_accuracy']:.3f}, AUC={result['roc_auc']:.3f}"
                )

    results = pd.DataFrame(result_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    coefficients = pd.concat(coefficient_frames, ignore_index=True)
    summary = build_bidirectional_summary(predictions)
    effects, stable = build_feature_effects(feature_table, feature_columns)

    results_path = output_dir / "cross_subject_results.csv"
    predictions_path = output_dir / "cross_subject_predictions.csv"
    summary_path = output_dir / "bidirectional_summary.csv"
    coefficients_path = output_dir / "selected_coefficients.csv"
    effects_path = output_dir / "feature_effects.csv"
    stable_path = output_dir / "stable_feature_effects.csv"
    report_path = output_dir / "q2_feature_model_report.txt"

    results.to_csv(results_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    coefficients.to_csv(coefficients_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    effects.to_csv(effects_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    stable.to_csv(stable_path, index=False, encoding="utf-8-sig", float_format="%.8g")

    plot_feature_group_performance(results, summary, output_dir / "feature_group_performance.png")
    plot_primary_confusion_matrices(predictions, output_dir / "primary_confusion_matrices.png")
    plot_stable_feature_heatmap(effects, stable, output_dir / "stable_feature_heatmap.png")

    report = build_report(feature_table, feature_columns, results, summary, stable)
    report_path.write_text(report, encoding="utf-8")

    print("\n问题2第一阶段完成。")
    print(f"清洁试验数：{len(feature_table)}")
    print(f"候选特征数：{len(feature_columns)}")
    print(f"逐试验特征：{feature_path}")
    print(f"双向汇总：{summary_path}")
    print(f"检查报告：{report_path}")
    print("\n请先打开 q2_feature_model_report.txt，确认自动验收项目均显示 PASS。")


if __name__ == "__main__":
    main()
