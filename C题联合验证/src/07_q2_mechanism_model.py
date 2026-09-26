"""C题问题2第二阶段：LGN—皮层—头皮受约束脑电形成机制模型。

把本文件放在 C:\\C\\src\\07_q2_mechanism_model.py 后直接运行。

前置条件：已经运行 03_preprocess_erp.py，并生成
    C:\\C\\data\\processed\\erp_epochs\\*_erp_epochs.npz

本程序只读取问题一保留下来的356个清洁提示锁定试验，只使用原始Fz、F3、F4
经0.5—30 Hz预处理后的片段，不读取FzDecon、F3Decon、F4Decon，也不会修改
原始.mat文件。

模型思想
--------
由于只有三个额区头皮电极，LGN和具体皮层源无法被唯一反演。本程序建立一个
受约束的“现象学生理前向模型”，而不是声称完成源定位：

1. LGN中继：视觉提示到达LGN后，由两个一阶稳定环节构成约50 ms峰值的中继响应；
2. 皮层前馈：LGN响应通过三个较慢环节形成约130 ms的早期皮层群体活动；
3. 皮层复发/P300：四级慢反馈形成约310 ms的P300相关潜变量；
4. 晚期情境活动：三级更慢反馈形成约670 ms的晚期潜变量；
5. 头皮观测：四个潜变量经过带符号的有效增益线性混合到Fz、F3、F4。

四条动力学基函数对所有被试、任务、方向完全相同。左右方向只允许改变各阶段
到各电极的有效增益，从而把“时间动力学”和“方向表征”分开，减少过拟合。

模型形式为
    tau_g * dg/dt = -g + u(t-delta_g)
    x_c = H_c(g),  x_p = H_p(x_c),  x_l = H_l(x_p)
    y_(e,d)(t) = sum_k a_(e,d,k) x_k(t) + b_(e,d) + c_(e,d)t + epsilon

其中k依次为LGN、早期皮层、P300复发和晚期情境阶段；a是“源活动强度×
容积传导”的有效增益，不能单独解释为真实源强度或解剖连接强度。

验证原则
--------
- 用固定动力学基函数拟合每个清洁试验，标签不参与单试验特征构造；
- 用Huber M估计汇总条件增益，使用1000次分层bootstrap给出R-L增益差区间；
- 用2000次文件内标签置换检验12维机制特征是否存在整体左右差异，并做BH-FDR；
- 用200次分层拆半验证检验机制曲线能否预测未参与估计的试验平均波形；
- 如已运行06，会把严格跨被试分类结果一并写入报告，避免用拟合优度代替解码证据。

输出
----
    data/processed/q2_mechanism_trial_features.csv
    results/07_q2_mechanism_model/mechanism_stage_gains.csv
    results/07_q2_mechanism_model/direction_gain_differences.csv
    results/07_q2_mechanism_model/direction_model_tests.csv
    results/07_q2_mechanism_model/model_fit_summary.csv
    results/07_q2_mechanism_model/stable_mechanism_features.csv
    results/07_q2_mechanism_model/mechanism_schematic.png
    results/07_q2_mechanism_model/mechanism_basis.png
    results/07_q2_mechanism_model/mechanism_fit_curves.png
    results/07_q2_mechanism_model/mechanism_gain_heatmap.png
    results/07_q2_mechanism_model/model_validation.png
    results/07_q2_mechanism_model/q2_mechanism_report.txt
"""

from __future__ import annotations

import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.signal import lfilter, savgol_filter


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
FILE_DISPLAY = {
    "VisualCogA_Task-1.mat": "A Task-1",
    "VisualCogA_Task-2.mat": "A Task-2",
    "VisualCogB_Task-1.mat": "B Task-1",
    "VisualCogB_Task-2.mat": "B Task-2",
}
EXPECTED_FILES = tuple(FILE_TO_EPOCH)
EEG_LABELS = ("Fz", "F3", "F4")
DIRECTIONS = ("L", "R")

STAGE_NAMES = (
    "LGN_relay",
    "cortical_feedforward",
    "P300_recurrent",
    "late_context",
)
STAGE_SHORT = ("LGN", "Cortex", "P300", "Late")
STAGE_COLORS = ("#6A51A3", "#3182BD", "#31A354", "#E6550D")
DIRECTION_COLORS = {"L": "#3366CC", "R": "#DC3912"}

EXPECTED_SAMPLE_RATE = 256.0
EXPECTED_FILTER_BAND = (0.5, 30.0)
FIT_WINDOW_S = (0.0, 0.90)
SMOOTH_WINDOW_S = 0.080
SMOOTH_POLYORDER = 3

# 固定动力学常数。固定而不逐文件搜索，是为了避免用时间参数追逐噪声。
LGN_DELAY_S = 0.035
LGN_TAU_S = 0.015
LGN_ORDER = 2
CORTEX_TAU_S = 0.035
CORTEX_ORDER = 3
P300_TAU_S = 0.050
P300_ORDER = 4
LATE_TAU_S = 0.150
LATE_ORDER = 3

RIDGE_ALPHA = 1.0
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 2000
N_SPLIT_HALF = 200
RANDOM_SEED = 20260924

HUBER_C = 1.345
HUBER_MAX_ITER = 30
HUBER_TOLERANCE = 1e-7
EPSILON = 1e-12


def deterministic_seed(*parts: object) -> int:
    text = "|".join(str(part) for part in parts).encode("utf-8")
    return (RANDOM_SEED + zlib.crc32(text)) % (2**32 - 1)


def pass_fail(condition: bool) -> str:
    return "PASS" if condition else "FAIL"


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    if p.size == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * p.size / np.arange(1, p.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def huber_location(samples: np.ndarray) -> np.ndarray:
    """沿第0维计算Huber M位置估计。"""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("Huber估计至少需要两个样本。")

    location = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - location), axis=0)
    positive_scale = scale[scale > 1e-10]
    fallback = float(np.median(positive_scale)) if positive_scale.size else 1.0
    scale = np.where(scale > 1e-10, scale, fallback)

    for _ in range(HUBER_MAX_ITER):
        standardized = (values - location) / scale
        absolute = np.abs(standardized)
        weights = np.ones_like(absolute)
        outside = absolute > HUBER_C
        weights[outside] = HUBER_C / np.maximum(absolute[outside], EPSILON)
        denominator = np.sum(weights, axis=0)
        updated = np.sum(weights * values, axis=0) / np.maximum(denominator, EPSILON)
        if np.max(np.abs(updated - location)) < HUBER_TOLERANCE:
            location = updated
            break
        location = updated
    return location


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
    if pooled_variance <= 1e-20:
        return 0.0
    d = (np.mean(right) - np.mean(left)) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (left.size + right.size) - 9.0)
    return float(correction * d)


def load_epoch_file(epoch_path: Path, file_name: str) -> dict[str, object]:
    with np.load(epoch_path, allow_pickle=False) as archive:
        epochs = np.asarray(archive["epochs_accepted"], dtype=np.float64)
        trial_ids = np.asarray(archive["accepted_trial_id"], dtype=np.int32)
        directions = np.asarray(archive["cue_direction_accepted"]).astype(str)
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        channel_labels = tuple(np.asarray(archive["channel_labels"]).astype(str).tolist())
        sample_rate = float(np.asarray(archive["sample_rate_hz"]).squeeze())
        filter_band = tuple(np.asarray(archive["filter_band_hz"], dtype=np.float64).tolist())

    if epochs.ndim != 3 or epochs.shape[1] != 3:
        raise ValueError(f"{epoch_path.name} 的epochs_accepted形状异常：{epochs.shape}")
    if epochs.shape[0] != trial_ids.size or epochs.shape[0] != directions.size:
        raise ValueError(f"{epoch_path.name} 的试验索引与片段数量不一致。")
    if channel_labels != EEG_LABELS:
        raise ValueError(f"{epoch_path.name} 的通道顺序为{channel_labels}，预期为{EEG_LABELS}。")
    if not np.isclose(sample_rate, EXPECTED_SAMPLE_RATE):
        raise ValueError(f"{epoch_path.name} 的采样率不是256 Hz。")
    if not np.allclose(filter_band, EXPECTED_FILTER_BAND):
        raise ValueError(f"{epoch_path.name} 的滤波频带不是0.5—30 Hz。")
    if not np.isfinite(epochs).all():
        raise ValueError(f"{epoch_path.name} 的清洁片段仍含NaN或Inf。")
    if set(np.unique(directions)) != {"L", "R"}:
        raise ValueError(f"{epoch_path.name} 没有同时包含L和R。")

    return {
        "file_name": file_name,
        "subject": SUBJECT_BY_FILE[file_name],
        "task": TASK_BY_FILE[file_name],
        "epochs": epochs,
        "trial_ids": trial_ids,
        "directions": directions,
        "time_s": time_s,
        "sample_rate": sample_rate,
    }


def first_order_filter(signal: np.ndarray, tau_s: float, sample_rate: float) -> np.ndarray:
    """精确离散化的一阶稳定低通：tau dx/dt=-x+u。"""
    alpha = float(np.exp(-1.0 / (sample_rate * tau_s)))
    return lfilter([1.0 - alpha], [1.0, -alpha], np.asarray(signal, dtype=np.float64))


def cascade_filter(
    signal: np.ndarray,
    tau_s: float,
    order: int,
    sample_rate: float,
) -> np.ndarray:
    output = np.asarray(signal, dtype=np.float64).copy()
    for _ in range(order):
        output = first_order_filter(output, tau_s, sample_rate)
    return output


def build_mechanistic_basis(
    time_s: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float], float]:
    """返回拟合窗、四条单位峰值潜变量、完整设计矩阵、峰潜伏期和条件数。"""
    time_s = np.asarray(time_s, dtype=np.float64)
    dt = 1.0 / sample_rate
    impulse = np.zeros_like(time_s)
    impulse_index = int(np.argmin(np.abs(time_s - LGN_DELAY_S)))
    impulse[impulse_index] = 1.0 / dt

    lgn = cascade_filter(impulse, LGN_TAU_S, LGN_ORDER, sample_rate)
    cortex = cascade_filter(lgn, CORTEX_TAU_S, CORTEX_ORDER, sample_rate)
    p300 = cascade_filter(cortex, P300_TAU_S, P300_ORDER, sample_rate)
    late = cascade_filter(p300, LATE_TAU_S, LATE_ORDER, sample_rate)

    states_full = np.column_stack([lgn, cortex, p300, late])
    maximum = np.max(np.abs(states_full), axis=0)
    if np.any(maximum <= EPSILON):
        raise ValueError("机制基函数构造失败。")
    states_full /= maximum

    fit_mask = (time_s >= FIT_WINDOW_S[0]) & (time_s <= FIT_WINDOW_S[1])
    fit_time = time_s[fit_mask]
    states = states_full[fit_mask]
    scaled_time = (fit_time - np.mean(fit_time)) / max(np.ptp(fit_time) / 2.0, EPSILON)
    design = np.column_stack([states, np.ones(fit_time.size), scaled_time])
    peak_latency = {
        stage: float(fit_time[int(np.argmax(states[:, index]))])
        for index, stage in enumerate(STAGE_NAMES)
    }
    condition_number = float(np.linalg.cond(design))
    return fit_mask, states, design, peak_latency, condition_number


def smoothing_window_samples(sample_rate: float, n_time: int) -> int:
    window = int(round(SMOOTH_WINDOW_S * sample_rate))
    if window % 2 == 0:
        window += 1
    window = max(window, SMOOTH_POLYORDER + 2)
    if window % 2 == 0:
        window += 1
    if window >= n_time:
        window = n_time - 1 if n_time % 2 == 0 else n_time
    if window % 2 == 0:
        window -= 1
    return window


def ridge_projection(design: np.ndarray) -> np.ndarray:
    """构造固定线性投影；四个机制增益受轻度岭约束，截距和趋势不惩罚。"""
    penalty = np.diag([RIDGE_ALPHA] * len(STAGE_NAMES) + [0.0, 0.0])
    system = design.T @ design + penalty
    return np.linalg.solve(system, design.T)


def trial_coefficients(
    epochs: np.ndarray,
    fit_mask: np.ndarray,
    projection: np.ndarray,
    sample_rate: float,
) -> tuple[np.ndarray, np.ndarray]:
    window = smoothing_window_samples(sample_rate, epochs.shape[-1])
    smoothed = savgol_filter(
        epochs,
        window_length=window,
        polyorder=SMOOTH_POLYORDER,
        axis=-1,
        mode="interp",
    )
    fit_epochs = smoothed[:, :, fit_mask]
    # projection: (parameter,time); fit_epochs: (trial,channel,time)
    coefficients = np.einsum("pt,nct->npc", projection, fit_epochs)
    return coefficients, fit_epochs


def curve_metrics(observed: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    observed = np.asarray(observed, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    residual = observed - fitted
    denominator = float(np.sum((observed - np.mean(observed, axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / max(denominator, EPSILON)
    rmse = float(np.sqrt(np.mean(residual**2)))
    if np.std(observed) <= EPSILON or np.std(fitted) <= EPSILON:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(observed.ravel(), fitted.ravel())[0, 1])
    return {"r2": r2, "rmse": rmse, "correlation": correlation}


def split_half_validation(
    coefficients: np.ndarray,
    fit_epochs: np.ndarray,
    labels: np.ndarray,
    design: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, dict[str, tuple[float, float, float]]]:
    output: dict[str, dict[str, tuple[float, float, float]]] = {}
    for direction in DIRECTIONS:
        indices = np.flatnonzero(labels == direction)
        r2_values = np.empty(N_SPLIT_HALF, dtype=np.float64)
        correlation_values = np.empty(N_SPLIT_HALF, dtype=np.float64)
        for repeat in range(N_SPLIT_HALF):
            shuffled = rng.permutation(indices)
            split = max(2, shuffled.size // 2)
            train_indices = shuffled[:split]
            test_indices = shuffled[split:]
            if test_indices.size < 2:
                test_indices = shuffled[-2:]
                train_indices = shuffled[:-2]

            train_coef = np.mean(coefficients[train_indices], axis=0)
            prediction = design @ train_coef
            observed = np.mean(fit_epochs[test_indices], axis=0).T
            metrics = curve_metrics(observed, prediction)
            r2_values[repeat] = metrics["r2"]
            correlation_values[repeat] = metrics["correlation"]

        r2_median, r2_low, r2_high = np.percentile(r2_values, [50.0, 2.5, 97.5])
        corr_median, corr_low, corr_high = np.percentile(
            correlation_values, [50.0, 2.5, 97.5]
        )
        output[direction] = {
            "r2": (float(r2_median), float(r2_low), float(r2_high)),
            "correlation": (float(corr_median), float(corr_low), float(corr_high)),
        }
    return output


def bootstrap_gain_statistics(
    stage_coefficients: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    indices = {direction: np.flatnonzero(labels == direction) for direction in DIRECTIONS}
    aggregate = {
        direction: huber_location(stage_coefficients[indices[direction]])
        for direction in DIRECTIONS
    }
    boot_condition = {
        direction: np.empty(
            (N_BOOTSTRAP, len(STAGE_NAMES), len(EEG_LABELS)), dtype=np.float64
        )
        for direction in DIRECTIONS
    }
    boot_difference = np.empty(
        (N_BOOTSTRAP, len(STAGE_NAMES), len(EEG_LABELS)), dtype=np.float64
    )

    for bootstrap_index in range(N_BOOTSTRAP):
        selected: dict[str, np.ndarray] = {}
        for direction in DIRECTIONS:
            local = indices[direction]
            resampled = local[rng.integers(0, local.size, size=local.size)]
            selected[direction] = huber_location(stage_coefficients[resampled])
            boot_condition[direction][bootstrap_index] = selected[direction]
        boot_difference[bootstrap_index] = selected["R"] - selected["L"]

    condition_low = {
        direction: np.percentile(boot_condition[direction], 2.5, axis=0)
        for direction in DIRECTIONS
    }
    condition_high = {
        direction: np.percentile(boot_condition[direction], 97.5, axis=0)
        for direction in DIRECTIONS
    }
    difference_low = np.percentile(boot_difference, 2.5, axis=0)
    difference_high = np.percentile(boot_difference, 97.5, axis=0)
    return aggregate, condition_low, condition_high, np.stack(
        [difference_low, difference_high], axis=0
    )


def multivariate_direction_statistic(features: np.ndarray, labels: np.ndarray) -> float:
    left = features[labels == "L"]
    right = features[labels == "R"]
    mean_difference = np.mean(right, axis=0) - np.mean(left, axis=0)
    standard_error_variance = (
        np.var(left, axis=0, ddof=1) / left.shape[0]
        + np.var(right, axis=0, ddof=1) / right.shape[0]
    )
    positive = standard_error_variance[standard_error_variance > 1e-12]
    fallback = float(np.median(positive)) if positive.size else 1.0
    standard_error_variance = np.where(
        standard_error_variance > 1e-12,
        standard_error_variance,
        fallback,
    )
    return float(np.sum(mean_difference**2 / standard_error_variance))


def permutation_direction_test(
    features: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    observed = multivariate_direction_statistic(features, labels)
    exceed = 0
    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(labels)
        statistic = multivariate_direction_statistic(features, permuted)
        exceed += int(statistic >= observed)
    p_value = (exceed + 1) / (N_PERMUTATIONS + 1)
    return observed, float(p_value)


def analyse_file(
    file_data: dict[str, object],
    fit_mask: np.ndarray,
    design: np.ndarray,
    projection: np.ndarray,
) -> dict[str, object]:
    file_name = str(file_data["file_name"])
    epochs = np.asarray(file_data["epochs"], dtype=np.float64)
    labels = np.asarray(file_data["directions"]).astype(str)
    trial_ids = np.asarray(file_data["trial_ids"], dtype=np.int32)
    sample_rate = float(file_data["sample_rate"])

    coefficients, fit_epochs = trial_coefficients(
        epochs, fit_mask, projection, sample_rate
    )
    stage_coefficients = coefficients[:, : len(STAGE_NAMES), :]

    rng_bootstrap = np.random.default_rng(deterministic_seed(file_name, "bootstrap"))
    aggregate, condition_low, condition_high, difference_interval = (
        bootstrap_gain_statistics(stage_coefficients, labels, rng_bootstrap)
    )

    validation = split_half_validation(
        coefficients,
        fit_epochs,
        labels,
        design,
        np.random.default_rng(deterministic_seed(file_name, "split_half")),
    )

    gain_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    observed_curves: dict[str, np.ndarray] = {}
    fitted_curves: dict[str, np.ndarray] = {}

    for direction in DIRECTIONS:
        direction_mask = labels == direction
        robust_full_coef = huber_location(coefficients[direction_mask])
        fitted = design @ robust_full_coef
        observed = huber_location(fit_epochs[direction_mask]).T
        observed_curves[direction] = observed
        fitted_curves[direction] = fitted
        metrics = curve_metrics(observed, fitted)
        split_median, split_low, split_high = validation[direction]["r2"]
        split_corr_median, split_corr_low, split_corr_high = validation[direction][
            "correlation"
        ]
        fit_rows.append(
            {
                "file_name": file_name,
                "subject": file_data["subject"],
                "task": file_data["task"],
                "direction": direction,
                "n_trials": int(np.count_nonzero(direction_mask)),
                "robust_curve_r2": metrics["r2"],
                "robust_curve_rmse": metrics["rmse"],
                "robust_curve_correlation": metrics["correlation"],
                "split_half_r2_median": split_median,
                "split_half_r2_ci_low": split_low,
                "split_half_r2_ci_high": split_high,
                "split_half_correlation_median": split_corr_median,
                "split_half_correlation_ci_low": split_corr_low,
                "split_half_correlation_ci_high": split_corr_high,
            }
        )

        for stage_index, stage in enumerate(STAGE_NAMES):
            for channel_index, channel in enumerate(EEG_LABELS):
                gain_rows.append(
                    {
                        "file_name": file_name,
                        "subject": file_data["subject"],
                        "task": file_data["task"],
                        "direction": direction,
                        "stage": stage,
                        "channel": channel,
                        "effective_gain": aggregate[direction][stage_index, channel_index],
                        "bootstrap_ci_low": condition_low[direction][stage_index, channel_index],
                        "bootstrap_ci_high": condition_high[direction][stage_index, channel_index],
                    }
                )

    left_indices = labels == "L"
    right_indices = labels == "R"
    for stage_index, stage in enumerate(STAGE_NAMES):
        for channel_index, channel in enumerate(EEG_LABELS):
            left_values = stage_coefficients[left_indices, stage_index, channel_index]
            right_values = stage_coefficients[right_indices, stage_index, channel_index]
            difference_rows.append(
                {
                    "file_name": file_name,
                    "subject": file_data["subject"],
                    "task": file_data["task"],
                    "stage": stage,
                    "channel": channel,
                    "gain_difference_R_minus_L": (
                        aggregate["R"][stage_index, channel_index]
                        - aggregate["L"][stage_index, channel_index]
                    ),
                    "bootstrap_ci_low": difference_interval[0, stage_index, channel_index],
                    "bootstrap_ci_high": difference_interval[1, stage_index, channel_index],
                    "hedges_g_R_minus_L": hedges_g(left_values, right_values),
                }
            )

    flattened = stage_coefficients.reshape(stage_coefficients.shape[0], -1)
    statistic, p_value = permutation_direction_test(
        flattened,
        labels,
        np.random.default_rng(deterministic_seed(file_name, "permutation")),
    )
    test_row = {
        "file_name": file_name,
        "subject": file_data["subject"],
        "task": file_data["task"],
        "n_left": int(np.count_nonzero(left_indices)),
        "n_right": int(np.count_nonzero(right_indices)),
        "joint_12d_welch_statistic": statistic,
        "permutation_p": p_value,
    }

    feature_rows: list[dict[str, object]] = []
    for trial_index in range(epochs.shape[0]):
        row: dict[str, object] = {
            "file_name": file_name,
            "subject": file_data["subject"],
            "task": file_data["task"],
            "trial_id": int(trial_ids[trial_index]),
            "cue_direction": labels[trial_index],
        }
        for stage_index, stage in enumerate(STAGE_NAMES):
            for channel_index, channel in enumerate(EEG_LABELS):
                row[f"mechanism__{stage}__{channel}__effective_gain"] = (
                    stage_coefficients[trial_index, stage_index, channel_index]
                )
        feature_rows.append(row)

    return {
        "gain_rows": gain_rows,
        "difference_rows": difference_rows,
        "fit_rows": fit_rows,
        "test_row": test_row,
        "feature_rows": feature_rows,
        "observed_curves": observed_curves,
        "fitted_curves": fitted_curves,
    }


def build_stable_mechanism_features(differences: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task in ("Task-1", "Task-2"):
        task_table = differences[differences["task"] == task]
        for (stage, channel), group in task_table.groupby(["stage", "channel"]):
            if set(group["subject"]) != {"A", "B"}:
                continue
            by_subject = group.set_index("subject")
            g_a = float(by_subject.loc["A", "hedges_g_R_minus_L"])
            g_b = float(by_subject.loc["B", "hedges_g_R_minus_L"])
            same_sign = bool(g_a * g_b > 0.0)
            rows.append(
                {
                    "task": task,
                    "stage": stage,
                    "channel": channel,
                    "hedges_g_subject_A": g_a,
                    "hedges_g_subject_B": g_b,
                    "same_sign_A_B": same_sign,
                    "conservative_abs_g": min(abs(g_a), abs(g_b)) if same_sign else 0.0,
                }
            )
    stable = pd.DataFrame(rows)
    return stable.sort_values(
        ["task", "same_sign_A_B", "conservative_abs_g"],
        ascending=[True, False, False],
    ).reset_index(drop=True)


def plot_mechanism_schematic(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 4.2)
    ax.axis("off")

    boxes = [
        (0.5, 1.15, 2.3, 1.65, "Visual cue", "Left / right triangle", "#FEE8C8"),
        (3.6, 1.15, 2.3, 1.65, "LGN relay", "~50 ms", "#E5D8F2"),
        (6.7, 0.75, 3.3, 2.45, "Cortical population", "Feedforward ~130 ms\nRecurrent P300 ~310 ms\nLate context ~670 ms", "#D9F0D3"),
        (10.9, 1.15, 2.5, 1.65, "Scalp observation", "F3   Fz   F4", "#DEEBF7"),
    ]
    for x, y, width, height, title, subtitle, color in boxes:
        patch = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.6, edgecolor="#333333", facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + width / 2, y + height * 0.64, title, ha="center", va="center", fontsize=14, weight="bold")
        ax.text(x + width / 2, y + height * 0.31, subtitle, ha="center", va="center", fontsize=11, linespacing=1.35)

    for start, end in [((2.8, 1.98), (3.6, 1.98)), ((5.9, 1.98), (6.7, 1.98)), ((10.0, 1.98), (10.9, 1.98))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18, linewidth=2.0, color="#444444"))

    ax.text(7.2, 3.65, "Direction-dependent feature gain", ha="center", fontsize=12, color="#7A0177")
    ax.add_patch(FancyArrowPatch((7.2, 3.52), (7.65, 3.13), arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#7A0177"))
    ax.text(11.95, 0.55, "Signed effective lead field", ha="center", fontsize=11, color="#08519C")
    ax.text(7.0, 0.18, "Constrained phenomenological forward model — not unique source localization", ha="center", fontsize=11, color="#555555")
    fig.suptitle("LGN–cortex–scalp EEG formation model", fontsize=18, y=0.98)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_mechanism_basis(
    fit_time: np.ndarray,
    states: np.ndarray,
    peak_latency: dict[str, float],
    path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for index, (stage, short, color) in enumerate(zip(STAGE_NAMES, STAGE_SHORT, STAGE_COLORS)):
        ax.plot(fit_time, states[:, index], color=color, linewidth=2.5, label=f"{short}: peak {peak_latency[stage] * 1000:.0f} ms")
        ax.axvline(peak_latency[stage], color=color, linewidth=0.8, alpha=0.35)
    ax.axvspan(0.25, 0.50, color="#FEE391", alpha=0.22, label="P300 window")
    ax.axvspan(0.55, 0.90, color="#C7E9C0", alpha=0.18, label="Late window")
    ax.set(xlabel="Time from cue onset (s)", ylabel="Normalized latent response", title="Shared causal temporal basis")
    ax.set_xlim(FIT_WINDOW_S)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_mechanism_fits(
    analyses: dict[str, dict[str, object]],
    fit_time: np.ndarray,
    fit_summary: pd.DataFrame,
    path: Path,
) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(17, 17), sharex=True)
    for row_index, file_name in enumerate(EXPECTED_FILES):
        analysis = analyses[file_name]
        observed = analysis["observed_curves"]
        fitted = analysis["fitted_curves"]
        for channel_index, channel in enumerate(EEG_LABELS):
            ax = axes[row_index, channel_index]
            for direction in DIRECTIONS:
                color = DIRECTION_COLORS[direction]
                ax.plot(fit_time, observed[direction][:, channel_index], color=color, linewidth=1.7, label=f"{direction} observed")
                ax.plot(fit_time, fitted[direction][:, channel_index], color=color, linewidth=2.0, linestyle="--", label=f"{direction} model")
            ax.axhline(0.0, color="#777777", linewidth=0.8)
            ax.axvspan(0.25, 0.50, color="#FEE391", alpha=0.18)
            ax.axvspan(0.55, 0.90, color="#C7E9C0", alpha=0.15)
            ax.grid(alpha=0.2)
            if row_index == 0:
                ax.set_title(channel, fontsize=13)
            if channel_index == 0:
                ax.set_ylabel(f"{FILE_DISPLAY[file_name]}\nAmplitude")
            if row_index == 3:
                ax.set_xlabel("Time from cue onset (s)")
            local = fit_summary[(fit_summary["file_name"] == file_name)]
            r2_text = ", ".join(
                f"{d} R²={float(local[local['direction'] == d]['robust_curve_r2'].iloc[0]):.2f}"
                for d in DIRECTIONS
            )
            ax.text(0.02, 0.96, r2_text, transform=ax.transAxes, va="top", fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.72, edgecolor="none"))
            if row_index == 0 and channel_index == 0:
                ax.legend(ncol=2, fontsize=8, loc="lower left")
    fig.suptitle("Observed robust ERP and constrained mechanism-model fits", fontsize=18, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_gain_heatmap(differences: pd.DataFrame, path: Path) -> None:
    columns = [(stage, channel) for stage in STAGE_NAMES for channel in EEG_LABELS]
    matrix = np.empty((len(EXPECTED_FILES), len(columns)), dtype=np.float64)
    for row_index, file_name in enumerate(EXPECTED_FILES):
        local = differences[differences["file_name"] == file_name].set_index(["stage", "channel"])
        for column_index, key in enumerate(columns):
            matrix[row_index, column_index] = float(local.loc[key, "hedges_g_R_minus_L"])

    limit = max(0.5, float(np.nanmax(np.abs(matrix))))
    fig, ax = plt.subplots(figsize=(18, 6.2))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-limit, vmax=limit, aspect="auto")
    labels = [f"{short}\n{channel}" for short in STAGE_SHORT for channel in EEG_LABELS]
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(EXPECTED_FILES)), labels=[FILE_DISPLAY[name] for name in EXPECTED_FILES])
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, f"{matrix[row, column]:+.2f}", ha="center", va="center", fontsize=9)
    for boundary in (2.5, 5.5, 8.5):
        ax.axvline(boundary, color="white", linewidth=3)
    ax.set_title("Mechanism-feature direction effects (Hedges g, R - L)", fontsize=16)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label("Hedges g")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_validation(fit_summary: pd.DataFrame, tests: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(EXPECTED_FILES))
    width = 0.34
    for offset, direction in zip((-width / 2, width / 2), DIRECTIONS):
        local = fit_summary[fit_summary["direction"] == direction].set_index("file_name").loc[list(EXPECTED_FILES)]
        values = local["split_half_correlation_median"].to_numpy(dtype=float)
        lows = local["split_half_correlation_ci_low"].to_numpy(dtype=float)
        highs = local["split_half_correlation_ci_high"].to_numpy(dtype=float)
        axes[0].bar(x + offset, values, width, color=DIRECTION_COLORS[direction], alpha=0.82, label=direction)
        axes[0].errorbar(x + offset, values, yerr=np.vstack([values - lows, highs - values]), fmt="none", ecolor="#222222", capsize=3, linewidth=1.2)
    axes[0].axhline(0.0, color="#555555", linestyle="--", linewidth=1.2)
    axes[0].set_xticks(x, [FILE_DISPLAY[name] for name in EXPECTED_FILES], rotation=20, ha="right")
    axes[0].set_ylim(-0.5, 1.02)
    axes[0].set_ylabel("Held-out split-half correlation")
    axes[0].set_title("Predictive reconstruction")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.22)

    ordered = tests.set_index("file_name").loc[list(EXPECTED_FILES)]
    p_values = ordered["permutation_p"].to_numpy(dtype=float)
    q_values = ordered["fdr_q"].to_numpy(dtype=float)
    y = -np.log10(np.maximum(q_values, 1.0 / (N_PERMUTATIONS + 1)))
    bars = axes[1].bar(x, y, color="#756BB1", alpha=0.85)
    axes[1].axhline(-np.log10(0.05), color="#CB181D", linestyle="--", linewidth=1.4, label="FDR 0.05")
    axes[1].set_xticks(x, [FILE_DISPLAY[name] for name in EXPECTED_FILES], rotation=20, ha="right")
    axes[1].set_ylabel("-log10(BH-FDR q)")
    axes[1].set_title("Joint 12-D direction effect")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.22)
    for bar, p_value, q_value in zip(bars, p_values, q_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, f"p={p_value:.3f}\nq={q_value:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Mechanism-model validation", fontsize=17)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def read_external_classification_summary(project_root: Path) -> dict[str, float] | None:
    path = project_root / "results" / "06_q2_feature_model" / "bidirectional_summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path, encoding="utf-8-sig")
    required = {"scope", "feature_group", "balanced_accuracy", "balanced_accuracy_ci_low", "balanced_accuracy_ci_high", "roc_auc", "permutation_p"}
    if not required.issubset(table.columns):
        return None
    row = table[(table["scope"] == "Pooled") & (table["feature_group"] == "full_multichannel")]
    if len(row) != 1:
        return None
    return {key: float(row.iloc[0][key]) for key in required if key not in {"scope", "feature_group"}}


def build_report(
    feature_table: pd.DataFrame,
    gains: pd.DataFrame,
    differences: pd.DataFrame,
    tests: pd.DataFrame,
    fit_summary: pd.DataFrame,
    stable: pd.DataFrame,
    peak_latency: dict[str, float],
    condition_number: float,
    external_result: dict[str, float] | None,
) -> str:
    lines = [
        "C题问题2第二阶段：LGN—皮层—头皮受约束机制模型报告",
        "=" * 72,
        "",
        "一、模型定位",
        "本模型是受约束的现象学生理前向模型，不是LGN/V1/额叶源定位。",
        "三个额区电极只能识别潜变量到头皮的有效增益，不能把源强度与容积传导唯一分离。",
        "所有方向、任务、被试共用同一组时间动力学，左右标签不参与单试验特征拟合。",
        "",
        "二、模型方程",
        "tau_g dg/dt = -g + u(t-delta_g)",
        "x_c = H_c(g),  x_p = H_p(x_c),  x_l = H_l(x_p)",
        "y_(e,d)(t) = sum_k a_(e,d,k)x_k(t) + b_(e,d) + c_(e,d)t + epsilon",
        "a_(e,d,k)表示源活动与头皮容积传导共同形成的带符号有效增益。",
        "",
        "固定潜变量峰潜伏期：",
    ]
    for stage, short in zip(STAGE_NAMES, STAGE_SHORT):
        lines.append(f"  {short}: {peak_latency[stage] * 1000:.1f} ms")
    lines.extend(
        [
            f"设计矩阵条件数：{condition_number:.3f}",
            f"岭约束强度：{RIDGE_ALPHA:g}",
            "",
            "三、波形重建与拆半预测",
        ]
    )

    for file_name in EXPECTED_FILES:
        lines.append(f"[{FILE_DISPLAY[file_name]}]")
        local = fit_summary[fit_summary["file_name"] == file_name]
        for direction in DIRECTIONS:
            row = local[local["direction"] == direction].iloc[0]
            lines.append(
                f"  {direction}: n={int(row['n_trials'])}, robust曲线R2={row['robust_curve_r2']:.3f}, "
                f"相关={row['robust_curve_correlation']:.3f}, RMSE={row['robust_curve_rmse']:.3f}, "
                f"拆半R2中位数={row['split_half_r2_median']:.3f} "
                f"({row['split_half_r2_ci_low']:.3f}—{row['split_half_r2_ci_high']:.3f}), "
                f"拆半相关中位数={row['split_half_correlation_median']:.3f} "
                f"({row['split_half_correlation_ci_low']:.3f}—{row['split_half_correlation_ci_high']:.3f})"
            )

    lines.extend(["", "四、12维机制特征的左右整体差异检验"])
    for row in tests.itertuples(index=False):
        lines.append(
            f"[{FILE_DISPLAY[row.file_name]}] nL={row.n_left}, nR={row.n_right}, "
            f"置换p={row.permutation_p:.4f}, BH-FDR q={row.fdr_q:.4f}"
        )
    significant_count = int(np.count_nonzero(tests["fdr_q"].to_numpy(dtype=float) < 0.05))
    if significant_count:
        lines.append(
            f"文件内共有{significant_count}/4组在12维机制特征上达到FDR<0.05；"
            "这表示该记录内存在方向相关增益差异，不等同于跨被试可解码。"
        )
    else:
        lines.append(
            "四组记录均未在12维机制特征上达到FDR<0.05；应把增益差异作为候选机制描述。"
        )

    lines.extend(["", "五、跨被试方向一致的候选机制特征（描述性）"])
    for task in ("Task-1", "Task-2"):
        lines.append(f"{task}：")
        local = stable[(stable["task"] == task) & stable["same_sign_A_B"]].head(5)
        if local.empty:
            lines.append("  无A、B同号候选。")
        else:
            for row in local.itertuples(index=False):
                lines.append(
                    f"  {row.stage}/{row.channel}: g_A={row.hedges_g_subject_A:+.3f}, "
                    f"g_B={row.hedges_g_subject_B:+.3f}, "
                    f"保守|g|={row.conservative_abs_g:.3f}"
                )

    lines.extend(["", "六、与严格跨被试分类的联合判读"])
    if external_result is None:
        lines.append("未找到06的bidirectional_summary.csv；机制报告不包含跨被试分类结论。")
    else:
        lines.append(
            "06主模型（Pooled | full_multichannel）："
            f"BAC={external_result['balanced_accuracy']:.4f}, "
            f"95% CI {external_result['balanced_accuracy_ci_low']:.4f}—"
            f"{external_result['balanced_accuracy_ci_high']:.4f}, "
            f"AUC={external_result['roc_auc']:.4f}, "
            f"置换p={external_result['permutation_p']:.4f}。"
        )
        if external_result["permutation_p"] >= 0.05:
            lines.append(
                "因此，机制增益可用于解释和组织左右响应差异，但当前数据未验证其为可靠的跨被试解码标志物。"
            )
        else:
            lines.append("严格跨被试分类达到置换显著，可把机制增益作为已得到外部初检的候选表征。")

    feature_columns = [column for column in feature_table.columns if column.startswith("mechanism__")]
    all_finite = bool(np.isfinite(feature_table[feature_columns].to_numpy(dtype=float)).all())
    forbidden = any("decon" in column.lower() for column in feature_columns)
    minimum_class = int(feature_table.groupby(["file_name", "cue_direction"]).size().min())
    expected_gain_rows = len(EXPECTED_FILES) * len(DIRECTIONS) * len(STAGE_NAMES) * len(EEG_LABELS)
    expected_difference_rows = len(EXPECTED_FILES) * len(STAGE_NAMES) * len(EEG_LABELS)
    ordered_peaks = [peak_latency[stage] for stage in STAGE_NAMES]
    peak_ranges_ok = (
        0.035 <= ordered_peaks[0] <= 0.080
        and 0.090 <= ordered_peaks[1] <= 0.200
        and 0.250 <= ordered_peaks[2] <= 0.450
        and 0.550 <= ordered_peaks[3] <= 0.850
    )

    lines.extend(
        [
            "",
            "自动验收",
            "-" * 72,
            f"{pass_fail(len(feature_table) == 356)}  应读取问题一的356个清洁试验",
            f"{pass_fail(len(feature_columns) == 12)}  每个试验应得到4阶段×3电极的12维机制特征",
            f"{pass_fail(all_finite)}  全部机制特征应为有限数",
            f"{pass_fail(not forbidden)}  机制特征不得包含Decon通道",
            f"{pass_fail(set(feature_table['cue_direction']) == {'L', 'R'})}  方向标签应仅包含L和R",
            f"{pass_fail(minimum_class >= 40)}  每个文件每个方向至少应有40个清洁试验",
            f"{pass_fail(peak_ranges_ok)}  四阶段峰值应按LGN、皮层、P300、晚期依次出现",
            f"{pass_fail(condition_number < 50.0)}  设计矩阵条件数应小于50",
            f"{pass_fail(len(gains) == expected_gain_rows)}  阶段增益表行数应完整",
            f"{pass_fail(len(differences) == expected_difference_rows)}  R-L增益差表行数应完整",
            f"{pass_fail(len(tests) == 4)}  应完成四组文件内12维置换检验",
            f"{pass_fail(len(fit_summary) == 8)}  应完成四文件×两方向的模型拟合与拆半验证",
            "",
            "边界说明",
            "1. 该模型把LGN—皮层通路简化为共享的稳定级联，是机制约束下的可检验近似。",
            "2. 有效增益同时包含神经群体强度、偶极方向和头皮容积传导，不能做唯一源定位。",
            "3. 波形拟合优度回答模型能否重建ERP，不回答左右标签能否在新被试上预测。",
            "4. 跨被试可区分性必须以06的严格A↔B外部验证为准，不能用本阶段拟合优度替代。",
            "5. 只有两名被试，所有机制差异仍需更多被试和更多电极进一步验证。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    epoch_dir = project_root / "data" / "processed" / "erp_epochs"
    processed_dir = project_root / "data" / "processed"
    output_dir = project_root / "results" / "07_q2_mechanism_model"
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

    print("正在读取问题一的清洁ERP试验……")
    file_data: dict[str, dict[str, object]] = {}
    reference_time: np.ndarray | None = None
    for file_name in EXPECTED_FILES:
        data = load_epoch_file(epoch_dir / FILE_TO_EPOCH[file_name], file_name)
        if reference_time is None:
            reference_time = np.asarray(data["time_s"], dtype=np.float64)
        elif not np.allclose(reference_time, np.asarray(data["time_s"], dtype=np.float64)):
            raise ValueError("四个ERP文件的时间轴不一致。")
        file_data[file_name] = data
        labels = np.asarray(data["directions"]).astype(str)
        print(
            f"  {file_name}: {len(labels)}个清洁试验 "
            f"(L={np.count_nonzero(labels == 'L')}, R={np.count_nonzero(labels == 'R')})"
        )

    if reference_time is None:
        raise RuntimeError("没有读取到ERP时间轴。")
    fit_mask, states, design, peak_latency, condition_number = build_mechanistic_basis(
        reference_time, EXPECTED_SAMPLE_RATE
    )
    fit_time = reference_time[fit_mask]
    projection = ridge_projection(design)

    print("正在拟合共享动力学与方向相关有效增益……")
    analyses: dict[str, dict[str, object]] = {}
    feature_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    difference_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    for file_name in EXPECTED_FILES:
        analysis = analyse_file(file_data[file_name], fit_mask, design, projection)
        analyses[file_name] = analysis
        feature_rows.extend(analysis["feature_rows"])
        gain_rows.extend(analysis["gain_rows"])
        difference_rows.extend(analysis["difference_rows"])
        fit_rows.extend(analysis["fit_rows"])
        test_rows.append(analysis["test_row"])
        print(f"  {file_name}: 完成")

    feature_table = pd.DataFrame(feature_rows)
    gains = pd.DataFrame(gain_rows)
    differences = pd.DataFrame(difference_rows)
    fit_summary = pd.DataFrame(fit_rows)
    tests = pd.DataFrame(test_rows)
    tests["fdr_q"] = fdr_bh(tests["permutation_p"].to_numpy(dtype=float))
    tests["significant_fdr_0_05"] = tests["fdr_q"] < 0.05
    stable = build_stable_mechanism_features(differences)

    feature_path = processed_dir / "q2_mechanism_trial_features.csv"
    gains_path = output_dir / "mechanism_stage_gains.csv"
    difference_path = output_dir / "direction_gain_differences.csv"
    tests_path = output_dir / "direction_model_tests.csv"
    fit_path = output_dir / "model_fit_summary.csv"
    stable_path = output_dir / "stable_mechanism_features.csv"
    report_path = output_dir / "q2_mechanism_report.txt"

    feature_table.to_csv(feature_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    gains.to_csv(gains_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    differences.to_csv(difference_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    tests.to_csv(tests_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    fit_summary.to_csv(fit_path, index=False, encoding="utf-8-sig", float_format="%.9g")
    stable.to_csv(stable_path, index=False, encoding="utf-8-sig", float_format="%.9g")

    plot_mechanism_schematic(output_dir / "mechanism_schematic.png")
    plot_mechanism_basis(fit_time, states, peak_latency, output_dir / "mechanism_basis.png")
    plot_mechanism_fits(analyses, fit_time, fit_summary, output_dir / "mechanism_fit_curves.png")
    plot_gain_heatmap(differences, output_dir / "mechanism_gain_heatmap.png")
    plot_validation(fit_summary, tests, output_dir / "model_validation.png")

    external_result = read_external_classification_summary(project_root)
    report = build_report(
        feature_table,
        gains,
        differences,
        tests,
        fit_summary,
        stable,
        peak_latency,
        condition_number,
        external_result,
    )
    report_path.write_text(report, encoding="utf-8")

    print("\n问题2第二阶段完成。")
    print(f"清洁试验数：{len(feature_table)}")
    print(f"机制特征数：{len([c for c in feature_table.columns if c.startswith('mechanism__')])}")
    print(f"逐试验机制特征：{feature_path}")
    print(f"模型拟合汇总：{fit_path}")
    print(f"方向整体检验：{tests_path}")
    print(f"检查报告：{report_path}")
    print("\n请先打开 q2_mechanism_report.txt，确认自动验收项目均显示 PASS。")


if __name__ == "__main__":
    main()
