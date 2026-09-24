"""C题问题1第四阶段：稳健ERP曲线拟合与左右视觉提示差异检验。

把本文件放在 C:\\C\\src\\04_erp_curves.py 后直接运行。

前置条件：已经运行 03_preprocess_erp.py，并生成
    C:\\C\\data\\processed\\erp_epochs\\*_erp_epochs.npz

分析方法：
1. 每个文件、方向和通道使用Huber M估计合并试验；
2. 使用约120 ms窗口的局部三次Savitzky-Golay回归拟合ERP曲线；
3. 通过500次试验级bootstrap给出逐时点95%置信区间；
4. 同时量化题面规定的250-500 ms早期窗口与数据提示的550-900 ms晚期窗口；
5. 使用5000次标签置换检验左右方向差异，并以Benjamini-Hochberg方法控制FDR；
6. 对所有试验统一构造固定F3/F4侧化F4-F3，再检验左右方向差异。

注意：逐时点置信带为点态区间，不等同于全时间轴同时置信带；固定侧化指标
不使用真实方向标签改变符号，避免后续判别中的标签泄漏，也不直接等同于皮层源定位。

输出目录：C:\\C\\results\\04_erp_analysis
    erp_curve_points.csv
    erp_features.csv
    direction_tests.csv
    laterality_curve_points.csv
    laterality_features.csv
    subject_consistency.csv
    erp_analysis_report.txt
    erp_fitted_curves.png
    direction_difference.png
    laterality_curves.png
    effect_size_heatmap.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


FILE_TO_EPOCH = {
    "VisualCogA_Task-1.mat": "A_Task-1_erp_epochs.npz",
    "VisualCogA_Task-2.mat": "A_Task-2_erp_epochs.npz",
    "VisualCogB_Task-1.mat": "B_Task-1_erp_epochs.npz",
    "VisualCogB_Task-2.mat": "B_Task-2_erp_epochs.npz",
}
EXPECTED_FILES = tuple(FILE_TO_EPOCH)
EEG_LABELS = ("Fz", "F3", "F4")
SIDE_COLORS = {"L": "#3366CC", "R": "#DC3912"}
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

EARLY_WINDOW = (0.25, 0.50)
LATE_WINDOW = (0.55, 0.90)
FULL_RESPONSE_WINDOW = (0.20, 0.90)
WINDOWS = {"early_250_500ms": EARLY_WINDOW, "late_550_900ms": LATE_WINDOW}

HUBER_C = 1.345
HUBER_MAX_ITER = 30
HUBER_TOLERANCE = 1e-7
SMOOTH_WINDOW_S = 0.12
SMOOTH_POLYORDER = 3
N_BOOTSTRAP = 500
N_PERMUTATIONS = 5000
RANDOM_SEED = 20260923


def safe_stem(file_name: str) -> str:
    return file_name.replace("VisualCog", "").replace(".mat", "")


def window_mask(time_s: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return (time_s >= limits[0]) & (time_s <= limits[1])


def huber_location(samples: np.ndarray) -> np.ndarray:
    """沿试验维计算Huber M位置估计，支持(n_trial, n_channel, n_time)。"""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] < 2:
        raise ValueError("Huber估计至少需要两个试验。")

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
        weights[outside] = HUBER_C / absolute[outside]
        denominator = np.sum(weights, axis=0)
        updated = np.sum(weights * values, axis=0) / np.maximum(denominator, 1e-12)
        if np.max(np.abs(updated - location)) < HUBER_TOLERANCE:
            location = updated
            break
        location = updated
    return location


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


def fit_curve(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    robust_location = huber_location(samples)
    window = smoothing_window_samples(sample_rate, robust_location.shape[-1])
    return savgol_filter(
        robust_location,
        window_length=window,
        polyorder=SMOOTH_POLYORDER,
        axis=-1,
        mode="interp",
    )


def bootstrap_fitted_curves(
    samples: np.ndarray,
    sample_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n_trials = samples.shape[0]
    output = np.empty(
        (N_BOOTSTRAP, samples.shape[1], samples.shape[2]),
        dtype=np.float32,
    )
    for bootstrap_index in range(N_BOOTSTRAP):
        selection = rng.integers(0, n_trials, size=n_trials)
        output[bootstrap_index] = fit_curve(samples[selection], sample_rate).astype(np.float32)
    return output


def permutation_test_independent(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """返回R-L均值差和双侧置换p值。"""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    observed = float(np.mean(right) - np.mean(left))
    combined = np.concatenate([left, right])
    n_left = left.size
    exceed = 0
    for _ in range(N_PERMUTATIONS):
        permuted = rng.permutation(combined)
        difference = float(np.mean(permuted[n_left:]) - np.mean(permuted[:n_left]))
        exceed += int(abs(difference) >= abs(observed))
    p_value = (exceed + 1) / (N_PERMUTATIONS + 1)
    return observed, float(p_value)


def hedges_g(left: np.ndarray, right: np.ndarray) -> float:
    """标准化效应量，方向定义为R-L。"""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n_left = left.size
    n_right = right.size
    degrees = n_left + n_right - 2
    if degrees <= 0:
        return float("nan")
    pooled_variance = (
        (n_left - 1) * np.var(left, ddof=1) + (n_right - 1) * np.var(right, ddof=1)
    ) / degrees
    if pooled_variance <= 1e-20:
        return 0.0
    cohens_d = (np.mean(right) - np.mean(left)) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n_left + n_right) - 9.0)
    return float(correction * cohens_d)


def bootstrap_difference_interval(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int = 2000,
) -> tuple[float, float]:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    differences = np.empty(n_bootstrap, dtype=np.float64)
    for index in range(n_bootstrap):
        left_sample = left[rng.integers(0, left.size, size=left.size)]
        right_sample = right[rng.integers(0, right.size, size=right.size)]
        differences[index] = np.mean(right_sample) - np.mean(left_sample)
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


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


def curve_features(curve: np.ndarray, time_s: np.ndarray) -> dict[str, float]:
    result: dict[str, float] = {}
    for prefix, limits in (
        ("early", EARLY_WINDOW),
        ("late", LATE_WINDOW),
        ("full", FULL_RESPONSE_WINDOW),
    ):
        mask = window_mask(time_s, limits)
        local_curve = curve[mask]
        local_time = time_s[mask]
        peak_index = int(np.argmax(local_curve))
        result[f"{prefix}_mean"] = float(np.mean(local_curve))
        result[f"{prefix}_positive_peak"] = float(local_curve[peak_index])
        result[f"{prefix}_peak_latency_s"] = float(local_time[peak_index])
        result[f"{prefix}_area"] = float(np.trapezoid(local_curve, local_time))
    return result


def load_epochs(epoch_path: Path) -> dict:
    with np.load(epoch_path, allow_pickle=False) as archive:
        epochs = np.asarray(archive["epochs_accepted"], dtype=np.float64)
        directions = np.asarray(archive["cue_direction_accepted"]).astype(str)
        trial_ids = np.asarray(archive["accepted_trial_id"], dtype=np.int32)
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        channels = tuple(np.asarray(archive["channel_labels"]).astype(str).tolist())
        sample_rate = float(np.asarray(archive["sample_rate_hz"]).squeeze())

    if epochs.ndim != 3 or epochs.shape[1] != 3:
        raise ValueError(f"{epoch_path.name} 的epochs_accepted形状异常：{epochs.shape}")
    if epochs.shape[0] != directions.size or epochs.shape[0] != trial_ids.size:
        raise ValueError(f"{epoch_path.name} 的试验索引与脑电片段数量不一致。")
    if channels != EEG_LABELS:
        raise ValueError(f"{epoch_path.name} 的通道顺序为{channels}，预期为{EEG_LABELS}。")
    if not np.isfinite(epochs).all():
        raise ValueError(f"{epoch_path.name} 的保留片段仍含NaN或Inf。")
    for side in ("L", "R"):
        if np.count_nonzero(directions == side) < 30:
            raise ValueError(f"{epoch_path.name} 的{side}方向有效试验不足30个。")
    return {
        "epochs": epochs,
        "directions": directions,
        "trial_ids": trial_ids,
        "time_s": time_s,
        "sample_rate": sample_rate,
    }


def save_fitted_curves_plot(curve_cache: dict, path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
    for row_index, file_name in enumerate(EXPECTED_FILES):
        for channel_index, channel in enumerate(EEG_LABELS):
            axis = axes[row_index, channel_index]
            for side in ("L", "R"):
                item = curve_cache[(file_name, side)]
                time_s = item["time_s"]
                curve = item["curve"][channel_index]
                low = item["ci_low"][channel_index]
                high = item["ci_high"][channel_index]
                color = SIDE_COLORS[side]
                axis.fill_between(time_s, low, high, color=color, alpha=0.13, linewidth=0)
                axis.plot(
                    time_s,
                    curve,
                    color=color,
                    linewidth=1.5,
                    label=f"{side} (n={item['n_trials']})",
                )
            axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.8)
            axis.axhline(0.0, color="#999999", linewidth=0.6)
            axis.axvspan(*EARLY_WINDOW, color="#FFC107", alpha=0.13)
            axis.axvspan(*LATE_WINDOW, color="#26A69A", alpha=0.09)
            axis.set_xlim(-0.2, 1.0)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(channel)
            if channel_index == 0:
                axis.set_ylabel(f"{safe_stem(file_name)}\nAmplitude")
            if row_index == len(EXPECTED_FILES) - 1:
                axis.set_xlabel("Time from cue onset (s)")
            axis.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Robust ERP fits with pointwise 95% bootstrap CI\n"
        "yellow: 250-500 ms; green: 550-900 ms",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_difference_plot(curve_cache: dict, path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
    for row_index, file_name in enumerate(EXPECTED_FILES):
        left = curve_cache[(file_name, "L")]
        right = curve_cache[(file_name, "R")]
        difference = right["curve"] - left["curve"]
        bootstrap_difference = right["bootstrap"] - left["bootstrap"]
        low, high = np.percentile(bootstrap_difference, [2.5, 97.5], axis=0)
        for channel_index, channel in enumerate(EEG_LABELS):
            axis = axes[row_index, channel_index]
            axis.fill_between(
                left["time_s"],
                low[channel_index],
                high[channel_index],
                color="#7B1FA2",
                alpha=0.16,
                linewidth=0,
            )
            axis.plot(
                left["time_s"],
                difference[channel_index],
                color="#7B1FA2",
                linewidth=1.4,
            )
            axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.8)
            axis.axhline(0.0, color="#555555", linewidth=0.8)
            axis.axvspan(*EARLY_WINDOW, color="#FFC107", alpha=0.13)
            axis.axvspan(*LATE_WINDOW, color="#26A69A", alpha=0.09)
            axis.set_xlim(-0.2, 1.0)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(channel)
            if channel_index == 0:
                axis.set_ylabel(f"{safe_stem(file_name)}\nR minus L")
            if row_index == len(EXPECTED_FILES) - 1:
                axis.set_xlabel("Time from cue onset (s)")
    fig.suptitle("Direction difference curves (R - L) with pointwise 95% bootstrap CI", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_laterality_plot(laterality_cache: dict, path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, file_name in zip(axes.ravel(), EXPECTED_FILES):
        for side in ("L", "R"):
            item = laterality_cache[(file_name, side)]
            color = SIDE_COLORS[side]
            axis.fill_between(
                item["time_s"],
                item["ci_low"],
                item["ci_high"],
                color=color,
                alpha=0.13,
                linewidth=0,
            )
            axis.plot(
                item["time_s"],
                item["curve"],
                color=color,
                linewidth=1.5,
                label=f"{side} (n={item['n_trials']})",
            )
        axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="#555555", linewidth=0.8)
        axis.axvspan(*EARLY_WINDOW, color="#FFC107", alpha=0.13)
        axis.axvspan(*LATE_WINDOW, color="#26A69A", alpha=0.09)
        axis.set_xlim(-0.2, 1.0)
        axis.set_title(file_name.replace(".mat", ""))
        axis.set_xlabel("Time from cue onset (s)")
        axis.set_ylabel("Fixed F4 - F3 asymmetry")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Fixed electrode asymmetry for both directions: F4 - F3", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_effect_heatmap(tests: pd.DataFrame, path: Path) -> None:
    row_labels: list[str] = []
    matrix_rows: list[list[float]] = []
    significance_rows: list[list[bool]] = []
    for file_name in EXPECTED_FILES:
        for window_name in WINDOWS:
            subset = tests[
                (tests["file_name"] == file_name) & (tests["window"] == window_name)
            ].set_index("channel")
            row_labels.append(
                f"{safe_stem(file_name)} | {'250-500 ms' if window_name.startswith('early') else '550-900 ms'}"
            )
            matrix_rows.append([float(subset.loc[channel, "hedges_g_R_minus_L"]) for channel in EEG_LABELS])
            significance_rows.append(
                [bool(subset.loc[channel, "significant_fdr_0_05"]) for channel in EEG_LABELS]
            )

    matrix = np.asarray(matrix_rows, dtype=np.float64)
    significance = np.asarray(significance_rows, dtype=bool)
    color_limit = max(0.5, float(np.nanmax(np.abs(matrix))))

    fig, axis = plt.subplots(figsize=(8.5, 7.0))
    image = axis.imshow(
        matrix,
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
        aspect="auto",
    )
    axis.set_xticks(np.arange(len(EEG_LABELS)))
    axis.set_xticklabels(EEG_LABELS)
    axis.set_yticks(np.arange(len(row_labels)))
    axis.set_yticklabels(row_labels)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            star = "*" if significance[row, column] else ""
            axis.text(
                column,
                row,
                f"{matrix[row, column]:+.2f}{star}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, shrink=0.85)
    colorbar.set_label("Hedges g (R - L)")
    axis.set_title("Direction effect sizes (*: BH-FDR < 0.05)")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_subject_consistency(direction_tests: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for task in ("Task-1", "Task-2"):
        for window_name in WINDOWS:
            for channel in EEG_LABELS:
                subset = direction_tests[
                    (direction_tests["task"] == task)
                    & (direction_tests["window"] == window_name)
                    & (direction_tests["channel"] == channel)
                ].set_index("subject")
                effect_a = float(subset.loc["A", "hedges_g_R_minus_L"])
                effect_b = float(subset.loc["B", "hedges_g_R_minus_L"])
                rows.append(
                    {
                        "task": task,
                        "window": window_name,
                        "channel": channel,
                        "hedges_g_A": effect_a,
                        "hedges_g_B": effect_b,
                        "same_direction": bool(np.sign(effect_a) == np.sign(effect_b)),
                        "mean_hedges_g": (effect_a + effect_b) / 2.0,
                    }
                )
    return pd.DataFrame(rows)


def build_report(
    epoch_data: dict,
    features: pd.DataFrame,
    tests: pd.DataFrame,
    laterality_tests: pd.DataFrame,
    consistency: pd.DataFrame,
    curves_finite: bool,
) -> str:
    total_trials = sum(item["epochs"].shape[0] for item in epoch_data.values())
    minimum_group = min(
        int(np.count_nonzero(item["directions"] == side))
        for item in epoch_data.values()
        for side in ("L", "R")
    )
    significant_count = int(tests["significant_fdr_0_05"].sum())
    consistent_count = int(consistency["same_direction"].sum())
    total_consistency = len(consistency)

    lines = [
        "C题问题1 稳健ERP曲线与左右视觉提示差异报告",
        "=" * 64,
        "曲线估计：Huber M估计 + 约120 ms局部三次Savitzky-Golay回归。",
        f"不确定性：试验级bootstrap {N_BOOTSTRAP}次，图中为逐时点95%点态置信区间。",
        f"方向检验：标签置换{N_PERMUTATIONS}次；共24项检验统一进行BH-FDR校正。",
        "早期窗口：250-500 ms（对应题面给出的P300典型范围）。",
        "晚期窗口：550-900 ms（用于捕捉本数据中明显的延迟正向响应）。",
        "侧化定义：所有试验统一使用F4-F3，不根据真实方向翻转符号，避免标签泄漏。",
        "",
    ]

    for file_name in EXPECTED_FILES:
        file_features = features[features["file_name"] == file_name]
        file_tests = tests[tests["file_name"] == file_name]
        largest = file_tests.loc[file_tests["hedges_g_R_minus_L"].abs().idxmax()]
        early_latency = file_features["early_peak_latency_s"]
        late_latency = file_features["late_peak_latency_s"]
        significant_file = int(file_tests["significant_fdr_0_05"].sum())
        data = epoch_data[file_name]
        n_left = int(np.count_nonzero(data["directions"] == "L"))
        n_right = int(np.count_nonzero(data["directions"] == "R"))
        lines.extend(
            [
                f"[{file_name}]",
                f"  有效试验：左{n_left}，右{n_right}",
                f"  早期正峰拟合潜伏期范围：{early_latency.min():.3f}-{early_latency.max():.3f} s",
                f"  晚期正峰拟合潜伏期范围：{late_latency.min():.3f}-{late_latency.max():.3f} s",
                "  最大左右效应："
                f"{largest['window']} / {largest['channel']}, "
                f"Hedges g={largest['hedges_g_R_minus_L']:+.3f}, "
                f"FDR p={largest['p_fdr_bh']:.4f}",
                f"  FDR显著窗口-通道数：{significant_file}/6",
                "",
            ]
        )

    strongest_lateral = laterality_tests.loc[
        laterality_tests["hedges_g_R_minus_L"].abs().idxmax()
    ]
    significant_lateral = int(laterality_tests["significant_fdr_0_05"].sum())
    lines.extend(
        [
            "总体结论",
            "-" * 64,
            f"共分析{total_trials}个清洁试验；左右方向检验中{significant_count}/24项通过BH-FDR 0.05。",
            f"A/B两名被试的效应方向在{consistent_count}/{total_consistency}个任务-窗口-通道组合中一致。",
            "固定F4-F3侧化的最大左右效应："
            f"{safe_stem(strongest_lateral['file_name'])} / {strongest_lateral['window']}, "
            f"Hedges g={strongest_lateral['hedges_g_R_minus_L']:+.3f}, "
            f"FDR p={strongest_lateral['p_fdr_bh']:.4f}；"
            f"通过FDR的侧化检验为{significant_lateral}/8。",
            "题面250-500 ms窗口必须报告；同时，图中550-900 ms若出现更强正向峰，",
            "应作为任务相关晚期成分单独报告，不应直接将其强行解释为经典P300。",
            "由于记录位置为Fz/F3/F4而非典型中央-顶区，对P300的生理命名应保持谨慎。",
            "",
            "自动验收",
            "-" * 64,
            f"{'PASS' if total_trials == 356 else 'FAIL'}  清洁试验总数应为356",
            f"{'PASS' if minimum_group >= 30 else 'FAIL'}  每个文件每个方向至少30个试验（最少{minimum_group}）",
            f"{'PASS' if curves_finite else 'FAIL'}  所有拟合曲线及置信区间均为有限数",
            f"{'PASS' if len(tests) == 24 else 'FAIL'}  应完成24项左右方向窗口检验",
            f"{'PASS' if tests['p_fdr_bh'].between(0, 1).all() else 'FAIL'}  所有FDR校正p值均位于[0,1]",
            f"{'PASS' if len(laterality_tests) == 8 else 'FAIL'}  应完成8项固定F4-F3侧化方向检验",
            f"{'PASS' if laterality_tests['p_fdr_bh'].between(0, 1).all() else 'FAIL'}  侧化检验FDR校正p值均位于[0,1]",
            "PASS  随机种子固定，bootstrap与置换检验可复现",
            "",
            "下一阶段：把曲线、窗口特征和固定F4-F3侧化整理为问题1的正式模型、公式与文字结论，",
            "并进行保留/剔除饱和试验的敏感性分析，确认主要结论不依赖单一伪影处理方案。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    epoch_dir = project_root / "data" / "processed" / "erp_epochs"
    output_dir = project_root / "results" / "04_erp_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [
        FILE_TO_EPOCH[file_name]
        for file_name in EXPECTED_FILES
        if not (epoch_dir / FILE_TO_EPOCH[file_name]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "缺少以下ERP片段，请先运行03_preprocess_erp.py：\n" + "\n".join(missing)
        )

    rng = np.random.default_rng(RANDOM_SEED)
    epoch_data: dict[str, dict] = {}
    curve_cache: dict[tuple[str, str], dict] = {}
    laterality_cache: dict[tuple[str, str], dict] = {}
    curve_rows: list[dict] = []
    feature_rows: list[dict] = []
    direction_test_rows: list[dict] = []
    laterality_curve_rows: list[dict] = []
    laterality_test_rows: list[dict] = []

    for file_name in EXPECTED_FILES:
        epoch_path = epoch_dir / FILE_TO_EPOCH[file_name]
        data = load_epochs(epoch_path)
        epoch_data[file_name] = data
        epochs = data["epochs"]
        directions = data["directions"]
        time_s = data["time_s"]
        sample_rate = data["sample_rate"]

        print(f"正在拟合：{file_name}")
        trial_lateral_by_side: dict[str, np.ndarray] = {}
        for side in ("L", "R"):
            side_epochs = epochs[directions == side]
            print(f"  {side}方向：{side_epochs.shape[0]}个试验，bootstrap {N_BOOTSTRAP}次")
            fitted = fit_curve(side_epochs, sample_rate)
            bootstrap = bootstrap_fitted_curves(side_epochs, sample_rate, rng)
            ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5], axis=0)
            curve_cache[(file_name, side)] = {
                "time_s": time_s,
                "curve": fitted,
                "bootstrap": bootstrap,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_trials": side_epochs.shape[0],
            }

            for channel_index, channel in enumerate(EEG_LABELS):
                for point_index, time_value in enumerate(time_s):
                    curve_rows.append(
                        {
                            "file_name": file_name,
                            "subject": SUBJECT_BY_FILE[file_name],
                            "task": TASK_BY_FILE[file_name],
                            "direction": side,
                            "channel": channel,
                            "n_trials": side_epochs.shape[0],
                            "time_s": time_value,
                            "fitted_amplitude": fitted[channel_index, point_index],
                            "ci95_low_pointwise": ci_low[channel_index, point_index],
                            "ci95_high_pointwise": ci_high[channel_index, point_index],
                        }
                    )
                feature_rows.append(
                    {
                        "file_name": file_name,
                        "subject": SUBJECT_BY_FILE[file_name],
                        "task": TASK_BY_FILE[file_name],
                        "direction": side,
                        "channel": channel,
                        "n_trials": side_epochs.shape[0],
                        **curve_features(fitted[channel_index], time_s),
                    }
                )

            # 固定侧化对所有试验均定义为F4-F3，不根据真实标签翻转符号。
            trial_lateral = side_epochs[:, 2, :] - side_epochs[:, 1, :]
            trial_lateral_by_side[side] = trial_lateral
            lateral_curve = fitted[2] - fitted[1]
            lateral_bootstrap = bootstrap[:, 2, :] - bootstrap[:, 1, :]
            lateral_low, lateral_high = np.percentile(lateral_bootstrap, [2.5, 97.5], axis=0)
            laterality_cache[(file_name, side)] = {
                "time_s": time_s,
                "curve": lateral_curve,
                "ci_low": lateral_low,
                "ci_high": lateral_high,
                "n_trials": side_epochs.shape[0],
            }
            for point_index, time_value in enumerate(time_s):
                laterality_curve_rows.append(
                    {
                        "file_name": file_name,
                        "subject": SUBJECT_BY_FILE[file_name],
                        "task": TASK_BY_FILE[file_name],
                        "direction": side,
                        "n_trials": side_epochs.shape[0],
                        "time_s": time_value,
                        "fitted_lateralization": lateral_curve[point_index],
                        "ci95_low_pointwise": lateral_low[point_index],
                        "ci95_high_pointwise": lateral_high[point_index],
                    }
                )
        for window_name, limits in WINDOWS.items():
            mask = window_mask(time_s, limits)
            for channel_index, channel in enumerate(EEG_LABELS):
                left_values = np.mean(
                    epochs[(directions == "L"), channel_index][:, mask],
                    axis=1,
                )
                right_values = np.mean(
                    epochs[(directions == "R"), channel_index][:, mask],
                    axis=1,
                )
                difference, p_value = permutation_test_independent(left_values, right_values, rng)
                ci_low, ci_high = bootstrap_difference_interval(left_values, right_values, rng)
                direction_test_rows.append(
                    {
                        "file_name": file_name,
                        "subject": SUBJECT_BY_FILE[file_name],
                        "task": TASK_BY_FILE[file_name],
                        "window": window_name,
                        "channel": channel,
                        "n_left": left_values.size,
                        "n_right": right_values.size,
                        "mean_left": float(np.mean(left_values)),
                        "mean_right": float(np.mean(right_values)),
                        "difference_R_minus_L": difference,
                        "difference_ci95_low": ci_low,
                        "difference_ci95_high": ci_high,
                        "hedges_g_R_minus_L": hedges_g(left_values, right_values),
                        "p_permutation": p_value,
                    }
                )

            left_lateral = np.mean(trial_lateral_by_side["L"][:, mask], axis=1)
            right_lateral = np.mean(trial_lateral_by_side["R"][:, mask], axis=1)
            lateral_difference, lateral_p = permutation_test_independent(
                left_lateral,
                right_lateral,
                rng,
            )
            lateral_ci_low, lateral_ci_high = bootstrap_difference_interval(
                left_lateral,
                right_lateral,
                rng,
            )
            laterality_test_rows.append(
                {
                    "file_name": file_name,
                    "subject": SUBJECT_BY_FILE[file_name],
                    "task": TASK_BY_FILE[file_name],
                    "window": window_name,
                    "n_left": left_lateral.size,
                    "n_right": right_lateral.size,
                    "mean_left_F4_minus_F3": float(np.mean(left_lateral)),
                    "mean_right_F4_minus_F3": float(np.mean(right_lateral)),
                    "difference_R_minus_L": lateral_difference,
                    "difference_ci95_low": lateral_ci_low,
                    "difference_ci95_high": lateral_ci_high,
                    "hedges_g_R_minus_L": hedges_g(left_lateral, right_lateral),
                    "p_permutation": lateral_p,
                }
            )

    curves_df = pd.DataFrame(curve_rows)
    features_df = pd.DataFrame(feature_rows)
    tests_df = pd.DataFrame(direction_test_rows)
    laterality_curves_df = pd.DataFrame(laterality_curve_rows)
    laterality_tests_df = pd.DataFrame(laterality_test_rows)

    tests_df["p_fdr_bh"] = fdr_bh(tests_df["p_permutation"].to_numpy())
    tests_df["significant_fdr_0_05"] = tests_df["p_fdr_bh"] < 0.05
    laterality_tests_df["p_fdr_bh"] = fdr_bh(
        laterality_tests_df["p_permutation"].to_numpy()
    )
    laterality_tests_df["significant_fdr_0_05"] = laterality_tests_df["p_fdr_bh"] < 0.05
    consistency_df = build_subject_consistency(tests_df)

    curve_path = output_dir / "erp_curve_points.csv"
    feature_path = output_dir / "erp_features.csv"
    test_path = output_dir / "direction_tests.csv"
    lateral_curve_path = output_dir / "laterality_curve_points.csv"
    lateral_feature_path = output_dir / "laterality_features.csv"
    consistency_path = output_dir / "subject_consistency.csv"
    report_path = output_dir / "erp_analysis_report.txt"

    curves_df.to_csv(curve_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    features_df.to_csv(feature_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    tests_df.to_csv(test_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    laterality_curves_df.to_csv(
        lateral_curve_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )
    laterality_tests_df.to_csv(
        lateral_feature_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )
    consistency_df.to_csv(
        consistency_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )

    save_fitted_curves_plot(curve_cache, output_dir / "erp_fitted_curves.png")
    save_difference_plot(curve_cache, output_dir / "direction_difference.png")
    save_laterality_plot(laterality_cache, output_dir / "laterality_curves.png")
    save_effect_heatmap(tests_df, output_dir / "effect_size_heatmap.png")

    curves_finite = bool(
        np.isfinite(
            curves_df[["fitted_amplitude", "ci95_low_pointwise", "ci95_high_pointwise"]]
            .to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            laterality_curves_df[
                ["fitted_lateralization", "ci95_low_pointwise", "ci95_high_pointwise"]
            ].to_numpy(dtype=float)
        ).all()
    )
    report_path.write_text(
        build_report(
            epoch_data,
            features_df,
            tests_df,
            laterality_tests_df,
            consistency_df,
            curves_finite,
        ),
        encoding="utf-8-sig",
    )

    print("\n稳健ERP拟合与统计检验完成。")
    print(f"分析报告：{report_path}")
    print(f"方向检验：{test_path}")
    print(f"曲线与图片目录：{output_dir}")
    print("\n请先打开 erp_analysis_report.txt，确认自动验收项目均显示 PASS。")


if __name__ == "__main__":
    main()
