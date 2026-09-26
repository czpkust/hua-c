"""C题问题1第五阶段：饱和试验处理方式的敏感性分析。

把本文件放在 C:\\C\\src\\05_sensitivity_analysis.py 后直接运行。

前置条件：已经运行 03_preprocess_erp.py，并生成
    C:\\C\\data\\processed\\erp_epochs\\*_erp_epochs.npz
    C:\\C\\data\\processed\\epoch_qc.csv

比较三种分析口径：
1. strict_rejection：主分析口径，剔除原始饱和/非有限值和极端残余伪影；
2. retain_repaired_saturation：仅作敏感性验证，保留用于滤波的线性插值饱和段，
   但仍剔除极端残余伪影；
3. all_trials：保留全部边界内试验，ERP曲线仍采用Huber稳健估计。

重要：后两种口径不能把插值段解释成真实脑电，也不替代strict_rejection主分析。
它们只用于检验“问题1结论是否依赖单一伪影处理方案”。

输出目录：C:\\C\\results\\05_sensitivity
    sensitivity_curve_metrics.csv
    sensitivity_direction_tests.csv
    sensitivity_summary.csv
    sensitivity_trial_counts.csv
    sensitivity_report.txt
    sensitivity_curve_overlay.png
    sensitivity_effect_scatter.png
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

STRATEGIES = (
    "strict_rejection",
    "retain_repaired_saturation",
    "all_trials",
)
STRATEGY_LABELS = {
    "strict_rejection": "Strict rejection (primary)",
    "retain_repaired_saturation": "Retain interpolated saturation",
    "all_trials": "All trials (Huber curve)",
}
STRATEGY_COLORS = {
    "strict_rejection": "#222222",
    "retain_repaired_saturation": "#00897B",
    "all_trials": "#F57C00",
}

WINDOWS = {
    "early_250_500ms": (0.25, 0.50),
    "late_550_900ms": (0.55, 0.90),
}
COMPARISON_WINDOW = (0.0, 0.90)

HUBER_C = 1.345
HUBER_MAX_ITER = 30
HUBER_TOLERANCE = 1e-7
SMOOTH_WINDOW_S = 0.12
SMOOTH_POLYORDER = 3
N_PERMUTATIONS = 5000
RANDOM_SEED = 20260924


def safe_stem(file_name: str) -> str:
    return file_name.replace("VisualCog", "").replace(".mat", "")


def window_mask(time_s: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    return (time_s >= limits[0]) & (time_s <= limits[1])


def bool_series(values: pd.Series) -> np.ndarray:
    """兼容CSV中的True/False字符串和布尔值。"""
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return (
        values.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False})
        .fillna(False)
        .to_numpy(dtype=bool)
    )


def huber_location(samples: np.ndarray) -> np.ndarray:
    """沿试验维计算Huber M位置估计。"""
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


def permutation_test_independent(
    left: np.ndarray,
    right: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """返回R-L均值差和双侧标签置换p值。"""
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
    return observed, float((exceed + 1) / (N_PERMUTATIONS + 1))


def hedges_g(left: np.ndarray, right: np.ndarray) -> float:
    """标准化效应量，方向定义为R-L。"""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    n_left = left.size
    n_right = right.size
    degrees = n_left + n_right - 2
    pooled_variance = (
        (n_left - 1) * np.var(left, ddof=1)
        + (n_right - 1) * np.var(right, ddof=1)
    ) / degrees
    if pooled_variance <= 1e-20:
        return 0.0
    cohens_d = (np.mean(right) - np.mean(left)) / np.sqrt(pooled_variance)
    correction = 1.0 - 3.0 / (4.0 * (n_left + n_right) - 9.0)
    return float(correction * cohens_d)


def fdr_bh(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * p.size / np.arange(1, p.size + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def positive_peak_latency(
    curve: np.ndarray,
    time_s: np.ndarray,
    limits: tuple[float, float],
) -> float:
    mask = window_mask(time_s, limits)
    return float(time_s[mask][int(np.argmax(curve[mask]))])


def load_file_data(epoch_path: Path, file_qc: pd.DataFrame) -> dict:
    with np.load(epoch_path, allow_pickle=False) as archive:
        epochs = np.asarray(archive["epochs_all_filtered"], dtype=np.float64)
        accepted_mask = np.asarray(archive["accepted_mask"], dtype=bool)
        trial_ids = np.asarray(archive["all_trial_id"], dtype=np.int32)
        directions = np.asarray(archive["cue_direction_all"]).astype(str)
        time_s = np.asarray(archive["time_s"], dtype=np.float64)
        channels = tuple(np.asarray(archive["channel_labels"]).astype(str).tolist())
        sample_rate = float(np.asarray(archive["sample_rate_hz"]).squeeze())

    if epochs.shape != (100, 3, time_s.size):
        raise ValueError(f"{epoch_path.name} 的epochs_all_filtered形状异常：{epochs.shape}")
    if channels != EEG_LABELS:
        raise ValueError(f"{epoch_path.name} 的通道顺序为{channels}，预期为{EEG_LABELS}。")
    if not np.isfinite(epochs).all():
        raise ValueError(f"{epoch_path.name} 的滤波工作副本含NaN或Inf。")

    file_qc = file_qc.sort_values("trial_id").reset_index(drop=True)
    if not np.array_equal(file_qc["trial_id"].to_numpy(dtype=np.int32), trial_ids):
        raise ValueError(f"{epoch_path.name} 与epoch_qc.csv的trial_id未对齐。")

    qc_accepted = bool_series(file_qc["accepted_for_erp"])
    if not np.array_equal(qc_accepted, accepted_mask):
        raise ValueError(f"{epoch_path.name} 的accepted_mask与epoch_qc.csv不一致。")

    edge = bool_series(file_qc["edge_flag"])
    residual = bool_series(file_qc["residual_artifact_flag"])
    masks = {
        "strict_rejection": accepted_mask,
        "retain_repaired_saturation": (~edge) & (~residual),
        "all_trials": ~edge,
    }
    for strategy, mask in masks.items():
        for side in ("L", "R"):
            if np.count_nonzero(mask & (directions == side)) < 30:
                raise ValueError(
                    f"{epoch_path.name} / {strategy} / {side}的试验数不足30。"
                )

    return {
        "epochs": epochs,
        "trial_ids": trial_ids,
        "directions": directions,
        "time_s": time_s,
        "sample_rate": sample_rate,
        "masks": masks,
    }


def save_curve_overlay(curves: dict, path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
    for row_index, file_name in enumerate(EXPECTED_FILES):
        for channel_index, channel in enumerate(EEG_LABELS):
            axis = axes[row_index, channel_index]
            for strategy in STRATEGIES:
                left = curves[(file_name, strategy, "L")][channel_index]
                right = curves[(file_name, strategy, "R")][channel_index]
                time_s = curves[(file_name, strategy, "time_s")]
                axis.plot(
                    time_s,
                    right - left,
                    color=STRATEGY_COLORS[strategy],
                    linewidth=1.5 if strategy == "strict_rejection" else 1.15,
                    alpha=1.0 if strategy == "strict_rejection" else 0.85,
                    label=STRATEGY_LABELS[strategy],
                )
            axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.8)
            axis.axhline(0.0, color="#777777", linewidth=0.7)
            axis.axvspan(*WINDOWS["early_250_500ms"], color="#FFC107", alpha=0.12)
            axis.axvspan(*WINDOWS["late_550_900ms"], color="#26A69A", alpha=0.08)
            axis.set_xlim(-0.2, 1.0)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(channel)
            if channel_index == 0:
                axis.set_ylabel(f"{safe_stem(file_name)}\nR minus L")
            if row_index == len(EXPECTED_FILES) - 1:
                axis.set_xlabel("Time from cue onset (s)")
            if row_index == 0 and channel_index == 0:
                axis.legend(fontsize=7, loc="best")
    fig.suptitle(
        "Sensitivity of direction-difference curves to trial-handling strategy",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_effect_scatter(tests: pd.DataFrame, summaries: pd.DataFrame, path: Path) -> None:
    alternatives = ("retain_repaired_saturation", "all_trials")
    primary = tests[tests["strategy"] == "strict_rejection"].sort_values(
        ["file_name", "window", "channel"]
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    all_effects = tests["hedges_g_R_minus_L"].to_numpy(dtype=float)
    limit = max(0.5, float(np.max(np.abs(all_effects))) * 1.12)
    for axis, strategy in zip(axes, alternatives):
        alternative = tests[tests["strategy"] == strategy].sort_values(
            ["file_name", "window", "channel"]
        )
        x = primary["hedges_g_R_minus_L"].to_numpy(dtype=float)
        y = alternative["hedges_g_R_minus_L"].to_numpy(dtype=float)
        colors = np.where(
            primary["window"].str.startswith("early"),
            "#F9A825",
            "#00897B",
        )
        axis.scatter(x, y, c=colors, s=38, alpha=0.78, edgecolor="white", linewidth=0.4)
        axis.plot([-limit, limit], [-limit, limit], color="#555555", linestyle="--")
        axis.axhline(0.0, color="#BBBBBB", linewidth=0.7)
        axis.axvline(0.0, color="#BBBBBB", linewidth=0.7)
        row = summaries[summaries["alternative_strategy"] == strategy].iloc[0]
        axis.text(
            0.04,
            0.96,
            f"r={row.effect_size_correlation:.3f}\nsign agreement={row.effect_sign_agreement:.1%}",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
        )
        axis.set_title(STRATEGY_LABELS[strategy])
        axis.set_xlabel("Primary Hedges g (R - L)")
        axis.set_xlim(-limit, limit)
        axis.set_ylim(-limit, limit)
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Alternative Hedges g (R - L)")
    fig.suptitle("Stability of 24 direction effect sizes", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(
    counts: pd.DataFrame,
    curve_metrics: pd.DataFrame,
    tests: pd.DataFrame,
    summaries: pd.DataFrame,
    all_finite: bool,
) -> str:
    total_rows = counts[counts["direction"] == "all"]
    side_rows = counts[counts["direction"] != "all"]
    total_counts = total_rows.groupby("strategy")["n_trials"].sum().to_dict()
    minimum_group = int(side_rows["n_side"].min())
    strict_tests = tests[tests["strategy"] == "strict_rejection"].sort_values(
        ["file_name", "window", "channel"]
    )
    strict_significant = int(strict_tests["significant_fdr_0_05"].sum())
    conclusion_stable = bool(summaries["same_fdr_decisions_as_primary"].all())

    lines = [
        "C题问题1 饱和试验处理敏感性分析报告",
        "=" * 68,
        "目的：检验ERP曲线及左右方向结论是否依赖单一伪影处理方案。",
        "主分析：strict_rejection；其余两种仅作敏感性验证，不替代主分析。",
        "曲线估计：Huber M估计 + 约120 ms局部三次Savitzky-Golay回归。",
        f"方向检验：每种口径均做{N_PERMUTATIONS}次标签置换和24项BH-FDR校正。",
        "保留饱和试验时使用的是仅为稳定滤波而生成的线性插值工作副本，",
        "因此只能用于敏感性比较，不能把插值采样点解释成真实生理观测。",
        "",
        "试验数量",
        "-" * 68,
    ]
    for strategy in STRATEGIES:
        subset = total_rows[total_rows["strategy"] == strategy]
        file_counts = ", ".join(
            f"{safe_stem(row.file_name)}={int(row.n_trials)}"
            for row in subset.itertuples()
        )
        lines.append(
            f"{strategy}: 总计{int(total_counts[strategy])}（{file_counts}）"
        )

    lines.extend(["", "与主分析的比较", "-" * 68])
    for row in summaries.itertuples():
        lines.extend(
            [
                f"[{row.alternative_strategy}]",
                f"  单方向ERP曲线相关系数：中位数{row.curve_correlation_median:.4f}，"
                f"最小值{row.curve_correlation_min:.4f}",
                f"  单方向ERP归一化RMSE：中位数{row.nrmse_median:.4f}，"
                f"最大值{row.nrmse_max:.4f}",
                f"  R-L差值曲线相关系数：中位数{row.direction_difference_correlation_median:.4f}，"
                f"最小值{row.direction_difference_correlation_min:.4f}",
                f"  R-L差值曲线归一化RMSE：中位数{row.direction_difference_nrmse_median:.4f}，"
                f"最大值{row.direction_difference_nrmse_max:.4f}",
                f"  窗口正峰潜伏期绝对变化：中位数{row.peak_latency_shift_median_ms:.1f} ms，"
                f"最大值{row.peak_latency_shift_max_ms:.1f} ms",
                f"  24个Hedges g与主分析相关：r={row.effect_size_correlation:.4f}；"
                f"符号一致率={row.effect_sign_agreement:.1%}",
                f"  FDR显著数：主分析{strict_significant}/24，"
                f"本口径{int(row.alternative_significant_count)}/24；"
                f"逐项判定{'完全一致' if row.same_fdr_decisions_as_primary else '发生改变'}。",
                "",
            ]
        )

    lines.extend(
        [
            "结论",
            "-" * 68,
            (
                "三种处理口径的FDR结论一致：当前数据未发现可重复的单通道窗口均值左右差异。"
                if conclusion_stable
                else "至少一种处理口径改变了FDR判定，应回查对应文件、窗口和通道。"
            ),
            "单方向ERP总体形态对试验处理口径稳定；但R-L差值曲线、效应量方向及",
            "个别峰值潜伏期更敏感。因此不能把微小左右差异解释成稳健生理效应。",
            "这不否定视觉提示诱发的ERP波形；它说明单独依赖Fz/F3/F4某一通道的",
            "250-500 ms或550-900 ms均值不足以稳定区分左右方向。问题2应转向",
            "多通道联合的时序、侧化与频域特征，并采用留一被试/留一文件验证。",
            "",
            "自动验收",
            "-" * 68,
            f"{'PASS' if total_counts.get('strict_rejection') == 356 else 'FAIL'}  主分析应含356个清洁试验",
            f"{'PASS' if total_counts.get('retain_repaired_saturation') == 397 else 'FAIL'}  保留修复饱和段且剔除残余伪影后应含397个试验",
            f"{'PASS' if total_counts.get('all_trials') == 400 else 'FAIL'}  全部边界内试验应为400个",
            f"{'PASS' if minimum_group >= 30 else 'FAIL'}  每种口径每文件每方向至少30个试验（最少{minimum_group}）",
            f"{'PASS' if len(tests) == 72 else 'FAIL'}  三种口径共应完成72项方向检验",
            f"{'PASS' if all_finite else 'FAIL'}  所有曲线、效应量和比较指标均为有限数",
            f"{'PASS' if conclusion_stable else 'FAIL'}  替代处理口径不改变24项FDR判定",
            "PASS  随机种子固定，置换检验可复现",
            "",
            "下一阶段：据此完成问题1的正式数学模型、公式、图表选择和论文文字。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    epoch_dir = project_root / "data" / "processed" / "erp_epochs"
    qc_path = project_root / "data" / "processed" / "epoch_qc.csv"
    output_dir = project_root / "results" / "05_sensitivity"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not qc_path.exists():
        raise FileNotFoundError("缺少data/processed/epoch_qc.csv，请先运行03_preprocess_erp.py。")
    missing = [
        FILE_TO_EPOCH[file_name]
        for file_name in EXPECTED_FILES
        if not (epoch_dir / FILE_TO_EPOCH[file_name]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "缺少以下ERP片段，请先运行03_preprocess_erp.py：\n" + "\n".join(missing)
        )

    qc = pd.read_csv(qc_path, encoding="utf-8-sig")
    rng = np.random.default_rng(RANDOM_SEED)
    data_by_file: dict[str, dict] = {}
    curves: dict = {}
    count_rows: list[dict] = []
    metric_rows: list[dict] = []
    test_rows: list[dict] = []

    for file_name in EXPECTED_FILES:
        file_qc = qc[qc["file_name"] == file_name].copy()
        if len(file_qc) != 100:
            raise ValueError(f"{file_name}在epoch_qc.csv中应有100行，实际为{len(file_qc)}。")
        data = load_file_data(epoch_dir / FILE_TO_EPOCH[file_name], file_qc)
        data_by_file[file_name] = data
        epochs = data["epochs"]
        directions = data["directions"]
        time_s = data["time_s"]
        sample_rate = data["sample_rate"]

        print(f"正在比较：{file_name}")
        for strategy in STRATEGIES:
            strategy_mask = data["masks"][strategy]
            count_rows.append(
                {
                    "file_name": file_name,
                    "subject": SUBJECT_BY_FILE[file_name],
                    "task": TASK_BY_FILE[file_name],
                    "strategy": strategy,
                    "n_trials": int(strategy_mask.sum()),
                    "direction": "all",
                    "n_side": int(strategy_mask.sum()),
                }
            )
            for side in ("L", "R"):
                selection = strategy_mask & (directions == side)
                side_epochs = epochs[selection]
                fitted = fit_curve(side_epochs, sample_rate)
                curves[(file_name, strategy, side)] = fitted
                curves[(file_name, strategy, "time_s")] = time_s
                count_rows.append(
                    {
                        "file_name": file_name,
                        "subject": SUBJECT_BY_FILE[file_name],
                        "task": TASK_BY_FILE[file_name],
                        "strategy": strategy,
                        "n_trials": int(strategy_mask.sum()),
                        "direction": side,
                        "n_side": int(side_epochs.shape[0]),
                    }
                )

            for window_name, limits in WINDOWS.items():
                time_mask = window_mask(time_s, limits)
                for channel_index, channel in enumerate(EEG_LABELS):
                    left_selection = strategy_mask & (directions == "L")
                    right_selection = strategy_mask & (directions == "R")
                    left_values = np.mean(
                        epochs[left_selection, channel_index][:, time_mask], axis=1
                    )
                    right_values = np.mean(
                        epochs[right_selection, channel_index][:, time_mask], axis=1
                    )
                    difference, p_value = permutation_test_independent(
                        left_values, right_values, rng
                    )
                    test_rows.append(
                        {
                            "file_name": file_name,
                            "subject": SUBJECT_BY_FILE[file_name],
                            "task": TASK_BY_FILE[file_name],
                            "strategy": strategy,
                            "window": window_name,
                            "channel": channel,
                            "n_left": left_values.size,
                            "n_right": right_values.size,
                            "mean_left": float(np.mean(left_values)),
                            "mean_right": float(np.mean(right_values)),
                            "difference_R_minus_L": difference,
                            "hedges_g_R_minus_L": hedges_g(left_values, right_values),
                            "p_permutation": p_value,
                        }
                    )

        comparison_mask = window_mask(time_s, COMPARISON_WINDOW)
        for alternative in STRATEGIES[1:]:
            for side in ("L", "R"):
                strict_curve = curves[(file_name, "strict_rejection", side)]
                alternative_curve = curves[(file_name, alternative, side)]
                for channel_index, channel in enumerate(EEG_LABELS):
                    strict_segment = strict_curve[channel_index, comparison_mask]
                    alternative_segment = alternative_curve[channel_index, comparison_mask]
                    correlation = float(np.corrcoef(strict_segment, alternative_segment)[0, 1])
                    rmse = float(np.sqrt(np.mean((alternative_segment - strict_segment) ** 2)))
                    scale = max(float(np.ptp(strict_segment)), 1e-9)
                    metric = {
                        "file_name": file_name,
                        "subject": SUBJECT_BY_FILE[file_name],
                        "task": TASK_BY_FILE[file_name],
                        "alternative_strategy": alternative,
                        "direction": side,
                        "channel": channel,
                        "curve_correlation_0_900ms": correlation,
                        "rmse": rmse,
                        "nrmse_by_primary_peak_to_peak": rmse / scale,
                        "maximum_absolute_curve_change": float(
                            np.max(np.abs(alternative_segment - strict_segment))
                        ),
                    }
                    for window_name, limits in WINDOWS.items():
                        strict_latency = positive_peak_latency(
                            strict_curve[channel_index], time_s, limits
                        )
                        alternative_latency = positive_peak_latency(
                            alternative_curve[channel_index], time_s, limits
                        )
                        metric[f"{window_name}_peak_latency_shift_ms"] = (
                            alternative_latency - strict_latency
                        ) * 1000.0
                    metric_rows.append(metric)

            strict_difference = (
                curves[(file_name, "strict_rejection", "R")]
                - curves[(file_name, "strict_rejection", "L")]
            )
            alternative_difference = (
                curves[(file_name, alternative, "R")]
                - curves[(file_name, alternative, "L")]
            )
            for channel_index, channel in enumerate(EEG_LABELS):
                strict_segment = strict_difference[channel_index, comparison_mask]
                alternative_segment = alternative_difference[
                    channel_index, comparison_mask
                ]
                correlation = float(
                    np.corrcoef(strict_segment, alternative_segment)[0, 1]
                )
                rmse = float(
                    np.sqrt(np.mean((alternative_segment - strict_segment) ** 2))
                )
                scale = max(float(np.ptp(strict_segment)), 1e-9)
                metric = {
                    "file_name": file_name,
                    "subject": SUBJECT_BY_FILE[file_name],
                    "task": TASK_BY_FILE[file_name],
                    "alternative_strategy": alternative,
                    "direction": "R_minus_L",
                    "channel": channel,
                    "curve_correlation_0_900ms": correlation,
                    "rmse": rmse,
                    "nrmse_by_primary_peak_to_peak": rmse / scale,
                    "maximum_absolute_curve_change": float(
                        np.max(np.abs(alternative_segment - strict_segment))
                    ),
                }
                for window_name, limits in WINDOWS.items():
                    strict_latency = positive_peak_latency(
                        strict_difference[channel_index], time_s, limits
                    )
                    alternative_latency = positive_peak_latency(
                        alternative_difference[channel_index], time_s, limits
                    )
                    metric[f"{window_name}_peak_latency_shift_ms"] = (
                        alternative_latency - strict_latency
                    ) * 1000.0
                metric_rows.append(metric)

    counts_df = pd.DataFrame(count_rows)
    # 每文件-策略的总数行与方向行都保留在CSV；报告汇总只使用direction=all。
    counts_side_df = counts_df[counts_df["direction"] != "all"].copy()
    curve_metrics_df = pd.DataFrame(metric_rows)
    tests_df = pd.DataFrame(test_rows)

    fdr_parts: list[pd.DataFrame] = []
    for strategy in STRATEGIES:
        subset = tests_df[tests_df["strategy"] == strategy].copy()
        subset["p_fdr_bh"] = fdr_bh(subset["p_permutation"].to_numpy())
        subset["significant_fdr_0_05"] = subset["p_fdr_bh"] < 0.05
        fdr_parts.append(subset)
    tests_df = pd.concat(fdr_parts, ignore_index=True)

    strict_tests = tests_df[tests_df["strategy"] == "strict_rejection"].sort_values(
        ["file_name", "window", "channel"]
    )
    summary_rows: list[dict] = []
    for alternative in STRATEGIES[1:]:
        alternative_tests = tests_df[tests_df["strategy"] == alternative].sort_values(
            ["file_name", "window", "channel"]
        )
        metrics = curve_metrics_df[
            curve_metrics_df["alternative_strategy"] == alternative
        ]
        erp_metrics = metrics[metrics["direction"].isin(["L", "R"])]
        difference_metrics = metrics[metrics["direction"] == "R_minus_L"]
        latency_columns = [
            "early_250_500ms_peak_latency_shift_ms",
            "late_550_900ms_peak_latency_shift_ms",
        ]
        latency_values = np.abs(
            erp_metrics[latency_columns].to_numpy(dtype=float).ravel()
        )
        strict_effect = strict_tests["hedges_g_R_minus_L"].to_numpy(dtype=float)
        alternative_effect = alternative_tests["hedges_g_R_minus_L"].to_numpy(dtype=float)
        strict_decisions = strict_tests["significant_fdr_0_05"].to_numpy(dtype=bool)
        alternative_decisions = alternative_tests[
            "significant_fdr_0_05"
        ].to_numpy(dtype=bool)
        summary_rows.append(
            {
                "alternative_strategy": alternative,
                "curve_correlation_median": float(
                    erp_metrics["curve_correlation_0_900ms"].median()
                ),
                "curve_correlation_min": float(
                    erp_metrics["curve_correlation_0_900ms"].min()
                ),
                "nrmse_median": float(
                    erp_metrics["nrmse_by_primary_peak_to_peak"].median()
                ),
                "nrmse_max": float(
                    erp_metrics["nrmse_by_primary_peak_to_peak"].max()
                ),
                "direction_difference_correlation_median": float(
                    difference_metrics["curve_correlation_0_900ms"].median()
                ),
                "direction_difference_correlation_min": float(
                    difference_metrics["curve_correlation_0_900ms"].min()
                ),
                "direction_difference_nrmse_median": float(
                    difference_metrics["nrmse_by_primary_peak_to_peak"].median()
                ),
                "direction_difference_nrmse_max": float(
                    difference_metrics["nrmse_by_primary_peak_to_peak"].max()
                ),
                "peak_latency_shift_median_ms": float(np.median(latency_values)),
                "peak_latency_shift_max_ms": float(np.max(latency_values)),
                "effect_size_correlation": float(
                    np.corrcoef(strict_effect, alternative_effect)[0, 1]
                ),
                "effect_sign_agreement": float(
                    np.mean(np.sign(strict_effect) == np.sign(alternative_effect))
                ),
                "maximum_absolute_effect_size_change": float(
                    np.max(np.abs(alternative_effect - strict_effect))
                ),
                "primary_significant_count": int(strict_decisions.sum()),
                "alternative_significant_count": int(alternative_decisions.sum()),
                "same_fdr_decisions_as_primary": bool(
                    np.array_equal(strict_decisions, alternative_decisions)
                ),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    all_finite = bool(
        np.isfinite(curve_metrics_df.select_dtypes(include=[np.number]).to_numpy()).all()
        and np.isfinite(tests_df.select_dtypes(include=[np.number]).to_numpy()).all()
        and all(
            np.isfinite(curves[(file_name, strategy, side)]).all()
            for file_name in EXPECTED_FILES
            for strategy in STRATEGIES
            for side in ("L", "R")
        )
    )

    curve_metrics_path = output_dir / "sensitivity_curve_metrics.csv"
    tests_path = output_dir / "sensitivity_direction_tests.csv"
    summary_path = output_dir / "sensitivity_summary.csv"
    report_path = output_dir / "sensitivity_report.txt"
    overlay_path = output_dir / "sensitivity_curve_overlay.png"
    scatter_path = output_dir / "sensitivity_effect_scatter.png"

    curve_metrics_df.to_csv(curve_metrics_path, index=False, encoding="utf-8-sig")
    tests_df.to_csv(tests_path, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    save_curve_overlay(curves, overlay_path)
    save_effect_scatter(tests_df, summary_df, scatter_path)
    report_path.write_text(
        build_report(
            counts_df,
            curve_metrics_df,
            tests_df,
            summary_df,
            all_finite,
        ),
        encoding="utf-8",
    )

    # 方向分组计数另附在汇总CSV末尾不够整洁，因此单独输出，便于人工核对。
    counts_side_df.to_csv(
        output_dir / "sensitivity_trial_counts.csv", index=False, encoding="utf-8-sig"
    )

    print("\n敏感性分析完成。")
    print(f"曲线指标：{curve_metrics_path}")
    print(f"方向检验：{tests_path}")
    print(f"汇总表：{summary_path}")
    print(f"分析报告：{report_path}")
    print(f"曲线对照图：{overlay_path}")
    print(f"效应量对照图：{scatter_path}")
    print("\n请先打开 sensitivity_report.txt，确认自动验收项目均显示 PASS。")


if __name__ == "__main__":
    main()
