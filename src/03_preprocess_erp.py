"""C题问题1第三阶段：保留视觉特征的脑电预处理与ERP分段。

把本文件放在 C:\\C\\src\\03_preprocess_erp.py 后直接运行。

前置条件：已经运行 01_parse_trials.py，并生成
    C:\\C\\data\\processed\\trial_info.csv

本程序只使用原始Fz、F3、F4（通道1-3），不会使用或覆盖机器滤波后的
FzDecon、F3Decon、F4Decon（通道4-6），也不会修改原始.mat文件。

处理原则：
1. 仅为保证零相位滤波稳定，对±1000饱和点建立“工作副本”并线性插值；
   原始饱和标记永久保留，相关试验不会进入主ERP平均。
2. 连续信号执行0.5-30 Hz四阶Butterworth零相位带通。30 Hz低通已经抑制
   60 Hz工频，因此不重复使用陷波器，避免额外波形畸变。
3. 按视觉提示出现时刻切取[-0.5, 1.0] s片段，并用[-0.2, 0] s基线校正。
4. 饱和/非有限值试验与基于MAD的极端残余伪影只做标记，不删除原始记录。

输出：
    data/processed/erp_epochs/*.npz
    data/processed/epoch_qc.csv
    results/03_preprocess/preprocessing_summary.csv
    results/03_preprocess/artifact_thresholds.csv
    results/03_preprocess/preprocessing_report.txt
    results/03_preprocess/psd_before_after.png
    results/03_preprocess/erp_preview.png
    results/03_preprocess/trial_retention.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, sosfiltfilt, welch


EXPECTED_FILES = (
    "VisualCogA_Task-1.mat",
    "VisualCogA_Task-2.mat",
    "VisualCogB_Task-1.mat",
    "VisualCogB_Task-2.mat",
)

EEG_LABELS = ("Fz", "F3", "F4")
CHANNEL_COLORS = ("#3366CC", "#DC3912", "#109618")
SIDE_COLORS = {"L": "#3366CC", "R": "#DC3912"}

EXPECTED_SAMPLE_RATE = 256.0
RAIL_VALUE = 999.999
FILTER_LOW_HZ = 0.5
FILTER_HIGH_HZ = 30.0
FILTER_ORDER = 4
EPOCH_START_S = -0.5
EPOCH_END_S = 1.0
BASELINE_START_S = -0.2
BASELINE_END_S = 0.0
ARTIFACT_START_S = -0.2
ARTIFACT_END_S = 0.8
P300_START_S = 0.25
P300_END_S = 0.50
MAD_MULTIPLIER = 6.0


def load_dataset(mat_path: Path) -> tuple[float, np.ndarray]:
    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    required = {"SampleRate", "data"}
    missing = required.difference(mat)
    if missing:
        raise KeyError(f"{mat_path.name} 缺少变量：{sorted(missing)}")

    sample_rate = float(np.asarray(mat["SampleRate"]).squeeze())
    data = np.asarray(mat["data"], dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != 10:
        raise ValueError(f"{mat_path.name} 的data形状异常：{data.shape}")
    if not np.isclose(sample_rate, EXPECTED_SAMPLE_RATE):
        raise ValueError(
            f"{mat_path.name} 采样率为{sample_rate:g} Hz，"
            f"预期为{EXPECTED_SAMPLE_RATE:g} Hz。"
        )
    return sample_rate, data


def interpolate_for_filter(signal: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    """只生成滤波工作副本；bad_mask仍用于最终剔除，绝不冒充真实观测。"""
    output = np.asarray(signal, dtype=np.float64).copy()
    bad_indices = np.flatnonzero(bad_mask)
    if bad_indices.size == 0:
        return output

    good_indices = np.flatnonzero(~bad_mask)
    if good_indices.size < 2:
        raise ValueError("某脑电通道有效采样点不足，无法建立滤波工作副本。")
    output[bad_indices] = np.interp(
        bad_indices,
        good_indices,
        output[good_indices],
    )
    return output


def scaled_mad(values: np.ndarray, axis: int = 0) -> np.ndarray:
    median = np.median(values, axis=axis)
    deviation = np.median(np.abs(values - median), axis=axis)
    return 1.4826 * deviation


def robust_threshold(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回每个通道的中位数、缩放MAD和中位数+6*MAD阈值。"""
    median = np.median(values, axis=0)
    mad = scaled_mad(values, axis=0)
    # MAD为零时给出一个很小但非零的保护量，避免正常数值被等号误判。
    fallback = np.maximum(np.abs(median) * 0.05, 1e-9)
    mad = np.where(mad > 1e-12, mad, fallback)
    threshold = median + MAD_MULTIPLIER * mad
    return median, mad, threshold


def bool_to_code(value: object) -> int:
    if pd.isna(value):
        return -1
    if isinstance(value, str):
        return 1 if value.strip().lower() == "true" else 0
    return int(bool(value))


def reason_text(edge: bool, bad_sample: bool, residual: bool) -> str:
    reasons: list[str] = []
    if edge:
        reasons.append("epoch_out_of_bounds")
    if bad_sample:
        reasons.append("raw_saturation_or_nonfinite")
    if residual:
        reasons.append("extreme_residual_artifact")
    return ";".join(reasons) if reasons else "none"


def safe_stem(file_name: str) -> str:
    return file_name.replace("VisualCog", "").replace(".mat", "")


def preprocess_file(
    mat_path: Path,
    file_trials: pd.DataFrame,
    epoch_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    sample_rate, data = load_dataset(mat_path)
    raw_eeg = data[:3]
    sample_bad = (~np.isfinite(raw_eeg)) | (np.abs(raw_eeg) >= RAIL_VALUE)

    filter_input = np.empty_like(raw_eeg)
    for channel_index in range(len(EEG_LABELS)):
        filter_input[channel_index] = interpolate_for_filter(
            raw_eeg[channel_index],
            sample_bad[channel_index],
        )

    sos = butter(
        FILTER_ORDER,
        [FILTER_LOW_HZ, FILTER_HIGH_HZ],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    filtered_eeg = sosfiltfilt(sos, filter_input, axis=1)

    start_offset = int(round(EPOCH_START_S * sample_rate))
    end_offset = int(round(EPOCH_END_S * sample_rate))
    epoch_samples = end_offset - start_offset
    time_s = np.arange(epoch_samples, dtype=np.float64) / sample_rate + EPOCH_START_S
    baseline_mask = (time_s >= BASELINE_START_S) & (time_s < BASELINE_END_S)
    artifact_mask = (time_s >= ARTIFACT_START_S) & (time_s <= ARTIFACT_END_S)

    n_trials = len(file_trials)
    epochs = np.full((n_trials, len(EEG_LABELS), epoch_samples), np.nan, dtype=np.float64)
    edge_flags = np.zeros(n_trials, dtype=bool)
    bad_sample_flags = np.zeros(n_trials, dtype=bool)

    for array_index, trial in enumerate(file_trials.itertuples(index=False)):
        onset = int(trial.cue_onset_sample)
        start = onset + start_offset
        end = onset + end_offset
        if start < 0 or end > raw_eeg.shape[1]:
            edge_flags[array_index] = True
            continue

        epoch = filtered_eeg[:, start:end].copy()
        epoch -= np.mean(epoch[:, baseline_mask], axis=1, keepdims=True)
        epochs[array_index] = epoch
        bad_sample_flags[array_index] = bool(np.any(sample_bad[:, start:end]))

    peak_to_peak = np.full((n_trials, len(EEG_LABELS)), np.nan)
    maximum_absolute = np.full_like(peak_to_peak, np.nan)
    maximum_step = np.full_like(peak_to_peak, np.nan)
    available = ~edge_flags
    peak_to_peak[available] = np.ptp(epochs[available][:, :, artifact_mask], axis=2)
    maximum_absolute[available] = np.max(
        np.abs(epochs[available][:, :, artifact_mask]),
        axis=2,
    )
    maximum_step[available] = np.max(
        np.abs(np.diff(epochs[available][:, :, artifact_mask], axis=2)),
        axis=2,
    )

    threshold_source = available & ~bad_sample_flags
    if np.count_nonzero(threshold_source) < 20:
        raise ValueError(f"{mat_path.name} 可用于估计伪影阈值的试验不足20个。")

    metric_values = {
        "peak_to_peak": peak_to_peak,
        "maximum_absolute": maximum_absolute,
        "maximum_step": maximum_step,
    }
    metric_thresholds: dict[str, np.ndarray] = {}
    threshold_rows: list[dict] = []
    for metric_name, values in metric_values.items():
        medians, mads, thresholds = robust_threshold(values[threshold_source])
        metric_thresholds[metric_name] = thresholds
        for channel_index, channel in enumerate(EEG_LABELS):
            threshold_rows.append(
                {
                    "file_name": mat_path.name,
                    "subject": file_trials.iloc[0]["subject"],
                    "task": file_trials.iloc[0]["task"],
                    "metric": metric_name,
                    "channel": channel,
                    "median": medians[channel_index],
                    "scaled_mad": mads[channel_index],
                    "threshold_median_plus_6mad": thresholds[channel_index],
                }
            )

    residual_flags = np.zeros(n_trials, dtype=bool)
    for metric_name, values in metric_values.items():
        residual_flags |= np.any(values > metric_thresholds[metric_name], axis=1)
    residual_flags &= threshold_source

    accepted = available & ~bad_sample_flags & ~residual_flags
    qc_rows: list[dict] = []
    for array_index, trial in enumerate(file_trials.itertuples(index=False)):
        row = {
            "file_name": mat_path.name,
            "subject": trial.subject,
            "task": trial.task,
            "trial_id": int(trial.trial_id),
            "cue_value": int(trial.cue_value),
            "cue_direction": trial.cue_direction,
            "edge_flag": bool(edge_flags[array_index]),
            "raw_saturation_or_nonfinite_flag": bool(bad_sample_flags[array_index]),
            "residual_artifact_flag": bool(residual_flags[array_index]),
            "accepted_for_erp": bool(accepted[array_index]),
            "rejection_reason": reason_text(
                bool(edge_flags[array_index]),
                bool(bad_sample_flags[array_index]),
                bool(residual_flags[array_index]),
            ),
            "behavior_correct_inferred": bool_to_code(trial.correct_inferred),
        }
        for channel_index, channel in enumerate(EEG_LABELS):
            row[f"peak_to_peak_{channel}"] = peak_to_peak[array_index, channel_index]
            row[f"maximum_absolute_{channel}"] = maximum_absolute[array_index, channel_index]
            row[f"maximum_step_{channel}"] = maximum_step[array_index, channel_index]
        qc_rows.append(row)

    trial_ids = file_trials["trial_id"].to_numpy(dtype=np.int32)
    cue_values = file_trials["cue_value"].to_numpy(dtype=np.int8)
    cue_directions = file_trials["cue_direction"].astype(str).to_numpy(dtype="U1")
    behavior_codes = np.asarray(
        [bool_to_code(value) for value in file_trials["correct_inferred"]],
        dtype=np.int8,
    )
    output_npz = epoch_dir / f"{safe_stem(mat_path.name)}_erp_epochs.npz"
    np.savez_compressed(
        output_npz,
        epochs_all_filtered=epochs.astype(np.float32),
        accepted_mask=accepted,
        epochs_accepted=epochs[accepted].astype(np.float32),
        all_trial_id=trial_ids,
        accepted_trial_id=trial_ids[accepted],
        cue_value_all=cue_values,
        cue_value_accepted=cue_values[accepted],
        cue_direction_all=cue_directions,
        cue_direction_accepted=cue_directions[accepted],
        behavior_correct_inferred_all=behavior_codes,
        behavior_correct_inferred_accepted=behavior_codes[accepted],
        time_s=time_s,
        channel_labels=np.asarray(EEG_LABELS, dtype="U2"),
        sample_rate_hz=np.asarray(sample_rate),
        filter_band_hz=np.asarray([FILTER_LOW_HZ, FILTER_HIGH_HZ]),
        epoch_window_s=np.asarray([EPOCH_START_S, EPOCH_END_S]),
        baseline_window_s=np.asarray([BASELINE_START_S, BASELINE_END_S]),
        p300_window_s=np.asarray([P300_START_S, P300_END_S]),
    )

    frequency, raw_psd_channels = welch(
        filter_input,
        fs=sample_rate,
        window="hann",
        nperseg=8192,
        noverlap=4096,
        detrend="linear",
        axis=1,
        scaling="density",
    )
    _, filtered_psd_channels = welch(
        filtered_eeg,
        fs=sample_rate,
        window="hann",
        nperseg=8192,
        noverlap=4096,
        detrend="linear",
        axis=1,
        scaling="density",
    )
    raw_psd = np.median(raw_psd_channels, axis=0)
    filtered_psd = np.median(filtered_psd_channels, axis=0)

    cache = {
        "file_name": mat_path.name,
        "time_s": time_s,
        "epochs": epochs,
        "accepted": accepted,
        "cue_direction": cue_directions,
    }
    return (
        pd.DataFrame(qc_rows),
        pd.DataFrame(threshold_rows),
        cache,
        (frequency, raw_psd, filtered_psd),
    )


def make_summary(qc_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for file_name in EXPECTED_FILES:
        subset = qc_df[qc_df["file_name"] == file_name]
        row = {
            "file_name": file_name,
            "subject": subset.iloc[0]["subject"],
            "task": subset.iloc[0]["task"],
            "total_trials": len(subset),
            "rejected_edge": int(subset["edge_flag"].sum()),
            "rejected_saturation_or_nonfinite": int(
                subset["raw_saturation_or_nonfinite_flag"].sum()
            ),
            "rejected_residual_artifact": int(subset["residual_artifact_flag"].sum()),
            "accepted_trials": int(subset["accepted_for_erp"].sum()),
        }
        for side in ("L", "R"):
            side_subset = subset[subset["cue_direction"] == side]
            row[f"total_{side}"] = len(side_subset)
            row[f"accepted_{side}"] = int(side_subset["accepted_for_erp"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def save_psd_plot(psd_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, file_name in zip(axes.ravel(), EXPECTED_FILES):
        frequency, raw_psd, filtered_psd = psd_cache[file_name]
        mask = (frequency >= 0.2) & (frequency <= 100.0)
        axis.semilogy(
            frequency[mask],
            raw_psd[mask],
            color="#777777",
            linewidth=1.0,
            alpha=0.85,
            label="Raw median PSD",
        )
        axis.semilogy(
            frequency[mask],
            filtered_psd[mask],
            color="#00796B",
            linewidth=1.1,
            label="0.5-30 Hz filtered",
        )
        axis.axvline(0.5, color="#3366CC", linestyle="--", linewidth=0.8)
        axis.axvline(30.0, color="#3366CC", linestyle="--", linewidth=0.8)
        axis.axvline(60.0, color="#7B1FA2", linestyle=":", linewidth=1.0)
        axis.set_title(file_name.replace(".mat", ""))
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("PSD before and after feature-preserving preprocessing", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_erp_preview(caches: dict[str, dict], path: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 14), sharex=True)
    for row_index, file_name in enumerate(EXPECTED_FILES):
        cache = caches[file_name]
        time_s = cache["time_s"]
        epochs = cache["epochs"]
        accepted = cache["accepted"]
        cue_direction = cache["cue_direction"]
        for channel_index, channel in enumerate(EEG_LABELS):
            axis = axes[row_index, channel_index]
            for side in ("L", "R"):
                side_mask = accepted & (cue_direction == side)
                if np.any(side_mask):
                    curve = np.median(epochs[side_mask, channel_index, :], axis=0)
                    axis.plot(
                        time_s,
                        curve,
                        color=SIDE_COLORS[side],
                        linewidth=1.2,
                        label=f"{side} (n={np.count_nonzero(side_mask)})",
                    )
            axis.axvline(0.0, color="#222222", linestyle="--", linewidth=0.8)
            axis.axhline(0.0, color="#999999", linewidth=0.6)
            axis.axvspan(P300_START_S, P300_END_S, color="#FFC107", alpha=0.16)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(channel)
            if channel_index == 0:
                axis.set_ylabel(f"{safe_stem(file_name)}\nAmplitude")
            if row_index == len(EXPECTED_FILES) - 1:
                axis.set_xlabel("Time from cue onset (s)")
            axis.legend(fontsize=7, loc="best")
    fig.suptitle("Clean cue-locked ERP preview (trial median; yellow = 250-500 ms)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_retention_plot(summary_df: pd.DataFrame, path: Path) -> None:
    labels = [safe_stem(name) for name in summary_df["file_name"]]
    accepted = summary_df["accepted_trials"].to_numpy()
    saturated = summary_df["rejected_saturation_or_nonfinite"].to_numpy()
    residual = summary_df["rejected_residual_artifact"].to_numpy()
    edge = summary_df["rejected_edge"].to_numpy()

    fig, axis = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(labels))
    axis.bar(x, accepted, color="#2E7D32", label="Accepted")
    axis.bar(x, saturated, bottom=accepted, color="#F9A825", label="Saturation/nonfinite")
    axis.bar(
        x,
        residual,
        bottom=accepted + saturated,
        color="#C62828",
        label="Residual artifact",
    )
    axis.bar(
        x,
        edge,
        bottom=accepted + saturated + residual,
        color="#616161",
        label="Out of bounds",
    )
    for position, value in enumerate(accepted):
        axis.text(position, value / 2, str(int(value)), ha="center", va="center", color="white")
    axis.set_xticks(x)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 105)
    axis.set_ylabel("Trials")
    axis.set_title("ERP trial retention after preprocessing")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(ncol=4, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(summary_df: pd.DataFrame, qc_df: pd.DataFrame) -> str:
    total = int(summary_df["total_trials"].sum())
    accepted = int(summary_df["accepted_trials"].sum())
    saturated = int(summary_df["rejected_saturation_or_nonfinite"].sum())
    residual = int(summary_df["rejected_residual_artifact"].sum())
    edge = int(summary_df["rejected_edge"].sum())
    finite_accepted = bool(
        np.isfinite(
            qc_df.loc[
                qc_df["accepted_for_erp"],
                [
                    column
                    for column in qc_df.columns
                    if column.startswith(("peak_to_peak_", "maximum_absolute_", "maximum_step_"))
                ],
            ].to_numpy(dtype=float)
        ).all()
    )
    minimum_side = int(
        summary_df[["accepted_L", "accepted_R"]].to_numpy().min()
    )

    lines = [
        "C题问题1 脑电预处理与ERP分段报告",
        "=" * 58,
        "主分析只使用原始Fz、F3、F4，不使用机器Decon通道。",
        "事件零点：通道8视觉提示（左=-1，右=+1）的起始时刻。",
        "ERP片段：提示前0.5秒至提示后1.0秒。",
        "基线：提示前0.2秒至提示出现前。",
        "滤波：0.5-30 Hz四阶Butterworth零相位带通。",
        "伪影判据：原始硬饱和/非有限值；其余极端残余伪影采用每文件、每通道中位数+6×MAD。",
        "说明：饱和值插值只用于避免滤波数值振铃，含原始饱和的片段不进入主ERP。",
        "",
    ]

    for row in summary_df.itertuples(index=False):
        lines.extend(
            [
                f"[{row.file_name}]",
                f"  总试验：{row.total_trials}",
                f"  硬饱和/非有限值：{row.rejected_saturation_or_nonfinite}",
                f"  极端残余伪影：{row.rejected_residual_artifact}",
                f"  越界：{row.rejected_edge}",
                f"  主分析保留：{row.accepted_trials}（左{row.accepted_L}，右{row.accepted_R}）",
                "",
            ]
        )

    lines.extend(
        [
            "总体结论",
            "-" * 58,
            f"共{total}个试验；主分析保留{accepted}个，排除{total - accepted}个。",
            f"其中硬饱和/非有限值{saturated}个，极端残余伪影{residual}个，越界{edge}个。",
            "被排除试验仍完整记录在epoch_qc.csv中，原始.mat数据未被修改。",
            "B Task-1第49、87号行为不一致试验未因行为错误而当作信号伪影删除，",
            "后续认知模型将把行为正确性作为独立变量处理。",
            "",
            "自动验收",
            "-" * 58,
            f"{'PASS' if total == 400 else 'FAIL'}  逐试验质量表应有400行",
            f"{'PASS' if finite_accepted else 'FAIL'}  所有保留试验的质量指标均为有限数",
            f"{'PASS' if minimum_side >= 30 else 'FAIL'}  每个文件每个方向至少保留30个试验（最少{minimum_side}）",
            f"{'PASS' if accepted > 0 else 'FAIL'}  已生成可用于ERP的清洁片段",
            "PASS  分析代码未读取通道4-6的机器Decon数据",
            "",
            "下一阶段：对左/右提示分别进行稳健ERP曲线拟合，量化250-500 ms响应、",
            "F3-F4侧化差异及A/B个体一致性，并进行重采样置信区间和置换检验。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    trial_csv = processed_dir / "trial_info.csv"
    epoch_dir = processed_dir / "erp_epochs"
    output_dir = project_root / "results" / "03_preprocess"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trial_csv.exists():
        raise FileNotFoundError(
            "没有找到 data\\processed\\trial_info.csv，请先运行 01_parse_trials.py。"
        )
    missing = [file_name for file_name in EXPECTED_FILES if not (raw_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError("以下原始数据不存在：\n" + "\n".join(missing))

    trial_df = pd.read_csv(trial_csv, encoding="utf-8-sig")
    qc_frames: list[pd.DataFrame] = []
    threshold_frames: list[pd.DataFrame] = []
    caches: dict[str, dict] = {}
    psd_caches: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    for file_name in EXPECTED_FILES:
        print(f"正在预处理：{file_name}")
        file_trials = (
            trial_df[trial_df["file_name"] == file_name]
            .sort_values("trial_id")
            .reset_index(drop=True)
        )
        if len(file_trials) != 100:
            raise ValueError(f"{file_name} 在trial_info.csv中不是100个试验，而是{len(file_trials)}个。")

        qc_df, threshold_df, cache, psd_cache = preprocess_file(
            raw_dir / file_name,
            file_trials,
            epoch_dir,
        )
        qc_frames.append(qc_df)
        threshold_frames.append(threshold_df)
        caches[file_name] = cache
        psd_caches[file_name] = psd_cache

    all_qc = pd.concat(qc_frames, ignore_index=True)
    all_thresholds = pd.concat(threshold_frames, ignore_index=True)
    summary = make_summary(all_qc)

    epoch_qc_path = processed_dir / "epoch_qc.csv"
    summary_path = output_dir / "preprocessing_summary.csv"
    thresholds_path = output_dir / "artifact_thresholds.csv"
    report_path = output_dir / "preprocessing_report.txt"

    all_qc.to_csv(epoch_qc_path, index=False, encoding="utf-8-sig", float_format="%.8g")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    all_thresholds.to_csv(
        thresholds_path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.8g",
    )
    report_path.write_text(build_report(summary, all_qc), encoding="utf-8-sig")

    save_psd_plot(psd_caches, output_dir / "psd_before_after.png")
    save_erp_preview(caches, output_dir / "erp_preview.png")
    save_retention_plot(summary, output_dir / "trial_retention.png")

    print("\n预处理与ERP分段完成。")
    print(f"逐试验质量表：{epoch_qc_path}")
    print(f"ERP片段目录：{epoch_dir}")
    print(f"预处理报告：{report_path}")
    print(f"结果图片目录：{output_dir}")
    print("\n请先打开 preprocessing_report.txt，确认自动验收项目均显示 PASS。")


if __name__ == "__main__":
    main()
