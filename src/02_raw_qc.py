"""C题问题1的第二阶段：原始脑电质量检查。

把本文件放在 C:\\C\\src\\02_raw_qc.py 后直接运行。

前置条件：已经运行 01_parse_trials.py，并生成
    C:\\C\\data\\processed\\trial_info.csv

本程序只读取原始Fz、F3、F4与事件通道，不执行滤波，不覆盖任何原始数据。

输出目录：C:\\C\\results\\02_raw_qc
    raw_qc_summary.csv
    raw_qc_report.txt
    power_spectrum.png
    saturation_summary.png
    A_Task-1_time_series.png
    A_Task-2_time_series.png
    B_Task-1_time_series.png
    B_Task-2_time_series.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import welch


EXPECTED_FILES = (
    "VisualCogA_Task-1.mat",
    "VisualCogA_Task-2.mat",
    "VisualCogB_Task-1.mat",
    "VisualCogB_Task-2.mat",
)

EEG_LABELS = ("Fz", "F3", "F4")
CHANNEL_COLORS = ("#3366CC", "#DC3912", "#109618")
RAIL_VALUE = 999.999


def integrate_band(frequency: np.ndarray, power: np.ndarray, low: float, high: float) -> float:
    mask = (frequency >= low) & (frequency <= high)
    if np.count_nonzero(mask) < 2:
        return float("nan")
    return float(np.trapezoid(power[mask], frequency[mask]))


def dominant_frequency(
    frequency: np.ndarray,
    power: np.ndarray,
    low: float,
    high: float,
) -> float:
    mask = (frequency >= low) & (frequency <= high)
    if not np.any(mask):
        return float("nan")
    local_frequency = frequency[mask]
    local_power = power[mask]
    return float(local_frequency[int(np.argmax(local_power))])


def load_dataset(mat_path: Path) -> tuple[float, np.ndarray]:
    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    sample_rate = float(np.asarray(mat["SampleRate"]).squeeze())
    data = np.asarray(mat["data"], dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != 10:
        raise ValueError(f"{mat_path.name} 的data形状异常：{data.shape}")
    return sample_rate, data


def save_time_series_plot(
    mat_path: Path,
    sample_rate: float,
    data: np.ndarray,
    first_trial: pd.Series,
    output_path: Path,
) -> None:
    cue_sample = int(first_trial["cue_onset_sample"])
    start = max(0, cue_sample - round(1.0 * sample_rate))
    end = min(data.shape[1], cue_sample + round(5.0 * sample_rate))
    relative_time = (np.arange(start, end) - cue_sample) / sample_rate

    fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
    for channel_index, (label, color) in enumerate(zip(EEG_LABELS, CHANNEL_COLORS)):
        axes[channel_index].plot(
            relative_time,
            data[channel_index, start:end],
            color=color,
            linewidth=0.8,
        )
        axes[channel_index].axhline(1000, color="#777777", linestyle="--", linewidth=0.7)
        axes[channel_index].axhline(-1000, color="#777777", linestyle="--", linewidth=0.7)
        axes[channel_index].axvline(0, color="#7B1FA2", linestyle="--", linewidth=1.0)
        axes[channel_index].set_ylabel(label)
        axes[channel_index].grid(alpha=0.2)

    axes[3].plot(relative_time, data[6, start:end], color="#666666", linewidth=0.8)
    axes[3].axvline(0, color="#7B1FA2", linestyle="--", linewidth=1.0)
    axes[3].set_ylabel("ECG")
    axes[3].grid(alpha=0.2)

    axes[4].step(relative_time, data[7, start:end], where="post", label="VisCue", linewidth=1.2)
    axes[4].step(
        relative_time,
        data[8, start:end],
        where="post",
        label="Channel 9",
        linewidth=1.0,
        alpha=0.85,
    )
    axes[4].axvline(0, color="#7B1FA2", linestyle="--", linewidth=1.0, label="Cue onset")
    if pd.notna(first_trial["target_onset_s"]):
        target_relative = float(first_trial["target_onset_s"] - first_trial["cue_onset_s"])
        axes[4].axvline(
            target_relative,
            color="#FF8F00",
            linestyle="--",
            linewidth=1.0,
            label="Target onset",
        )
    axes[4].set_ylabel("Event")
    axes[4].set_xlabel("Time relative to cue onset (s)")
    axes[4].set_ylim(-2.4, 2.4)
    axes[4].grid(alpha=0.2)
    axes[4].legend(loc="upper right", ncol=4, fontsize=8)

    fig.suptitle(f"Raw recording around Trial 1: {mat_path.stem}", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(channel_df: pd.DataFrame, trial_df: pd.DataFrame) -> str:
    lines = [
        "C题问题1 原始脑电质量检查报告",
        "=" * 56,
        "本阶段仅检查原始Fz、F3、F4，不使用机器处理后的Decon通道。",
        "",
    ]

    for file_name in EXPECTED_FILES:
        file_rows = channel_df[channel_df["file_name"] == file_name]
        file_trials = trial_df[trial_df["file_name"] == file_name]
        lines.append(f"[{file_name}]")
        lines.append(
            f"  提示对齐窗口含±1000饱和值："
            f"{int(file_trials['saturation_flag'].sum())}/{len(file_trials)} 个试验"
        )
        for row in file_rows.itertuples(index=False):
            lines.append(
                f"  {row.channel}: mean={row.mean:.3f}, std={row.std:.3f}, "
                f"range=[{row.minimum:.3f}, {row.maximum:.3f}], "
                f"饱和={row.rail_samples}点({row.rail_percent:.3f}%), "
                f"45-70Hz主峰={row.dominant_45_70_hz:.2f}Hz, "
                f"59-61Hz/1-30Hz功率比={row.line60_to_1_30_ratio:.4f}"
            )
        lines.append("")

    total_saturated_trials = int(trial_df["saturation_flag"].sum())
    total_trials = len(trial_df)
    files_with_60_peak = int(
        channel_df["dominant_45_70_hz"].between(59.5, 60.5).groupby(channel_df["file_name"]).any().sum()
    )

    lines.extend(
        [
            "检查结论",
            "-" * 56,
            f"1. 共{total_trials}个试验，其中{total_saturated_trials}个提示对齐窗口含有硬饱和值。",
            f"2. {files_with_60_peak}/4个文件至少有一个脑电通道在45-70Hz范围内的主峰位于60Hz附近。",
            "3. 原始通道存在明显个体差异、直流偏移和低频漂移，不能直接叠加求ERP。",
            "4. 后续预处理应先处理饱和与严重伪影，再执行去趋势、带通和事件切片。",
            "5. 对ERP分析可采用0.5-30Hz带通；若分析更高频段，应单独处理60Hz工频成分。",
            "6. 饱和试验暂不删除，只保留标记；后续比较保留、修复和剔除三种策略。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    trial_csv = project_root / "data" / "processed" / "trial_info.csv"
    output_dir = project_root / "results" / "02_raw_qc"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not trial_csv.exists():
        raise FileNotFoundError(
            "没有找到 data\\processed\\trial_info.csv，请先运行 01_parse_trials.py。"
        )
    missing = [file_name for file_name in EXPECTED_FILES if not (raw_dir / file_name).exists()]
    if missing:
        raise FileNotFoundError("以下原始数据不存在：\n" + "\n".join(missing))

    trial_df = pd.read_csv(trial_csv, encoding="utf-8-sig")
    channel_rows: list[dict] = []
    psd_cache: dict[str, tuple[np.ndarray, list[np.ndarray]]] = {}

    for file_name in EXPECTED_FILES:
        print(f"正在检查：{file_name}")
        mat_path = raw_dir / file_name
        sample_rate, data = load_dataset(mat_path)
        file_trials = trial_df[trial_df["file_name"] == file_name].sort_values("trial_id")
        if file_trials.empty:
            raise ValueError(f"trial_info.csv中没有找到 {file_name} 的试验记录。")

        safe_stem = mat_path.stem.replace("VisualCog", "")
        save_time_series_plot(
            mat_path,
            sample_rate,
            data,
            file_trials.iloc[0],
            output_dir / f"{safe_stem}_time_series.png",
        )

        file_psd: list[np.ndarray] = []
        for channel_index, channel_label in enumerate(EEG_LABELS):
            signal = data[channel_index]
            frequency, power = welch(
                signal,
                fs=sample_rate,
                window="hann",
                nperseg=8192,
                noverlap=4096,
                detrend="linear",
                scaling="density",
            )
            file_psd.append(power)
            power_01_05 = integrate_band(frequency, power, 0.1, 0.5)
            power_1_30 = integrate_band(frequency, power, 1.0, 30.0)
            power_59_61 = integrate_band(frequency, power, 59.0, 61.0)
            rail_samples = int(np.count_nonzero(np.abs(signal) >= RAIL_VALUE))

            channel_rows.append(
                {
                    "file_name": file_name,
                    "subject": "A" if "VisualCogA" in file_name else "B",
                    "task": "Task-1" if "Task-1" in file_name else "Task-2",
                    "channel": channel_label,
                    "sample_rate_hz": sample_rate,
                    "samples": signal.size,
                    "mean": float(np.mean(signal)),
                    "std": float(np.std(signal)),
                    "minimum": float(np.min(signal)),
                    "q001": float(np.quantile(signal, 0.001)),
                    "median": float(np.median(signal)),
                    "q999": float(np.quantile(signal, 0.999)),
                    "maximum": float(np.max(signal)),
                    "rail_samples": rail_samples,
                    "rail_percent": rail_samples / signal.size * 100,
                    "power_0_1_0_5": power_01_05,
                    "power_1_30": power_1_30,
                    "power_59_61": power_59_61,
                    "low_to_1_30_ratio": power_01_05 / power_1_30 if power_1_30 > 0 else np.nan,
                    "line60_to_1_30_ratio": power_59_61 / power_1_30 if power_1_30 > 0 else np.nan,
                    "dominant_45_70_hz": dominant_frequency(frequency, power, 45.0, 70.0),
                }
            )
        psd_cache[file_name] = (frequency, file_psd)

    channel_df = pd.DataFrame(channel_rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, file_name in zip(axes.ravel(), EXPECTED_FILES):
        frequency, powers = psd_cache[file_name]
        mask = (frequency >= 0.5) & (frequency <= 100)
        for label, color, power in zip(EEG_LABELS, CHANNEL_COLORS, powers):
            axis.semilogy(frequency[mask], power[mask], label=label, color=color, linewidth=1.0)
        axis.axvline(60, color="#7B1FA2", linestyle="--", linewidth=1.0, label="60 Hz")
        axis.set_title(file_name.replace(".mat", ""))
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("PSD")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    fig.suptitle("Power spectral density of raw EEG channels", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "power_spectrum.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    pivot = channel_df.pivot(index="file_name", columns="channel", values="rail_percent")
    pivot = pivot.reindex(index=EXPECTED_FILES, columns=EEG_LABELS)
    fig, axis = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(pivot.index))
    width = 0.23
    for offset_index, (label, color) in enumerate(zip(EEG_LABELS, CHANNEL_COLORS)):
        offset = (offset_index - 1) * width
        bars = axis.bar(x + offset, pivot[label].to_numpy(), width, label=label, color=color)
        axis.bar_label(bars, fmt="%.2f%%", padding=3, fontsize=8)
    axis.set_xticks(x)
    axis.set_xticklabels([name.replace("VisualCog", "").replace(".mat", "") for name in pivot.index])
    axis.set_ylabel("Samples at ±1000 (%)")
    axis.set_title("Hard saturation in raw EEG channels")
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "saturation_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    summary_csv = output_dir / "raw_qc_summary.csv"
    report_txt = output_dir / "raw_qc_report.txt"
    channel_df.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.8g")
    report_txt.write_text(build_report(channel_df, trial_df), encoding="utf-8-sig")

    print("\n原始脑电质量检查完成。")
    print(f"通道指标：{summary_csv}")
    print(f"检查报告：{report_txt}")
    print(f"图片目录：{output_dir}")
    print("\n请先打开 raw_qc_report.txt、power_spectrum.png 和 saturation_summary.png。")


if __name__ == "__main__":
    main()
