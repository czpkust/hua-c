"""C题第一阶段：解析四组连续脑电记录并建立逐试验事件表。

把本文件放在 C:\\C\\src\\01_parse_trials.py 后直接运行。

输入：
    C:\\C\\data\\raw\\VisualCogA_Task-1.mat
    C:\\C\\data\\raw\\VisualCogA_Task-2.mat
    C:\\C\\data\\raw\\VisualCogB_Task-1.mat
    C:\\C\\data\\raw\\VisualCogB_Task-2.mat

输出：
    data/processed/trial_info.csv
    results/01_data_check/dataset_summary.csv
    results/01_data_check/data_check_report.txt
    results/01_data_check/event_timeline.png

本程序只解析原始数据和事件，不进行滤波，也不使用通道4-6的机器滤波结果。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat


EXPECTED_FILES = (
    "VisualCogA_Task-1.mat",
    "VisualCogA_Task-2.mat",
    "VisualCogB_Task-1.mat",
    "VisualCogB_Task-2.mat",
)

RAIL_VALUE = 999.999
ERP_WINDOW_START_S = -0.5
ERP_WINDOW_END_S = 1.0


@dataclass(frozen=True)
class EventRun:
    start: int
    end: int
    value: int

    @property
    def length(self) -> int:
        return self.end - self.start + 1


def find_runs(signal: np.ndarray) -> list[EventRun]:
    """查找分段常值的非零事件；数值变化时会自动切成新事件。"""
    x = np.asarray(signal).ravel()
    if x.size == 0:
        return []

    starts = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    ends = np.r_[starts[1:] - 1, x.size - 1]
    output: list[EventRun] = []
    for start, end in zip(starts, ends):
        value = float(x[start])
        if value == 0:
            continue
        if not np.isclose(value, round(value)):
            raise ValueError(f"事件通道出现非整数编码：{value}")
        output.append(EventRun(int(start), int(end), int(round(value))))
    return output


def extract_labels(data_label: np.ndarray) -> list[str]:
    labels: list[str] = []
    for cell in np.asarray(data_label).ravel():
        value = cell
        while isinstance(value, np.ndarray) and value.size == 1:
            value = value.item()
        if isinstance(value, np.ndarray):
            value = "".join(str(item) for item in value.ravel())
        labels.append(str(value))
    return labels


def side_from_value(value: int | float | None) -> str:
    if value is None or pd.isna(value):
        return ""
    if value < 0:
        return "L"
    if value > 0:
        return "R"
    return ""


def first_or_none(events: list[EventRun]) -> EventRun | None:
    return events[0] if events else None


def safe_time(sample: int | None, sample_rate: float) -> float:
    return np.nan if sample is None else sample / sample_rate


def parse_file(mat_path: Path) -> tuple[list[dict], dict, dict]:
    mat = loadmat(mat_path, squeeze_me=False, struct_as_record=False)
    required = {"SampleRate", "DataLabel", "data"}
    missing = required.difference(mat)
    if missing:
        raise KeyError(f"{mat_path.name} 缺少变量：{sorted(missing)}")

    sample_rate = float(np.asarray(mat["SampleRate"]).squeeze())
    labels = extract_labels(mat["DataLabel"])
    data = np.asarray(mat["data"], dtype=np.float64)

    if data.ndim != 2 or data.shape[0] != 10:
        raise ValueError(f"{mat_path.name} 的data形状不是10×N，而是 {data.shape}")
    if len(labels) != 10:
        raise ValueError(f"{mat_path.name} 的DataLabel数量不是10，而是 {len(labels)}")

    subject = "A" if "VisualCogA" in mat_path.name else "B"
    task = 1 if "Task-1" in mat_path.name else 2
    n_samples = data.shape[1]
    timestamps = data[9]
    timestamp_step = np.diff(timestamps)
    expected_step = 1.0 / sample_rate
    timestamp_continuous = bool(
        timestamp_step.size > 0
        and np.all(timestamp_step > 0)
        and np.allclose(timestamp_step, expected_step, rtol=0, atol=1e-12)
    )

    cue_runs = [event for event in find_runs(data[7]) if abs(event.value) == 1]
    channel9_runs = find_runs(data[8])

    trial_rows: list[dict] = []
    raw_eeg = data[:3]

    for index, cue in enumerate(cue_runs):
        next_cue_start = cue_runs[index + 1].start if index + 1 < len(cue_runs) else n_samples
        trial_events = [
            event
            for event in channel9_runs
            if cue.start < event.start < next_cue_start
        ]

        target: EventRun | None
        response: EventRun | None
        target_candidates: list[EventRun]
        response_candidates: list[EventRun]

        if task == 1:
            target_candidates = [event for event in trial_events if abs(event.value) == 1]
            response_candidates = []
            target = first_or_none(target_candidates)
            response = None
        else:
            target_candidates = [event for event in trial_events if abs(event.value) == 1]
            response_candidates = [event for event in trial_events if abs(event.value) == 2]
            target = first_or_none(target_candidates)
            response = first_or_none(response_candidates)

        epoch_start = max(0, cue.start + round(ERP_WINDOW_START_S * sample_rate))
        epoch_end = min(n_samples, cue.start + round(ERP_WINDOW_END_S * sample_rate))
        epoch = raw_eeg[:, epoch_start:epoch_end]
        rail_mask = np.abs(epoch) >= RAIL_VALUE
        finite_epoch = np.isfinite(epoch)

        target_value = target.value if target else np.nan
        response_value = response.value if response else np.nan
        cue_target_match = (
            bool(np.sign(cue.value) == np.sign(target.value)) if target else pd.NA
        )
        target_response_match = (
            bool(np.sign(target.value) == np.sign(response.value))
            if target and response
            else pd.NA
        )

        abnormal_reasons: list[str] = []
        if len(target_candidates) == 0:
            abnormal_reasons.append("missing_target_event")
        elif len(target_candidates) > 1:
            abnormal_reasons.append("multiple_target_events")

        if task == 1:
            if target is not None and not cue_target_match:
                abnormal_reasons.append("task1_cue_channel9_mismatch")
        else:
            if len(response_candidates) == 0:
                abnormal_reasons.append("missing_click_event")
            elif len(response_candidates) > 1:
                abnormal_reasons.append("multiple_click_events")
            if target is not None and response is not None and not target_response_match:
                abnormal_reasons.append("task2_target_click_mismatch")

        if rail_mask.any():
            abnormal_reasons.append("raw_eeg_rail_saturation")
        if not finite_epoch.all():
            abnormal_reasons.append("nan_or_inf_in_epoch")

        target_onset_sample = target.start if target else None
        target_offset_sample = target.end if target else None
        response_sample = response.start if response else None

        if task == 2 and target is not None and response is not None:
            reaction_time_s = (response.start - target.start) / sample_rate
            reaction_time_proxy_s = np.nan
            reaction_time_source = "target_onset_to_click_pulse"
        elif task == 1 and target is not None:
            reaction_time_s = np.nan
            reaction_time_proxy_s = target.length / sample_rate
            reaction_time_source = "channel9_segment_duration_proxy"
        else:
            reaction_time_s = np.nan
            reaction_time_proxy_s = np.nan
            reaction_time_source = "unavailable"

        if task == 1:
            correct_inferred = cue_target_match
        else:
            correct_inferred = target_response_match

        trial_rows.append(
            {
                "subject": subject,
                "task": f"Task-{task}",
                "file_name": mat_path.name,
                "trial_id": index + 1,
                "sample_rate_hz": sample_rate,
                "cue_value": cue.value,
                "cue_direction": side_from_value(cue.value),
                "cue_onset_sample": cue.start,
                "cue_offset_sample": cue.end,
                "cue_onset_s": cue.start / sample_rate,
                "cue_offset_s": cue.end / sample_rate,
                "cue_duration_s": cue.length / sample_rate,
                "target_value": target_value,
                "target_side": side_from_value(target_value),
                "target_onset_sample": target_onset_sample,
                "target_offset_sample": target_offset_sample,
                "target_onset_s": safe_time(target_onset_sample, sample_rate),
                "target_offset_s": safe_time(target_offset_sample, sample_rate),
                "cue_to_target_s": (
                    (target.start - cue.start) / sample_rate if target else np.nan
                ),
                "response_value": response_value,
                "response_side": side_from_value(response_value),
                "response_sample": response_sample,
                "response_time_s": safe_time(response_sample, sample_rate),
                "reaction_time_s": reaction_time_s,
                "reaction_time_proxy_s": reaction_time_proxy_s,
                "reaction_time_source": reaction_time_source,
                "cue_target_match": cue_target_match,
                "target_response_match": target_response_match,
                "correct_inferred": correct_inferred,
                "rail_Fz": bool(rail_mask[0].any()),
                "rail_F3": bool(rail_mask[1].any()),
                "rail_F4": bool(rail_mask[2].any()),
                "saturation_flag": bool(rail_mask.any()),
                "saturation_fraction": float(rail_mask.mean()),
                "nonfinite_flag": bool(not finite_epoch.all()),
                "abnormal_flag": bool(abnormal_reasons),
                "abnormal_reason": ";".join(abnormal_reasons) if abnormal_reasons else "none",
            }
        )

    trial_df = pd.DataFrame(trial_rows)
    rail_counts = [int(np.count_nonzero(np.abs(channel) >= RAIL_VALUE)) for channel in raw_eeg]

    summary = {
        "subject": subject,
        "task": f"Task-{task}",
        "file_name": mat_path.name,
        "sample_rate_hz": sample_rate,
        "channels": data.shape[0],
        "samples": n_samples,
        "duration_s": n_samples / sample_rate,
        "duration_min": n_samples / sample_rate / 60,
        "all_values_finite": bool(np.isfinite(data).all()),
        "timestamp_continuous": timestamp_continuous,
        "timestamp_step_s": float(np.median(timestamp_step)),
        "cue_trials": len(cue_runs),
        "cue_left": int(sum(event.value < 0 for event in cue_runs)),
        "cue_right": int(sum(event.value > 0 for event in cue_runs)),
        "target_events": int(trial_df["target_onset_s"].notna().sum()),
        "click_events": int(trial_df["response_time_s"].notna().sum()),
        "inferred_correct": int((trial_df["correct_inferred"] == True).sum()),
        "inferred_incorrect": int((trial_df["correct_inferred"] == False).sum()),
        "saturated_trials": int(trial_df["saturation_flag"].sum()),
        "rail_Fz_samples": rail_counts[0],
        "rail_F3_samples": rail_counts[1],
        "rail_F4_samples": rail_counts[2],
        "channel9_label": labels[8],
    }

    plot_info = {
        "subject": subject,
        "task": task,
        "sample_rate": sample_rate,
        "cue_signal": data[7],
        "channel9_signal": data[8],
        "first_cue_sample": cue_runs[0].start if cue_runs else 0,
    }
    return trial_rows, summary, plot_info


def save_event_plot(plot_items: list[dict], output_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=False)
    for axis, item in zip(axes, plot_items):
        fs = item["sample_rate"]
        start = max(0, item["first_cue_sample"] - round(2 * fs))
        end = min(len(item["cue_signal"]), start + round(22 * fs))
        time = np.arange(start, end) / fs
        axis.step(time, item["cue_signal"][start:end], where="post", label="VisCue", linewidth=1.2)
        axis.step(
            time,
            item["channel9_signal"][start:end],
            where="post",
            label="Channel 9",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.set_title(f"Subject {item['subject']}  Task-{item['task']}")
        axis.set_ylabel("Event code")
        axis.set_ylim(-2.4, 2.4)
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Event-code overview around the first trials", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_report(trials: pd.DataFrame, summary: pd.DataFrame) -> str:
    lines = [
        "C题脑电数据第一阶段检查报告",
        "=" * 50,
        f"文件数：{len(summary)}",
        f"试验总数：{len(trials)}",
        "",
    ]

    for row in summary.itertuples(index=False):
        subset = trials[(trials["subject"] == row.subject) & (trials["task"] == row.task)]
        lines.extend(
            [
                f"[{row.subject} {row.task}] {row.file_name}",
                f"  数据形状：10 × {row.samples}",
                f"  采样率：{row.sample_rate_hz:g} Hz",
                f"  时长：{row.duration_min:.3f} min",
                f"  提示试验：{row.cue_trials}（左 {row.cue_left}，右 {row.cue_right}）",
                f"  目标事件：{row.target_events}",
                f"  点击事件：{row.click_events}",
                f"  推断一致/正确：{row.inferred_correct}",
                f"  推断不一致/错误：{row.inferred_incorrect}",
                f"  ERP窗口含饱和值的试验：{row.saturated_trials}",
                f"  时间戳连续：{row.timestamp_continuous}",
                f"  全部数值有限：{row.all_values_finite}",
            ]
        )
        mismatches = subset[subset["correct_inferred"] == False]["trial_id"].tolist()
        if mismatches:
            lines.append(f"  不一致试验编号：{mismatches}")
        lines.append("")

    checks = {
        "总试验数应为400": len(trials) == 400,
        "每个文件应有100次提示": bool((summary["cue_trials"] == 100).all()),
        "采样率均应为256 Hz": bool(np.allclose(summary["sample_rate_hz"], 256)),
        "时间戳均连续": bool(summary["timestamp_continuous"].all()),
        "数据中无NaN或Inf": bool(summary["all_values_finite"].all()),
        "两个Task-2文件均有100次点击": bool(
            (summary.loc[summary["task"] == "Task-2", "click_events"] == 100).all()
        ),
    }
    lines.append("自动验收")
    lines.append("-" * 50)
    for name, passed in checks.items():
        lines.append(f"{'PASS' if passed else 'CHECK'}  {name}")

    lines.extend(
        [
            "",
            "说明：",
            "1. Task-1没有独立的±2点击脉冲，reaction_time_proxy_s只是通道9非零段持续时间。",
            "2. Task-2的±1段按目标侧解释，±2单点按点击侧解释。",
            "3. saturation_flag只检查提示开始前0.5秒至开始后1.0秒的原始Fz/F3/F4。",
            "4. 本阶段不使用FzDecon、F3Decon、F4Decon进行分析。",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    result_dir = project_root / "results" / "01_data_check"
    processed_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    missing_files = [name for name in EXPECTED_FILES if not (raw_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(
            "以下数据文件不存在，请检查是否放在 data\\raw 中：\n"
            + "\n".join(missing_files)
        )

    all_trials: list[dict] = []
    summaries: list[dict] = []
    plot_items: list[dict] = []

    for file_name in EXPECTED_FILES:
        print(f"正在解析：{file_name}")
        trial_rows, summary, plot_info = parse_file(raw_dir / file_name)
        all_trials.extend(trial_rows)
        summaries.append(summary)
        plot_items.append(plot_info)

    trial_df = pd.DataFrame(all_trials)
    summary_df = pd.DataFrame(summaries)

    trial_csv = processed_dir / "trial_info.csv"
    summary_csv = result_dir / "dataset_summary.csv"
    report_txt = result_dir / "data_check_report.txt"
    event_png = result_dir / "event_timeline.png"

    trial_df.to_csv(trial_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig", float_format="%.6f")
    report_txt.write_text(build_report(trial_df, summary_df), encoding="utf-8-sig")
    save_event_plot(plot_items, event_png)

    print("\n解析完成。")
    print(f"试验总数：{len(trial_df)}")
    print(f"逐试验表：{trial_csv}")
    print(f"数据汇总：{summary_csv}")
    print(f"检查报告：{report_txt}")
    print(f"事件图：{event_png}")
    print("\n请先打开 data_check_report.txt，确认自动验收项目均显示 PASS。")


if __name__ == "__main__":
    main()
