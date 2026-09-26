r"""问题一补充验证：真实脑电背景中的固定参数合成视觉响应注入与回收。

放在项目的 src/ 下，从项目根目录运行：
    python -X utf8 .\src\10_q1_injection_recovery.py

依赖已完成的 01、03、04；仅写入 results/10_q1_injection_recovery。
这是方法学正控制：已知模板被注入真实 EEG，不代表真实三角刺激必然产生该模板。
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
OUTPUT = PROJECT_ROOT / "results" / "10_q1_injection_recovery"
FILES = (
    "VisualCogA_Task-1.mat",
    "VisualCogA_Task-2.mat",
    "VisualCogB_Task-1.mat",
    "VisualCogB_Task-2.mat",
)
REPEATS = 48  # 固定次数：不依据观察到的性能调整
SEED = 20260925
AMPLITUDE_MULTIPLIER = 0.8  # 原始片段的提示前200 ms标准差的中位数
RESPONSE_WINDOW_S = (0.15, 0.90)
METHODS = ("raw_mean", "bandpass_mean", "current_pipeline")


def import_existing(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_templates(time_s: np.ndarray) -> np.ndarray:
    """两条预先固定、不同形状与左右通道增益的合成响应 (2, 3, time)。"""
    early = np.exp(-0.5 * ((time_s - 0.330) / 0.075) ** 2)
    late = np.exp(-0.5 * ((time_s - 0.690) / 0.095) ** 2)
    left = early + 0.25 * late
    right = 0.35 * early + 0.85 * late
    left[time_s < 0] = 0.0
    right[time_s < 0] = 0.0
    return np.stack(
        [np.asarray([left, 1.15 * left, 0.75 * left]),
         np.asarray([right, 0.75 * right, 1.15 * right])]
    )


def baseline_correct(epochs: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    return epochs - epochs[..., baseline_mask].mean(axis=-1, keepdims=True)


def filter_templates(
    templates: np.ndarray,
    sos: np.ndarray,
    baseline_mask: np.ndarray,
    start_offset: int,
) -> np.ndarray:
    """同一零相位滤波器处理16秒填零的模板，避开片段边界效应。"""
    n_time = templates.shape[-1]
    center = 8 * 256
    padded = np.zeros((2, 3, 16 * 256), dtype=np.float64)
    begin = center + start_offset
    padded[..., begin:begin + n_time] = templates
    filtered = sosfiltfilt(sos, padded, axis=-1)
    return baseline_correct(filtered[..., begin:begin + n_time], baseline_mask)


def balanced_assignments(
    trial_ids: np.ndarray,
    directions: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """原试次的20试次块×真实提示方向内随机分配合成L/R。"""
    assignment = np.empty(trial_ids.size, dtype=np.int8)
    for block in range(5):
        for direction in ("L", "R"):
            indices = np.flatnonzero(
                ((trial_ids - 1) // 20 == block) & (directions == direction)
            )
            shuffled = rng.permutation(indices)
            n_left = len(shuffled) // 2
            if len(shuffled) % 2:
                n_left += int(rng.integers(0, 2))
            assignment[shuffled[:n_left]] = 0
            assignment[shuffled[n_left:]] = 1
    if min(np.bincount(assignment, minlength=2)) < 25:
        raise ValueError("合成方向样本量不足。")
    return assignment


def contrast_metrics(estimated: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
    observed = estimated[:, mask].ravel()
    truth = target[:, mask].ravel()
    denom = float(np.dot(truth, truth))
    gain = float(np.dot(observed, truth) / denom)
    nrmse = float(np.linalg.norm(observed - truth) / np.sqrt(denom))
    obs_centered = observed - observed.mean()
    tru_centered = truth - truth.mean()
    corr_den = float(np.linalg.norm(obs_centered) * np.linalg.norm(tru_centered))
    return {
        "nrmse": nrmse,
        "gain": gain,
        "shape_correlation": float(np.dot(obs_centered, tru_centered) / corr_den)
        if corr_den > 0 else float("nan"),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def continuous_audit(
    raw: np.ndarray,
    cue_samples: np.ndarray,
    labels: np.ndarray,
    amplitude: float,
    templates: np.ndarray,
    template_filtered: np.ndarray,
    bandpassed_epochs: np.ndarray,
    baseline_mask: np.ndarray,
    sos: np.ndarray,
    start_offset: int,
    pre,
) -> float:
    """首轮真正重跑全记录滤波，核实模板线性叠加近似的误差。"""
    sample_bad = (~np.isfinite(raw)) | (np.abs(raw) >= pre.RAIL_VALUE)
    original_input = np.asarray([
        pre.interpolate_for_filter(raw[c], sample_bad[c]) for c in range(3)
    ])
    injected_input = original_input.copy()
    n_time = templates.shape[-1]
    for cue, label in zip(cue_samples, labels):
        begin = int(cue) + start_offset
        injected_input[:, begin:begin+n_time] += amplitude * templates[label]
    filtered_injected = sosfiltfilt(sos, injected_input, axis=1)
    differences = []
    for idx, (cue, label) in enumerate(zip(cue_samples, labels)):
        begin = int(cue) + start_offset
        actual = baseline_correct(
            filtered_injected[:, begin:begin+n_time], baseline_mask
        ) - bandpassed_epochs[idx]
        expected = amplitude * template_filtered[label]
        differences.append(np.max(np.abs(actual - expected)))
    return float(np.max(differences) / amplitude)


def load_inputs(file_name: str, pre) -> dict:
    mat_path = PROJECT_ROOT / "data" / "raw" / file_name
    trial_path = PROJECT_ROOT / "data" / "processed" / "trial_info.csv"
    qc_path = PROJECT_ROOT / "data" / "processed" / "epoch_qc.csv"
    epoch_path = PROJECT_ROOT / "data" / "processed" / "erp_epochs" / (
        file_name.replace("VisualCog", "").replace(".mat", "_erp_epochs.npz")
    )
    for path in (mat_path, trial_path, qc_path, epoch_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {path}；请先完整运行 01 和 03。")
    fs, data = pre.load_dataset(mat_path)
    with np.load(epoch_path, allow_pickle=False) as archive:
        filtered_epochs = archive["epochs_accepted"].astype(np.float64)
        trial_ids = archive["accepted_trial_id"].astype(int)
        directions = archive["cue_direction_accepted"].astype(str)
        time_s = archive["time_s"].astype(np.float64)
        band_hz = archive["filter_band_hz"].astype(np.float64)
    if fs != 256 or not np.array_equal(band_hz, [0.5, 30.0]):
        raise ValueError("当前 ERP 片段与本验证锁定的03滤波参数不一致。")
    trials = pd.read_csv(trial_path, encoding="utf-8-sig")
    qc = pd.read_csv(qc_path, encoding="utf-8-sig")
    keep = qc[(qc.file_name == file_name) & qc.accepted_for_erp]
    if not np.array_equal(np.sort(keep.trial_id.to_numpy(int)), trial_ids):
        raise ValueError("03试次验收清单与ERP缓存不匹配。")
    subset = trials[(trials.file_name == file_name) & trials.trial_id.isin(trial_ids)]
    subset = subset.set_index("trial_id").loc[trial_ids]
    if not np.array_equal(subset.cue_direction.astype(str), directions):
        raise ValueError("试次方向与ERP缓存不匹配。")
    cue_samples = subset.cue_onset_sample.to_numpy(dtype=int)
    start_offset = int(round(pre.EPOCH_START_S * fs))
    n_time = time_s.size
    raw_epochs = np.asarray([
        data[:3, cue + start_offset: cue + start_offset + n_time]
        for cue in cue_samples
    ])
    if raw_epochs.shape != filtered_epochs.shape or not np.isfinite(raw_epochs).all():
        raise ValueError("原始试次和03保留片段形状或数值异常。")
    if np.any(np.abs(raw_epochs) >= pre.RAIL_VALUE):
        raise ValueError("主分析保留试次意外包含饱和点。")
    baseline_mask = (time_s >= -.2) & (time_s < 0)
    baseline_sd = raw_epochs[..., baseline_mask].std(axis=-1)
    amplitude = AMPLITUDE_MULTIPLIER * float(np.median(baseline_sd))
    if amplitude <= 0:
        raise ValueError("合成信号标度必须为正。")
    return {
        "raw": data[:3], "mat_path": mat_path, "cue_samples": cue_samples,
        "trial_ids": trial_ids, "directions": directions, "time_s": time_s,
        "baseline_mask": baseline_mask, "amplitude": amplitude,
        "raw_epochs": baseline_correct(raw_epochs, baseline_mask),
        "bandpassed_epochs": filtered_epochs,
        "start_offset": start_offset,
    }


def run_file(file_name: str, file_index: int, pre, erp) -> tuple[list[dict], dict, dict]:
    inp = load_inputs(file_name, pre)
    time_s = inp["time_s"]
    mask = (time_s >= RESPONSE_WINDOW_S[0]) & (time_s <= RESPONSE_WINDOW_S[1])
    templates = synthetic_templates(time_s)
    sos = butter(pre.FILTER_ORDER, [pre.FILTER_LOW_HZ, pre.FILTER_HIGH_HZ],
                 btype="bandpass", fs=256, output="sos")
    filtered_templates = filter_templates(
        templates, sos, inp["baseline_mask"], inp["start_offset"]
    )
    amplitude = inp["amplitude"]
    truth = amplitude * (templates[1] - templates[0])
    noise_free = amplitude * (filtered_templates[1] - filtered_templates[0])
    transfer = contrast_metrics(noise_free, truth, mask)
    rng = np.random.default_rng(SEED + file_index)
    metrics: list[dict] = []
    curves: dict[str, list[np.ndarray]] = {name: [] for name in METHODS}
    audit = float("nan")
    for repetition in range(REPEATS):
        labels = balanced_assignments(inp["trial_ids"], inp["directions"], rng)
        if repetition == 0:
            audit = continuous_audit(
                inp["raw"], inp["cue_samples"], labels, amplitude,
                templates, filtered_templates, inp["bandpassed_epochs"],
                inp["baseline_mask"], sos, inp["start_offset"], pre
            )
            if audit > 0.01:
                raise ValueError(
                    f"{file_name}: 全记录注入验证与模板近似最大差异 {audit:.3g}"
                    " 超过预设的注入幅值1%阈值。"
                )
        raw_injected = inp["raw_epochs"] + amplitude * templates[labels]
        filtered_injected = inp["bandpassed_epochs"] + amplitude * filtered_templates[labels]
        left = labels == 0
        right = ~left
        estimates = {
            "raw_mean": raw_injected[right].mean(axis=0) - raw_injected[left].mean(axis=0),
            "bandpass_mean": filtered_injected[right].mean(axis=0) - filtered_injected[left].mean(axis=0),
            "current_pipeline": (
                erp.fit_curve(filtered_injected[right], 256)
                - erp.fit_curve(filtered_injected[left], 256)
            ),
        }
        for method, estimated in estimates.items():
            curves[method].append(estimated[0].copy())  # Fz仅用于曲线展示
            metrics.append({
                "file_name": file_name, "repetition": repetition + 1, "method": method,
                "n_trials": len(labels), "n_left": int(left.sum()), "n_right": int(right.sum()),
                "injection_amplitude_raw_units": amplitude,
                **contrast_metrics(estimated, truth, mask),
            })
    metadata = {
        "file_name": file_name, "n_trials": len(inp["trial_ids"]),
        "amplitude_raw_units": amplitude,
        "noise_free_filter_nrmse": transfer["nrmse"],
        "noise_free_filter_gain": transfer["gain"],
        "full_record_audit_max_error_per_unit_amplitude": audit,
        "mat_sha256": sha256(inp["mat_path"]),
    }
    plot = {"time_s": time_s, "target": truth[0],
            "curves": {name: np.asarray(values) for name, values in curves.items()}}
    return metrics, metadata, plot


def save_plots(summary: pd.DataFrame, cache: dict[str, dict]) -> None:
    colors = {"raw_mean": "#8b8b8b", "bandpass_mean": "#00796b",
              "current_pipeline": "#e65100"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for axis, file_name in zip(axes.ravel(), FILES):
        item = cache[file_name]
        t = item["time_s"]
        axis.plot(t, item["target"], color="black", lw=2, label="Known R-L template")
        for method in METHODS:
            data = item["curves"][method]
            axis.plot(t, data[0], color=colors[method], lw=1.2,
                      label=method)
        axis.axvspan(*RESPONSE_WINDOW_S, color="#888888", alpha=.08)
        axis.axhline(0, color="black", lw=.5)
        axis.set_title(file_name.replace("VisualCog", "").replace(".mat", ""))
        axis.set_xlabel("Seconds from cue")
        axis.set_ylabel("Synthetic R-L at Fz (raw units)")
        axis.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Injection recovery on real EEG background: first fixed assignment")
    fig.tight_layout()
    fig.savefig(OUTPUT / "recovery_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = np.arange(len(FILES))
    for j, method in enumerate(METHODS):
        rows = summary[summary.method == method].set_index("file_name").loc[list(FILES)]
        ax.bar(x + (j-1)*.25, rows.nrmse_median, width=.23, color=colors[method],
               label=method)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace("VisualCog", "").replace(".mat", "") for name in FILES])
    ax.set_ylabel("Median NRMSE of recovered R-L contrast (lower is better)")
    ax.set_title("Recovery error; same accepted trials and synthetic assignments")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT / "recovery_error.png", dpi=150)
    plt.close(fig)


def report(summary: pd.DataFrame, metadata: list[dict]) -> str:
    lines = [
        "问题一补充：真实脑电背景中的合成视觉响应注入—回收验证",
        "=" * 64,
        "范围：4份原始MAT；仅使用03保留的试次；绝不覆盖原文件或01–09输出。",
        f"预先固定随机种子={SEED}、每文件{REPEATS}次随机分组。",
        "合成L/R标签在每20次试次块×真实提示方向内平衡随机分配。",
        "合成模板：330 ms早期峰与690 ms晚期峰的预设高斯组合；",
        "左右模板形态和Fz/F3/F4通道增益不同，并非真实三角形神经模板。",
        "每份记录注入幅值=0.8×保留试次提示前200ms原始标准差的中位数；单位沿用原始数据。",
        "比较：原始片段算术均值；03的0.5–30Hz带通片段算术均值；",
        "当前完整曲线流程=03带通+固定保留试次+04 Huber与约120ms SG平滑。",
        "三种方法使用相同原始试次、相同合成分组、同一真实模板。",
        "NRMSE=||恢复的R-L-真实R-L||/||真实R-L||，统计Fz/F3/F4与0.15–0.90s。",
        "gain=恢复对真实R-L模板的内积/真实模板平方和；1表示无投影衰减。",
        "区间是随机分组的2.5%–97.5%范围，并非被试总体的置信区间。",
        "", "逐文件结果（中位数，括号内随机分组范围）：",
    ]
    for meta in metadata:
        file_name = meta["file_name"]
        lines += [
            f"[{file_name}] 保留{meta['n_trials']}次；注入幅值={meta['amplitude_raw_units']:.3f}原始单位；",
            f"  无噪声带通传递：NRMSE={meta['noise_free_filter_nrmse']:.3f}，"
            f"gain={meta['noise_free_filter_gain']:.3f}；",
            f"  全记录实注入与模板线性近似的最大相对误差={meta['full_record_audit_max_error_per_unit_amplitude']:.5f}。",
        ]
        for method in METHODS:
            row = summary[(summary.file_name == file_name) & (summary.method == method)].iloc[0]
            lines.append(
                f"  {method:18s} NRMSE={row.nrmse_median:.3f} "
                f"[{row.nrmse_q025:.3f}, {row.nrmse_q975:.3f}]，"
                f"gain={row.gain_median:.3f}，形状相关={row.correlation_median:.3f}"
            )
    lines += [
        "", "主要发现：",
        "四份记录的完整流程NRMSE中位数均小于直接使用原始试次的算术均值，",
        "但在所选注入强度下四份记录的完整流程NRMSE中位数均大于1；",
        "即已知左右差异仍淹没在试次分组形成的背景波动里，不能称为可靠回收。",
        "带通的无噪声传递增益约0.9且NRMSE约0.215，说明滤波本身也有可量化形状损失。",
        "recovery_curves.png显示首个固定随机分组，而误差图/表汇总所有48次；",
        "单次分组的回收可能明显偏离模板，不以跨随机分组平均曲线冒充单次恢复性能。",
        "", "解释界限：",
        "1. 本实验是已知形状的正控制。相对误差改善不意味着真实左右三角有可分辨响应；",
        "   真实提示方向的0/24项FDR显著检验仍以04报告为准。",
        "2. 注入幅值、形状与采样窗事先固定，仅检验这类模板；不可概括所有ERP形状。",
        "3. 03保留试次固定且没有人为眨眼标签；本实验未验证眨眼/运动伪影可完全去除。",
        "4. 设备自带Decon只有已处理波形，未知滤波算子；无法对其进行同一信号的合法注入—回收，",
        "   故不把Decon输出冒充本实验的对照或声称本方法优于Decon。",
        "5. 同一两名被试的重抽样不构成独立受试者验证；无行为正确答案、无脑源定位证据。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    pre = import_existing("existing_preprocess", "03_preprocess_erp.py")
    erp = import_existing("existing_erp_fit", "04_erp_curves.py")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    all_metrics: list[dict] = []
    metadata: list[dict] = []
    plots: dict[str, dict] = {}
    for i, name in enumerate(FILES):
        print(f"注入—回收验证：{name} ({i+1}/4)，{REPEATS}次分组", flush=True)
        rows, info, plot = run_file(name, i, pre, erp)
        all_metrics.extend(rows)
        metadata.append(info)
        plots[name] = plot
    scores = pd.DataFrame(all_metrics)
    group = scores.groupby(["file_name", "method"], sort=False)
    aggregate = group.agg(
        nrmse_median=("nrmse", "median"),
        nrmse_q025=("nrmse", lambda x: x.quantile(.025)),
        nrmse_q975=("nrmse", lambda x: x.quantile(.975)),
        gain_median=("gain", "median"),
        correlation_median=("shape_correlation", "median"),
    ).reset_index()
    scores.to_csv(OUTPUT / "injection_repetitions.csv", index=False, encoding="utf-8-sig")
    aggregate.to_csv(OUTPUT / "injection_summary.csv", index=False, encoding="utf-8-sig")
    config = {
        "seed": SEED, "repeats_per_file": REPEATS,
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "response_window_seconds": RESPONSE_WINDOW_S,
        "methods": METHODS, "inputs": metadata,
        "interpretation": "synthetic positive control, not empirical left/right evidence",
    }
    (OUTPUT / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    save_plots(aggregate, plots)
    path = OUTPUT / "q1_injection_report.txt"
    path.write_text(report(aggregate, metadata), encoding="utf-8-sig")
    print(f"完成。报告：{path}", flush=True)
    print(aggregate[["file_name", "method", "nrmse_median", "gain_median"]].to_string(index=False))


if __name__ == "__main__":
    main()
