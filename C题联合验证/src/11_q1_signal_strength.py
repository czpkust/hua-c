r"""问题一注入回收的信号强度诊断，依赖已经安装的10号补丁。

从项目根目录运行：python -X utf8 .\src\11_q1_signal_strength.py
仅输出到 results/11_q1_signal_strength；不改写01—10的代码或结果。
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "11_q1_signal_strength"
FACTORS = (0.4, 0.8, 1.6, 3.2)


def load_benchmark():
    path = Path(__file__).with_name("10_q1_injection_recovery.py")
    if not path.is_file():
        raise FileNotFoundError("请先将10号补丁安装到同一个src目录。")
    spec = importlib.util.spec_from_file_location("q1_injection_benchmark", path)
    if spec is None or spec.loader is None:
        raise ImportError(str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def zero_background_reference(base, pre, erp) -> pd.DataFrame:
    t = np.arange(-128, 256, dtype=float) / 256
    baseline = (t >= -.2) & (t < 0)
    mask = (t >= base.RESPONSE_WINDOW_S[0]) & (t <= base.RESPONSE_WINDOW_S[1])
    templates = base.synthetic_templates(t)
    sos = butter(pre.FILTER_ORDER, [pre.FILTER_LOW_HZ, pre.FILTER_HIGH_HZ],
                 btype="bandpass", fs=256, output="sos")
    filtered = base.filter_templates(templates, sos, baseline, -128)
    fitted = np.asarray([
        erp.fit_curve(np.repeat(template[None, :, :], 30, axis=0), 256)
        for template in filtered
    ])
    truth = templates[1] - templates[0]
    curves = {"raw_mean": truth, "bandpass_mean": filtered[1] - filtered[0],
              "current_pipeline": fitted[1] - fitted[0]}
    return pd.DataFrame([
        {"method": method, **base.contrast_metrics(curve, truth, mask)}
        for method, curve in curves.items()
    ])


def save_plot(summary: pd.DataFrame, clean: pd.DataFrame, files: tuple[str, ...]) -> None:
    colors = {"raw_mean": "#888888", "bandpass_mean": "#00796b", "current_pipeline": "#e65100"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    reference = float(clean.set_index("method").loc["current_pipeline", "nrmse"])
    for ax, filename in zip(axes.ravel(), files):
        for method, color in colors.items():
            rows = summary[(summary.file_name == filename) & (summary.method == method)].sort_values("amplitude_multiplier")
            x = rows.amplitude_multiplier.to_numpy()
            ax.plot(x, rows.nrmse_median, "o-", color=color, label=method)
            if method == "current_pipeline":
                ax.fill_between(x, rows.nrmse_q025, rows.nrmse_q975, color=color, alpha=.15,
                                label="Current: random-group 2.5-97.5%")
        ax.axhline(1, color="#444444", ls=":", lw=1, label="NRMSE = 1")
        ax.axhline(reference, color="#2a5599", ls="--", lw=1, label="Current: no background")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(FACTORS, [str(v) for v in FACTORS])
        ax.set_title(filename.replace("VisualCog", "").replace(".mat", ""))
        ax.set_xlabel("Injected amplitude / median pre-cue SD")
        ax.set_ylabel("NRMSE of synthetic R-L contrast")
        ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle("Synthetic strength sensitivity: fixed preprocessing and paired random assignments")
    fig.tight_layout(rect=(0, .085, 1, .96))
    fig.savefig(OUT / "signal_strength_curve.png", dpi=160)
    plt.close(fig)


def build_report(summary: pd.DataFrame, clean: pd.DataFrame, base, matched: bool | None) -> str:
    lines = [
        "问题一补充诊断：信号强度与无背景失真",
        "=" * 64,
        "目的：量化已知合成左右差异的恢复对信号强度的依赖；不改变预处理参数。",
        "本实验由10号结果启发，是同一数据上的探索性诊断，不是独立数据验证。",
        f"注入幅值倍数固定为{FACTORS}；每文件每强度{base.REPEATS}次分组，种子{base.SEED}。",
        "四种强度使用同一批保留试次、相同合成模板、相同随机分组；所有设置完整报告。",
        "幅值单位：各记录保留试次提示前200ms原始标准差的中位数。",
        "当前流程=03带通+固定试次保留+04 Huber与SG；参数未重新选择。",
        "", "一、去掉全部真实背景，仅处理已知模板时的形状变化：",
    ]
    for row in clean.itertuples(index=False):
        lines.append(f"  {row.method:18s} NRMSE={row.nrmse:.4f}，模板投影gain={row.gain:.4f}，相关={row.shape_correlation:.4f}")
    lines += ["无背景误差是模板在算法中的形状改变，不能视为所有数据误差的严格下界。",
              "", "二、加入真实脑电背景后的当前流程："]
    for filename in base.FILES:
        lines.append(f"[{filename}]")
        rows = summary[(summary.file_name == filename) & (summary.method == "current_pipeline")].sort_values("amplitude_multiplier")
        for row in rows.itertuples(index=False):
            lines.append(
                f"  强度{row.amplitude_multiplier:3.1f}×：NRMSE中位数={row.nrmse_median:.3f} "
                f"[{row.nrmse_q025:.3f}, {row.nrmse_q975:.3f}]，"
                f"gain={row.gain_median:.3f}，相关={row.correlation_median:.3f}"
            )
    low = summary[(summary.method == "current_pipeline") & (summary.amplitude_multiplier == FACTORS[0])].set_index("file_name")
    high = summary[(summary.method == "current_pipeline") & (summary.amplitude_multiplier == FACTORS[-1])].set_index("file_name")
    n_drop = int((high.nrmse_median < low.nrmse_median).sum())
    n_under_one = int((high.nrmse_median < 1).sum())
    lines += [
        "", "三、诊断解释：",
        f"从{FACTORS[0]}×提高到{FACTORS[-1]}×，当前流程误差中位数下降的记录为{n_drop}/4；",
        f"在{FACTORS[-1]}×强度下误差中位数小于1的记录为{n_under_one}/4。",
        "NRMSE随强度明显下降说明真实背景相对目标信号的大小会限制回收；",
        "无背景误差反映另一个因素——滤波和平滑本身对这类模板的失真。",
        "本实验没有证明唯一误差来源，也没有确定真实视觉响应具有哪种强度或形状。",
        "NRMSE=1只表示估计误差平方和等于模板平方和；它不是显著性、临床或普适合格阈值。",
        "gain含有背景投影；例如gain=0.8不能直接解释成真实神经特征保留了80%。",
        "提高合成幅值只是诊断，不是改善真实数据的方法；不能以最好强度替代10号主结果。",
        "区间是48次随机分组的分位范围，不是被试总体置信区间。",
        "真实左右标签结果仍以04、06、08为准，不能据此改写成可可靠区分左右三角。",
        "", "四、复现核对：",
        "PASS  0.8×结果与已有10号汇总一致。" if matched is True else
        "未核对：没有10号汇总表。" if matched is None else "FAIL  0.8×结果与10号不一致。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    base = load_benchmark()
    pre = base.import_existing("existing_preprocess", "03_preprocess_erp.py")
    erp = base.import_existing("existing_erp_fit", "04_erp_curves.py")
    OUT.mkdir(parents=True, exist_ok=True)
    clean = zero_background_reference(base, pre, erp)
    scores = []
    metadata = []
    for factor in FACTORS:
        base.AMPLITUDE_MULTIPLIER = factor
        print(f"合成强度 {factor:.1f}×：", flush=True)
        for index, filename in enumerate(base.FILES):
            rows, info, _ = base.run_file(filename, index, pre, erp)
            for row in rows:
                row["amplitude_multiplier"] = factor
            scores.extend(rows)
            metadata.append({"amplitude_multiplier": factor, **info})
            current = [r["nrmse"] for r in rows if r["method"] == "current_pipeline"]
            print(f"  {filename}: 当前流程NRMSE中位数={np.median(current):.3f}", flush=True)
    scores = pd.DataFrame(scores)
    summary = scores.groupby(["file_name", "amplitude_multiplier", "method"], sort=False).agg(
        nrmse_median=("nrmse", "median"),
        nrmse_q025=("nrmse", lambda x: x.quantile(.025)),
        nrmse_q975=("nrmse", lambda x: x.quantile(.975)),
        gain_median=("gain", "median"), correlation_median=("shape_correlation", "median"),
    ).reset_index()
    old_path = ROOT / "results" / "10_q1_injection_recovery" / "injection_summary.csv"
    matched = None
    if old_path.is_file():
        old = pd.read_csv(old_path).set_index(["file_name", "method"]).sort_index()
        current = summary[summary.amplitude_multiplier == .8].set_index(["file_name", "method"]).sort_index()
        columns = ["nrmse_median", "nrmse_q025", "nrmse_q975", "gain_median", "correlation_median"]
        matched = bool(old.index.equals(current.index) and np.allclose(old[columns], current[columns], rtol=1e-7, atol=1e-8))
        if not matched:
            raise ValueError("0.8×与现有10号结果不一致，请检查脚本版本和输入；不继续生成正式报告。")
    scores.to_csv(OUT / "strength_repetitions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "strength_summary.csv", index=False, encoding="utf-8-sig")
    clean.to_csv(OUT / "zero_background_distortion.csv", index=False, encoding="utf-8-sig")
    config = {
        "factors": FACTORS, "seed": base.SEED, "repeats": base.REPEATS,
        "matched_step_10": matched,
        "dependency_hashes": {name: base.sha256(ROOT / "src" / name) for name in (
            "03_preprocess_erp.py", "04_erp_curves.py", "10_q1_injection_recovery.py")},
        "inputs": metadata,
    }
    (OUT / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    save_plot(summary, clean, base.FILES)
    path = OUT / "q1_signal_strength_report.txt"
    path.write_text(build_report(summary, clean, base, matched), encoding="utf-8-sig")
    print(f"\n完成。中文报告：{path}", flush=True)
    print(summary[summary.method == "current_pipeline"].pivot(index="file_name", columns="amplitude_multiplier", values="nrmse_median").round(3).to_string())


if __name__ == "__main__":
    main()
