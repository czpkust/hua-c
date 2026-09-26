"""第三问：将08固定的四阶段视觉响应连接到目标后的候选记忆比较状态。

只分析有真实±2点击脉冲的Task-2。每个试验截止于点击前0.1秒，
滤波也只能看到截止点以前的样本；Task-1没有独立点击标记，不能伪造应答。
三个头皮电极不足以定位LGN或海马，本脚本验证的是功能状态模型及其预测，
不能把候选状态拟合系数当成真实海马活动或行为正确率。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


FS = 256
TRIALS = ("A_Task-2", "B_Task-2")
MODELS = ("trend", "q2_visual", "q2_plus_target", "q3_comparison")
CUTOFF_S = 0.1
DECIMATE = 4
RIDGE = 5.0
SEED = 20260924


def load_q2_module():
    """直接复用08的窗口、分块和动力学设计，不另行调参。"""
    path = Path(__file__).with_name("08_q2_mechanism_validation.py")
    spec = importlib.util.spec_from_file_location("c_question_q2_validation", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def cascade(x: np.ndarray, tau: float, order: int) -> np.ndarray:
    a = float(np.exp(-1 / (FS * tau)))
    for _ in range(order):
        x = lfilter([1 - a], [1, -a], x)
    return x


def fixed_stages(n: int, onset_sample: int) -> np.ndarray:
    """与08名义设置完全一致的逐级稳定滤波器。"""
    x = np.zeros(n, dtype=float)
    if onset_sample < n:
        x[onset_sample] = FS
    stages = []
    for tau, order in zip((.015, .035, .050, .150), (2, 3, 4, 3)):
        x = cascade(x, tau, order)
        stages.append(x / max(np.max(np.abs(x)), 1e-12))
    return np.column_stack(stages)


def design_for_trial(time: np.ndarray, target_offset: int, cue_sign: int,
                     target_sign: int, q2_stages: np.ndarray) -> np.ndarray:
    target_stages = fixed_stages(len(time), target_offset + round(.035 * FS))
    # 最后的两个候选状态分别是目标后的晚期情境活动、提示与目标编码的比较。
    # 符号乘积是实验标记的关系，不是独立证实的“识别正确性”。
    rel = cue_sign * target_sign
    trend = np.column_stack((np.ones(len(time)), time / 2.0))
    return np.column_stack((trend, q2_stages, target_stages[:, 1:3],
                            target_stages[:, 3], rel * target_stages[:, 3]))


def prepare_file(root: Path, name: str, q2_record: dict, q2) -> tuple[list[dict], dict]:
    raw_path = root / "data" / "raw" / f"VisualCog{name}.mat"
    if not raw_path.is_file():
        raise FileNotFoundError(f"缺少原始数据：{raw_path}")
    csv_path = root / "data" / "processed" / "trial_info.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"缺少逐试验事件表：{csv_path}；先运行01。")
    events = pd.read_csv(csv_path)
    events = events[events.file_name == raw_path.name].set_index("trial_id")
    if set(q2_record["ids"]) - set(events.index):
        raise ValueError(f"{name}：01事件表与03保留试验编号不一致。")
    mat = loadmat(raw_path)
    if int(mat["SampleRate"].item()) != FS:
        raise ValueError(f"{name}：预期256 Hz。")
    raw = np.asarray(mat["data"][:3], dtype=float)
    # 与03相同的滤波器，但在每个试验的截止点结束滤波，绝不输入点击以后数据。
    sos = butter(4, [.5, 30], btype="bandpass", fs=FS, output="sos")
    long_trials = []
    excluded = []
    for i, trial_id in enumerate(q2_record["ids"]):
        row = events.loc[trial_id]
        cue = int(row.cue_onset_sample)
        target = int(row.target_onset_sample)
        click = int(row.response_sample)
        if not (cue < target < click) or int(row.target_value) not in (-1, 1) or int(row.response_value) not in (-2, 2):
            excluded.append((int(trial_id), "missing_or_disordered_event"))
            continue
        start, stop = cue - round(.5 * FS), click - round(CUTOFF_S * FS)
        if start < 0 or stop > raw.shape[1] or stop <= target + round(.85 * FS):
            excluded.append((int(trial_id), "bounds_or_short_recognition_window"))
            continue
        prefix = raw[:, start:stop]
        if not np.isfinite(prefix).all() or np.any(np.abs(prefix) >= 999.999):
            excluded.append((int(trial_id), "additional_artifact_in_long_window"))
            continue
        filtered = sosfiltfilt(sos, prefix, axis=1)
        baseline = filtered[:, round(.3 * FS):round(.5 * FS)].mean(axis=1)
        filtered -= baseline[:, None]
        rel_time = (np.arange(start, stop) - cue) / FS
        take = np.flatnonzero(rel_time >= 0)[::DECIMATE]
        t = rel_time[take]
        visual = fixed_stages(len(rel_time), round((cue-start+.035 * FS)))
        # 这里的“目标偏移”相对于整个滤波片段（起于cue前0.5秒）。
        x = design_for_trial(rel_time, target-start, int(row.cue_value),
                             int(row.target_value), visual)[take]
        y = filtered[:, take].T
        assert np.isfinite(y).all() and np.isfinite(x).all()
        stage_time = q2_record["time"]
        if len(long_trials) == 0:
            nominal = q2.build_design(stage_time, FS, q2.Setting("nominal"))
            # 以08的原始固定设计矩阵检验形状，而非给Q3重新估计时间常数。
            check = fixed_stages(len(stage_time), int(np.argmin(np.abs(stage_time-.035))))
            if not np.allclose(check[nominal["mask"]], nominal["full"][:, :4], atol=1e-12):
                raise ValueError("09与08的四阶段视觉基函数不一致。")
        long_trials.append(dict(file=name, trial_id=int(trial_id), time=t, design=x,
                                eeg=y, target_s=(target-cue)/FS,
                                cutoff_s=(stop-cue)/FS, cue=int(row.cue_value),
                                target=int(row.target_value), click=int(np.sign(row.response_value)),
                                q2_features=None))
    return long_trials, dict(file=name, q2_accepted=len(q2_record["ids"]),
                             long_window_accepted=len(long_trials), exclusions=excluded)


def solve_train(trials: list[dict], indices: np.ndarray, model: str) -> np.ndarray:
    p = {"trend": 2, "q2_visual": 6, "q2_plus_target": 8, "q3_comparison": 10}[model]
    gram = np.zeros((p, p))
    xy = np.zeros((p, 3))
    for k in indices:
        x, y = trials[k]["design"][:, :p], trials[k]["eeg"]
        gram += x.T @ x
        xy += x.T @ y
    penalty = np.diag([0., 0.] + [RIDGE] * (p - 2))
    return np.linalg.solve(gram + penalty, xy)


def blocked_fit(trials: list[dict], record: dict, q2) -> tuple[pd.DataFrame, dict]:
    splits, _ = q2.make_block_splits(record)
    available = {item["trial_id"]: index for index, item in enumerate(trials)}
    rows, examples = [], {}
    for fold, (orig_train, orig_test, _) in enumerate(splits, 1):
        tr = np.asarray([available[int(record["ids"][i])] for i in orig_train
                         if int(record["ids"][i]) in available], dtype=int)
        te = np.asarray([available[int(record["ids"][i])] for i in orig_test
                         if int(record["ids"][i]) in available], dtype=int)
        if len(tr) < 30 or len(te) < 5:
            raise ValueError(f"{record['name']}第{fold}折清洁试验不足。")
        models = {model: solve_train(trials, tr, model) for model in MODELS}
        for ix in te:
            r = trials[ix]
            for region in ("visual", "recognition", "all_pre_click"):
                if region == "visual":
                    mask = (.05 <= r["time"]) & (r["time"] <= .90)
                elif region == "recognition":
                    mask = (r["time"] >= r["target_s"] + .05)
                else:
                    mask = r["time"] >= .05
                x, obs = r["design"][mask], r["eeg"][mask]
                if not len(obs):
                    raise ValueError("响应前观测窗口为空。")
                for model in MODELS:
                    p = models[model].shape[0]
                    err = obs - x[:, :p] @ models[model]
                    rows.append(dict(file=record["name"], fold=fold, trial_id=r["trial_id"],
                                     region=region, model=model, n_values=err.size,
                                     sse=float(np.sum(err**2)), mse=float(np.mean(err**2)),
                                     cutoff_s=r["cutoff_s"],
                                     marker_relation=r["cue"] * r["target"]))
            if fold == 5 and len(examples) < 3:
                examples[r["trial_id"]] = dict(t=r["time"], observed=r["eeg"][:, 0],
                    visual=r["design"][:, :6] @ models["q2_visual"][:, 0],
                    full=r["design"] @ models["q3_comparison"][:, 0],
                    target=r["target_s"], cutoff=r["cutoff_s"])
    result = pd.DataFrame(rows)
    if result[result.model == "trend"].groupby("trial_id").size().ne(3).any():
        raise ValueError("每个保留试验没有恰好一次作为测试。")
    return result, examples


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (name, region), g in metrics.groupby(["file", "region"], sort=False):
        baseline = g[g.model == "trend"].sse.sum()
        for model, h in g.groupby("model", sort=False):
            sse = float(h.sse.sum())
            rows.append(dict(file=name, region=region, model=model, n_trials=len(h),
                             rmse=float(np.sqrt(sse/h.n_values.sum())),
                             r2_vs_train_trend=float(1-sse/baseline),
                             sse=sse, n_values=int(h.n_values.sum())))
    return pd.DataFrame(rows)


def plot_results(summary: pd.DataFrame, examples: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    sub = summary[summary.region == "recognition"]
    for j, model in enumerate(MODELS):
        v = sub[sub.model == model].set_index("file").loc[list(TRIALS), "rmse"]
        ax.bar(np.arange(2)+(j-1.5)*.2, v, .19, label=model)
    ax.set(xticks=np.arange(2), xticklabels=TRIALS,
           ylabel="Held-out pre-click recognition RMSE (data units)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(out/"heldout_cognitive_comparison.png", dpi=180)
    plt.close(fig)
    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(10, 3*len(examples)),
                                 layout="constrained", squeeze=False)
        for ax, (trial, x) in zip(axes.ravel(), examples.items()):
            ax.plot(x["t"], x["observed"], color="#777777", lw=.7, label="Fz observed")
            ax.plot(x["t"], x["visual"], label="Q2 visual")
            ax.plot(x["t"], x["full"], label="Q3 comparison candidate")
            ax.axvline(x["target"], color="black", lw=.7, ls="--")
            ax.set(title=f"B Task-2 held-out trial {trial}; target at dashed line",
                   xlim=(0, x["cutoff"]), ylabel="Signal (data units)")
            ax.legend(fontsize=8, ncol=3)
        axes[-1, 0].set_xlabel("Seconds after visual cue (stops 100 ms before click)")
        fig.savefig(out/"heldout_preclick_examples.png", dpi=180)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="C题问题三：问题二四阶段模型联动的留出验证")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = root / "results" / "09_q3_cognitive_joint_validation"
    out.mkdir(parents=True, exist_ok=True)
    q2 = load_q2_module()
    records = {r["name"]: r for r in q2.load_data(root)}
    q2_features_path = root / "results" / "08_q2_mechanism_validation" / "mechanism_12d_features.csv"
    if not q2_features_path.is_file():
        raise FileNotFoundError(f"缺少第二问08特征表：{q2_features_path}。请先运行08。")
    q2_features = pd.read_csv(q2_features_path)
    metrics_list, eligibility, examples = [], [], {}
    with threadpool_limits(limits=1):
        for name in TRIALS:
            r = records[name]
            features = q2_features[q2_features.file == name]
            if not np.array_equal(features.trial_id.to_numpy(), r["ids"]):
                raise ValueError(f"{name}：08特征与03保留试验编号不一致。")
            if not np.array_equal(features.direction.to_numpy(), r["directions"]):
                raise ValueError(f"{name}：08方向标签与03原始试验不一致。")
            trial_data, eligibility_row = prepare_file(root, name, r, q2)
            eligibility.append(eligibility_row)
            scores, curves = blocked_fit(trial_data, r, q2)
            metrics_list.append(scores)
            if name == "B_Task-2":
                examples.update(curves)
            print(f"{name}: 第一问清洁 {len(r['ids'])}，截止点前无饱和 {len(trial_data)}。", flush=True)
    metrics = pd.concat(metrics_list, ignore_index=True)
    summary = summarize(metrics)
    metrics.to_csv(out/"heldout_trial_errors.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(out/"heldout_model_comparison.csv", index=False, encoding="utf-8-sig")
    (out/"eligibility_and_parameters.json").write_text(json.dumps(dict(
        task1="No verified ±2 click: cannot set 100 ms pre-response cutoff",
        task2=eligibility, window_end="click - 0.1 s", fs=FS, ridge=RIDGE,
        reference="08 nominal four-stage basis and 08 same five chronological 20-trial blocks, ±2 trial purge",
        filter="within-trial prefix through cutoff only: 0.5-30 Hz order-4 zero-phase",
        marker_relation="cue sign × Task-2 channel-9 ±1 sign; behavioral accuracy unknown",
        basis="Q2 visual [four states] + target-locked two stages + candidate late context and cue-target relation",
        interpretation="functional proxy only; LGN/hippocampus locations not identifiable from F3/Fz/F4",
    ), ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    plot_results(summary, examples, out)
    lines = ["问题三：与第二问固定时间基函数一致的候选认知模型（留出验证）", "="*65,
        "输入：01事件表、03清洁试次、08的12维特征试次编号，以及四份原始MAT。",
        "模型：提示→08四阶段视觉状态；目标事件→固定时间基函数的目标响应；",
        "候选记忆对照：目标后晚期状态×(提示符号×目标标记符号)，只表示假设中的功能比较。",
        "观测：各状态通过可训练的有效增益投影到Fz/F3/F4；仅训练块估计增益。",
        "视觉、目标、情境状态时间常数不从测试试次重新优化。",
        "验证：同08每20次原试次留一个块，训练额外排除相邻各2次；每试次恰好测试一次。",
        "预处理只读取点击前0.1秒截止点以前原始信号；另剔除长窗硬饱和。",
        "识别窗口：目标标记出现后0.05秒至实际±2点击前0.1秒；观察是离线对齐。",
        "SSE分母为训练数据拟合的截距加线性趋势基线；留出R²可以为负。", ""]
    for name in TRIALS:
        e = next(z for z in eligibility if z["file"] == name)
        lines.append(f"{name}: 03保留{e['q2_accepted']}；额外长窗排除{len(e['exclusions'])}；第三问{e['long_window_accepted']}试次。")
        g = summary[(summary.file == name) & (summary.region == "recognition")].set_index("model")
        for model in MODELS:
            v = g.loc[model]
            lines.append(f"  {model:16s}: 留出RMSE={v.rmse:.4f}, 对训练趋势基线R²={v.r2_vs_train_trend:+.4f}")
        base, m2, m3 = [g.loc[k, "sse"] for k in ("q2_visual", "q2_plus_target", "q3_comparison")]
        lines.append(f"  目标输入相对视觉模型SSE变化={100*(base-m2)/base:+.2f}%；比较候选相对目标输入SSE变化={100*(m2-m3)/m2:+.2f}%。")
    lines.extend(["", "边界与判定：", "Task-1无独立点击标记，无法验证响应前100 ms和其行为正确性。",
        "Task-2的±1与±2同号属于记录编码一致性，并非独立核验的行为正确率。",
        "切片停止在点击前100 ms，运动准备仍可能出现在停止点以前。",
        "三电极无法单独定位海马或LGN；拟合改进也不能证明其生理来源或临床诊断效力。",
        "分析参数与排除规则均在同一数据集上制定，是探索性回顾验证，不是新被试盲测。",
        "第二问的提示左右分类及跨被试结果参阅08报告；两问共享固定视觉状态，并不意味着已能可靠区分三角形。"])
    (out/"q3_cognitive_joint_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("第三问联动验证完成：", out/"q3_cognitive_joint_report.txt", flush=True)


if __name__ == "__main__":
    main()
