r"""第三问：连接12的固定视觉网络，验证记忆维持/目标后再激活候选状态。

python -X utf8 .\src\13_q3_shape_memory_validation.py
依赖01事件表、03保留试次、08/09/12脚本、12模拟输出及原始Task-2 MAT。
只写results/13_q3_shape_memory_validation；不推断行为正确率或真实海马来源。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import butter, lfilter, sosfiltfilt
from threadpoolctl import threadpool_limits


FS = 256
FILES = ("A_Task-2", "B_Task-2")
CHANNELS = ("Fz", "F3", "F4")
COMPONENTS = ("intercept", "linear_trend", "cue_left_outer", "cue_left_center",
              "cue_middle_outer", "cue_middle_center", "cue_right_outer", "cue_right_center",
              "target_relay", "target_readout", "memory_L", "memory_R", "reactivation_L", "reactivation_R")
MODELS = {"trend": ("nominal", 2), "shape_visual": ("nominal", 8),
          "shape_target": ("nominal", 10), "memory_only": ("nominal", 12),
          "shape_memory": ("nominal", 14), "memory_blind": ("memory_blind", 14),
          "short_memory": ("short_memory", 14)}
REGIONS = ("visual", "delay", "recognition", "all_pre_click")


@dataclass(frozen=True)
class Parameters:
    memory_tau_s: float = 1.5
    reactivation_tau_s: float = .25
    short_memory_tau_s: float = .15
    ridge: float = .1
    cutoff_before_click_s: float = .1
    decimate: int = 4
    target_probe_s: float = 3.0


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(filename, name):
    path = Path(__file__).with_name(filename)
    require(path.is_file(), f"缺少前置脚本：{path}")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_visual(root, q12, p):
    folder = root/"results"/"12_q2_shape_driven_model"
    config_path, sim_path = folder/"run_config.json", folder/"simulation_shape_wc.npz"
    require(config_path.is_file() and sim_path.is_file(), "请先运行12_q2_shape_driven_model.py。")
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    require(config["script_sha256"] == digest(q12.__file__), "12脚本与其模拟输出版本不一致；请重新运行12。")
    fixed = json.loads(json.dumps(asdict(q12.Parameters())))
    require(config["parameters"] == fixed, "12模拟参数不是当前脚本的固定主设置。")
    with np.load(sim_path, allow_pickle=False) as a:
        t, states, source = a["time_s"].copy(), a["state"].copy(), a["source"].copy()
    require(states.shape == (2, len(t), 120) and source.shape == (2, len(t), 6), "12状态维度异常。")
    require(np.isfinite(states).all() and np.isfinite(source).all() and np.allclose(np.diff(t), 1/FS), "12状态或时间轴异常。")
    spatial = q12.spatial_inputs(q12.Parameters())
    prototypes = spatial["features"]/spatial["features"].sum(axis=1, keepdims=True)
    above_rest = np.maximum(states[:, :, 36:72]-states[:, :1, 36:72], 0)
    encoding = above_rest@prototypes.T
    encoding /= encoding.max()  # 只用固定模拟，不用EEG。
    # 目标显示期间输入持续存在：仅延长外部输入，神经网络参数完全复用12。
    # 对两种图形的响应取平均作为不带目标方向信息的候选探测输入。
    target_sim = q12.simulate(spatial, replace(q12.Parameters(), pulse_s=p.target_probe_s))
    relay = target_sim["states"][:, :, :36].mean(axis=(0, 2))
    relay /= relay.max()
    readout = target_sim["source"].mean(axis=(0, 2))
    readout /= np.max(np.abs(readout))
    require(np.allclose(encoding[0], encoding[1, :, ::-1], atol=1e-12), "记忆输入不满足镜像约束。")
    require(np.linalg.norm(encoding[0]-encoding[1]) > 1e-4, "候选编码没有图形差异。")
    return dict(time=t, source=source, encoding=encoding, target_relay=relay,
                target_readout=readout, source_file=sim_path, config_file=config_path,
                parent_script_sha256=config["script_sha256"])


def interpolate(t, source_t, values):
    if values.ndim == 1:
        return np.interp(t, source_t, values, left=0, right=0)
    return np.column_stack([np.interp(t, source_t, col, left=0, right=0) for col in values.T])


def lowpass(values, tau):
    a = np.exp(-1/(FS*tau))
    return lfilter([1-a], [1, -a], values, axis=0)


def observation_filter(values):
    sos = butter(4, [.5, 30], btype="bandpass", fs=FS, output="sos")
    filtered = sosfiltfilt(sos, values, axis=0)
    return filtered-filtered[round(.3*FS):round(.5*FS)].mean(axis=0, keepdims=True)


def build_design(time_after_cue, target_s, cutoff_s, cue_sign, visual, p,
                 memory_tau=None, blind=False):
    """仅接收提示方向及事件时刻；不接收目标符号、点击方向或正确性标签。"""
    n = int(round((cutoff_s+.5)*FS))
    t = -.5+np.arange(n)/FS
    direction = int(cue_sign == 1)
    require(t[-1]-target_s <= visual["time"][-1], "目标后窗口超出固定模拟长度。")
    cue = interpolate(t, visual["time"], visual["source"][direction])
    target_g = interpolate(t-target_s, visual["time"], visual["target_relay"])
    target_v = interpolate(t-target_s, visual["time"], visual["target_readout"])
    u = interpolate(t, visual["time"], visual["encoding"][direction])
    if blind:
        u[:] = u.mean(axis=1, keepdims=True)
    memory = lowpass(u, p.memory_tau_s if memory_tau is None else memory_tau)
    reactivation = lowpass(memory*target_g[:, None], p.reactivation_tau_s)
    # 所有候选观测分量均接受与本试次EEG相同的、截止点内的滤波/基线处理。
    raw = np.column_stack([cue, target_g, target_v, memory, reactivation])
    filtered = observation_filter(raw)
    take = np.flatnonzero(t >= 0)[::p.decimate]
    require(np.array_equal(t[take], time_after_cue), "13与09截取或抽样时间不一致。")
    trend = np.column_stack([np.ones(len(take)), time_after_cue/2])
    x = np.column_stack([trend, filtered[take]])
    require(np.isfinite(x).all(), "候选状态含NaN/Inf。")
    return x, dict(time=t, encoding=u, memory=memory, reactivation=reactivation, gate=target_g)


def prepare(root, records, q08, q09, visual, p):
    all_trials, eligibility, checks = [], [], []
    require(q09.CUTOFF_S == p.cutoff_before_click_s and q09.DECIMATE == p.decimate, "09截取设置不是既定100ms/64Hz。")
    for name in FILES:
        trials, info = q09.prepare_file(root, name, records[name], q08)
        eligibility.append(info)
        for trial in trials:
            trial["designs"] = {}
            for variant, tau, blind in (("nominal", p.memory_tau_s, False),
                                        ("memory_blind", p.memory_tau_s, True),
                                        ("short_memory", p.short_memory_tau_s, False),
                                        ("memory_tau_x0.8", .8*p.memory_tau_s, False),
                                        ("memory_tau_x1.2", 1.2*p.memory_tau_s, False)):
                x, state = build_design(trial["time"], trial["target_s"], trial["cutoff_s"],
                                         trial["cue"], visual, p, tau, blind)
                trial["designs"][variant] = x
                if variant == "nominal":
                    trial["state"] = state
            nominal = trial["designs"]["nominal"]
            for x in trial["designs"].values():
                require(np.array_equal(x[:, :10], nominal[:, :10]), "认知消融改变了视觉/目标基线列。")
            require(np.allclose(trial["designs"]["memory_blind"][:, 10], trial["designs"]["memory_blind"][:, 11]), "去方向记忆仍有方向差异。")
            # 确认再激活在原始状态空间中只发生于目标出现之后。
            st = trial["state"]
            require(np.max(np.abs(st["reactivation"][st["time"] < trial["target_s"]])) == 0, "目标前已有再激活输入。")
        require(len(trials) > 0, f"{name}没有可用应答前试次。")
        all_trials.extend(trials)
        checks.append(dict(check=f"{name}_state_and_window_alignment", passed=True, value=len(trials)))
        print(f"  {name}：03保留{info['q2_accepted']}，应答前长窗保留{len(trials)}。", flush=True)
    return all_trials, eligibility, checks


def make_evaluations(trials, records, q08):
    evaluations, manifest = [], []
    for name in FILES:
        available = {tr["trial_id"]: k for k, tr in enumerate(trials) if tr["file"] == name}
        local, _ = q08.make_block_splits(records[name])
        splits = []
        for a, b, label in local:
            train = np.array([available[int(records[name]["ids"][i])] for i in a if int(records[name]["ids"][i]) in available])
            test = np.array([available[int(records[name]["ids"][i])] for i in b if int(records[name]["ids"][i]) in available])
            require(len(train) >= 30 and len(test) >= 5, "清洁应答前试次不足以使用既定分块。")
            splits.append((train, test, label))
        evaluations.append(("within_"+name, "within_record", splits))
    cross = []
    for name in FILES:
        train = np.array([i for i, r in enumerate(trials) if r["file"] != name])
        test = np.array([i for i, r in enumerate(trials) if r["file"] == name])
        cross.append((train, test, f"{'B' if name[0]=='A' else 'A'}_to_{name[0]}"))
    evaluations.append(("cross_Task-2", "cross_subject", cross))
    for evaluation, scope, splits in evaluations:
        coverage = np.zeros(len(trials), dtype=int)
        for tr, te, fold in splits:
            require(not np.intersect1d(tr, te).size, "训练测试重叠。")
            coverage[te] += 1
            manifest.append(dict(evaluation=evaluation, scope=scope, fold=fold, n_train=len(tr), n_test=len(te),
                                 train_trials=";".join(f"{trials[i]['file']}:{trials[i]['trial_id']}" for i in tr),
                                 test_trials=";".join(f"{trials[i]['file']}:{trials[i]['trial_id']}" for i in te)))
        selected = np.arange(len(trials)) if scope == "cross_subject" else np.array([i for i, r in enumerate(trials) if evaluation == "within_"+r["file"]])
        require(np.all(coverage[selected] == 1) and coverage.sum() == len(selected), "每个选中试次须恰好留出一次。")
    return evaluations, pd.DataFrame(manifest)


def train_mapping(trials, indices, variant, n_columns, p):
    gram, xy = np.zeros((n_columns, n_columns)), np.zeros((n_columns, 3))
    scale_square, n = np.zeros(n_columns), 0
    for i in indices:
        trial = trials[i]
        x = trial["designs"][variant][:, :n_columns]
        reference = trial["designs"]["nominal"][:, :n_columns]
        gram += x.T@x
        xy += x.T@trial["eeg"]
        scale_square += (reference*reference).sum(axis=0)
        n += len(x)
    scale = np.maximum(np.sqrt(scale_square/n), 1e-10)
    scale[:2] = 1
    norm = gram/n/scale[:, None]/scale[None, :]
    penalty = np.diag([0., 0.]+[p.ridge]*(n_columns-2))
    coefs_scaled = np.linalg.solve(norm+penalty, xy/n/scale[:, None])
    coef = coefs_scaled/scale[:, None]
    return coef, scale, float(np.trace(np.linalg.solve(norm+penalty, norm)))


def phase_mask(trial, region):
    t, target = trial["time"], trial["target_s"]
    if region == "visual":
        return (t >= .05) & (t <= .9)
    if region == "delay":
        return (t >= 1.0) & (t < target-.05)
    if region == "recognition":
        return t >= target+.05
    return t >= .05


def evaluate(trials, evaluations, p, models=MODELS, variant_label="nominal", collect_examples=True):
    rows, coefs, examples = [], [], {}
    for name, scope, splits in evaluations:
        for tr, te, fold in splits:
            for model, (variant, columns) in models.items():
                coef, scale, effective_df = train_mapping(trials, tr, variant, columns, p)
                for k in range(columns):
                    for ci, ch in enumerate(CHANNELS):
                        coefs.append(dict(evaluation=name, fold=fold, model=model, component=COMPONENTS[k],
                                          channel=ch, coefficient=coef[k, ci], train_reference_rms=scale[k], effective_df=effective_df))
                for i in te:
                    r = trials[i]
                    prediction = r["designs"][variant][:, :columns]@coef
                    for region in REGIONS:
                        mask = phase_mask(r, region)
                        require(mask.sum() >= 10, "评估窗口太短。")
                        obs = r["eeg"][mask]
                        err = obs-prediction[mask]
                        rows.append(dict(evaluation=name, scope=scope, file=r["file"], fold=fold, trial_id=r["trial_id"],
                                         variant=variant_label, region=region, model=model, n_values=err.size,
                                         sse=float(np.sum(err*err)), mse=float(np.mean(err*err)),
                                         sst=float(np.sum((obs-obs.mean(axis=0))**2))))
                    if collect_examples and scope == "within_record" and fold == "block_5":
                        key = (r["file"], r["trial_id"])
                        if key not in examples:
                            examples[key] = dict(time=r["time"], eeg=r["eeg"], target_s=r["target_s"], cutoff_s=r["cutoff_s"], predictions={})
                        examples[key]["predictions"][model] = prediction
    return pd.DataFrame(rows), pd.DataFrame(coefs), examples


def summarize(metrics):
    rows = []
    for (evaluation, scope, file, region, variant), group in metrics.groupby(["evaluation", "scope", "file", "region", "variant"], sort=False):
        reference = float(group[group.model == "trend"].sse.sum())
        for model, g in group.groupby("model", sort=False):
            sse, sst, nv = float(g.sse.sum()), float(g.sst.sum()), int(g.n_values.sum())
            rows.append(dict(evaluation=evaluation, scope=scope, file=file, region=region, variant=variant,
                             model=model, n_trials=len(g), n_values=nv, sse=sse, rmse=np.sqrt(sse/nv),
                             r2_trial_centered=1-sse/sst, skill_vs_train_trend=1-sse/reference))
    return pd.DataFrame(rows)


def compare_models(metrics, summary):
    comparisons = (("memory_added", "shape_memory", "shape_target"),
                   ("reactivation_added", "shape_memory", "memory_only"),
                   ("directional_memory", "shape_memory", "memory_blind"),
                   ("long_vs_short_memory", "shape_memory", "short_memory"))
    rows = []
    for (evaluation, file, region), group in summary.groupby(["evaluation", "file", "region"], sort=False):
        g = group.set_index("model")
        for comparison, full, reference in comparisons:
            sub = metrics[(metrics.evaluation == evaluation) & (metrics.file == file) & (metrics.region == region)]
            wide = sub.pivot(index=["fold", "trial_id"], columns="model", values="sse")
            blocks = wide.groupby(level="fold").sum()
            rows.append(dict(evaluation=evaluation, scope=g.iloc[0].scope, file=file, region=region,
                             comparison=comparison, full_model=full, reference_model=reference,
                             sse_reduction_pct=100*(1-g.loc[full, "sse"]/g.loc[reference, "sse"]),
                             n_trials=len(wide), n_trials_improved=int((wide[full] < wide[reference]).sum()),
                             n_test_blocks=len(blocks), n_test_blocks_improved=int((blocks[full] < blocks[reference]).sum())))
    return pd.DataFrame(rows)


def audit(root, trials, evaluations, records, q08, q09, visual, p, metrics):
    checks, old_rows = [], []
    events = pd.read_csv(root/"data"/"processed"/"trial_info.csv")
    for name in FILES:
        selected = [r for r in trials if r["file"] == name]
        old, _ = q09.blocked_fit(selected, records[name], q08)
        old_rows.append(old)
        new_trend = metrics[(metrics.file == name) & (metrics.evaluation == "within_"+name) & (metrics.model == "trend")]
        merged = old[old.model == "trend"].merge(new_trend, on=["trial_id", "region"], suffixes=("_09", "_13"))
        difference = np.max(np.abs(merged.sse_09-merged.sse_13)/np.maximum(np.abs(merged.sse_09), 1e-12))
        require(len(merged) == 3*len(selected) and difference < 1e-10, "13与09训练趋势基线不一致。")
        checks.append(dict(check=f"{name}_same_09_trend", passed=True, value=float(difference)))
        first = selected[0]
        event = events[(events.file_name == f"VisualCog{name}.mat") & (events.trial_id == first["trial_id"])].iloc[0]
        raw = np.asarray(loadmat(root/"data"/"raw"/f"VisualCog{name}.mat")["data"][:3], dtype=float)
        cue, click = int(event.cue_onset_sample), int(event.response_sample)
        start, stop = cue-round(.5*FS), click-round(p.cutoff_before_click_s*FS)
        altered = raw.copy()
        altered[:, stop:] = 1e8  # 大幅改变截止点及以后的实际原始数据。
        original = observation_filter(raw[:, start:stop].T)
        changed = observation_filter(altered[:, start:stop].T)
        require(np.array_equal(original, changed), "截止点后数据影响了预处理。")
        take = np.flatnonzero((np.arange(start, stop)-cue)/FS >= 0)[::p.decimate]
        err = float(np.max(np.abs(original[take]-first["eeg"])))
        require(err < 1e-10 and (click-(stop-1))/FS >= p.cutoff_before_click_s, "与09截取/滤波结果不一致。")
        checks.append(dict(check=f"{name}_future_data_excluded", passed=True, value=err))
        copied = dict(first, target=-first["target"], click=-first["click"])
        repeated, _ = build_design(copied["time"], copied["target_s"], copied["cutoff_s"], copied["cue"], visual, p)
        require(np.array_equal(repeated, first["designs"]["nominal"]), "目标符号或点击方向改变了模型设计。")
        checks.append(dict(check=f"{name}_behavior_codes_unused", passed=True, value=0.))
    for name, scope, splits in evaluations:
        train, test, fold = splits[0]
        coef, scale, _ = train_mapping(trials, train, "nominal", 14, p)
        modified = list(trials)
        j = test[0]
        modified[j] = dict(trials[j], eeg=trials[j]["eeg"]+1e8,
                           designs={key: value*1e8 for key, value in trials[j]["designs"].items()})
        changed_coef, changed_scale, _ = train_mapping(modified, train, "nominal", 14, p)
        require(np.array_equal(coef, changed_coef) and np.array_equal(scale, changed_scale), "测试数据进入了模型估计或缩放。")
        checks.append(dict(check=name+"_train_only_fit_and_scale", passed=True, value=0.))
    checks.append(dict(check="mirror_memory_encoding", passed=True,
                       value=float(np.max(np.abs(visual["encoding"][0]-visual["encoding"][1, :, ::-1])))))
    old_metrics = pd.concat(old_rows, ignore_index=True)
    return checks, q09.summarize(old_metrics)


def save_plots(summary, gains, examples, trials, out, q08):
    shown = ("shape_target", "memory_only", "shape_memory", "memory_blind", "short_memory")
    colors = ("#267d9d", "#c9a24e", "#c14e36", "#8e8e8e", "#8d709c")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for ax, scope, title in zip(axes, ("within_record", "cross_subject"), ("Within-record held-out blocks", "Transfer to the other participant")):
        sub = summary[(summary.scope == scope) & (summary.region == "recognition")]
        for j, model in enumerate(shown):
            values = sub[sub.model == model].set_index("file").loc[list(FILES), "skill_vs_train_trend"]*100
            bars = ax.bar(np.arange(2)+(j-2)*.16, values, .155, color=colors[j], label=model)
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=2)
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(np.arange(2), FILES)
        ax.set_ylabel("SSE reduction vs trained trend (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)
        ax.margins(y=.3)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3, fontsize=9)
    fig.suptitle("Pre-click target period: positive values improve on the trend baseline")
    q08.save_figure(fig, out, "cognitive_model_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
    for ax, scope in zip(axes, ("within_record", "cross_subject")):
        sub = gains[(gains.scope == scope) & (gains.comparison == "memory_added")]
        for j, name in enumerate(FILES):
            values = sub[sub.file == name].set_index("region").loc[list(REGIONS), "sse_reduction_pct"]
            bars = ax.bar(np.arange(4)+(j-.5)*.34, values, .32, label=name)
            ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
        ax.axhline(0, color="black", lw=.8)
        ax.set_xticks(np.arange(4), REGIONS, rotation=18, ha="right")
        ax.set_title(scope.replace("_", " "))
        ax.set_ylabel("Full memory vs visual + target: SSE reduction (%)")
        ax.margins(y=.3)
        ax.grid(axis="y", alpha=.2)
        ax.legend(fontsize=8)
    q08.save_figure(fig, out, "memory_increment_by_phase.png")

    keys = [min((k for k in examples if k[0] == name), key=lambda k:k[1]) for name in FILES]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for row, key in enumerate(keys):
        r = examples[key]
        for c, ch in enumerate(CHANNELS):
            ax = axes[row, c]
            ax.plot(r["time"], r["eeg"][:, c], color="#999999", lw=.8, label="Observed")
            for model, color in (("shape_target", "#267d9d"), ("shape_memory", "#c14e36")):
                ax.plot(r["time"], r["predictions"][model][:, c], color=color, lw=1.4, label=model)
            ax.axvline(r["target_s"], color="black", lw=.8, ls="--")
            ax.set_title(f"{key[0]} trial {key[1]} / {ch}")
            ax.set_xlabel("Seconds from cue")
            ax.set_ylabel("Input data units")
            ax.grid(alpha=.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
    fig.suptitle("First retained trial in final held-out block; dashed line = target onset")
    q08.save_figure(fig, out, "heldout_preclick_examples.png")

    first = trials[0]
    state = first["state"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True, layout="constrained")
    for ax, field, title in zip(axes, ("encoding", "memory", "reactivation"),
                                ("Fixed visual-network input", "Candidate memory trace", "Target-gated reactivation")):
        for d, color in enumerate(("#3366cc", "#dd482f")):
            ax.plot(state["time"], state[field][:, d], color=color, label="LR"[d]+" candidate")
        ax.axvline(first["target_s"], color="black", ls="--", lw=.8)
        ax.set_ylabel(title+"\n(arbitrary units)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("Seconds from cue; raw latent states before observation filtering")
    fig.suptitle(f"Illustration using first retained trial timing ({first['file']}, {first['trial_id']}); simulated, not measured sources")
    q08.save_figure(fig, out, "candidate_state_dynamics.png")


def write_report(eligibility, summary, gains, sensitivity, checks, old, p, elapsed, out):
    lines = ["问题三：连接第12步视觉网络的记忆维持与目标后再激活候选模型", "="*72,
        "状态：探索性、离线、已知提示与目标时刻的脑电条件预测。没有新增行为正确答案。",
        "", "一、数据与截止点",
        "仅Task-2有可用的真实±2点击事件；Task-1不能设置可信的应答前100ms终点。",
        "复用09逐试次截止点内滤波：从提示前0.5s到点击前约0.1s，0.5–30Hz四阶零相位带通。",
        "零相位滤波不读取截止点及以后的原始数据；基线、降采样、长窗饱和筛选均与09相同。",
        "实际最后一个输入样本距点击至少100ms，抽样后可能更早；仍无法排除更早的运动准备。"]
    for e in eligibility:
        lines.append(f"  {e['file']}: 03保留{e['q2_accepted']}，额外长窗排除{len(e['exclusions'])}，分析{e['long_window_accepted']}次。")
    lines += ["", "二、与第二问连接的模型",
        "直接读取12的simulation_shape_wc.npz，验证脚本指纹；提示图形、空间编码、36组E/I网络参数保持不变。",
        "提示视觉观测：沿用12的6个功能池原始读出，按本试次截止点内的观测滤波重处理。",
        "目标输入：同一视觉网络仅延长外部刺激为3s，网络参数不变；使用两种图形响应的平均作为非定向探测。",
        "目标实际分析段均短于模拟长度；此处是目标持续显示的候选简化，未声称复刻双目标逐像素空间布局。",
        "目标的中继均值和读出均值构成2个基线状态；不使用目标标记正负号或点击方向。",
        "记忆输入u_j(t)：12兴奋群体高于静息态的部分，与第j种固定图形空间特征加权汇聚。",
        "tau_M M_j'=-M_j+u_j(t)，j=L/R；M表示形状相关的候选痕迹。",
        "tau_C C_j'=-C_j+M_j(t)*g_target(t)，C表示目标门控的候选再激活。",
        f"固定tau_M={p.memory_tau_s}s，tau_C={p.reactivation_tau_s}s；不是辨识出的真实海马时间常数。",
        "观测：EEG=截距+线性趋势+6个提示视觉池+2个目标状态+2个M状态+2个C状态的线性混合。",
        "所有潜在分量与EEG接受同一截止点内观测滤波；只在训练试次估计有效混合系数。",
        "这是聚合状态之间的宏观候选网络，不进行脑源定位，也未实现或验证题目参考文献中的theta–gamma耦合机制。",
        "本模型实现提示信息维持和目标后再激活；因缺少实际目标图形/正确答案日志，未验证真实匹配判断或错误决策。",
        "", "三、对照与验证规则",
        "trend：训练截距与线性趋势；shape_visual：再加6个提示视觉池；shape_target：再加2个目标状态。",
        "memory_only：shape_target再加2个M；shape_memory：再加2个C，为固定主模型。",
        "memory_blind：将两个形状编码输入取平均，抹去记忆中的方向信息；视觉输入仍保持真实提示方向。",
        "short_memory：记忆时间常数改为0.15s，其他设置不变。",
        "记忆时间常数×0.8/×1.2敏感性全部报告，不从结果中选最优主模型。",
        "各折仅用名义模型训练数据计算一套RMS缩放，并在所有对照/敏感性中复用，避免单独缩放抹去衰减差异。",
        f"拟合损失为逐采样点平均平方误差+{p.ridge}×缩放后神经状态系数平方和，截距/趋势不罚。",
        "同09/12原编号20次一块，训练额外排除测试块两侧2次；另执行A->B、B->A严格跨被试预测。",
        "长窗口含更多采样点，拟合与汇总SSE中权重相应更大。全部保留试次均恰好留出一次/每类验证。",
        "主评价窗口：目标后50ms至截止点；另报告提示后50–900ms、1s至目标前50ms、全部应答前窗口。",
        "skill=1-SSE_model/SSE_训练趋势基线；该误差改进分数不是普通R²。正值表示优于可训练基线。",
        "CSV另给按各试次/通道时间均值中心化的描述性R²；不能与12条件平均波形的R²直接比较。",
        "重叠训练折不作为独立被试；本次不提供未经论证的记忆效应p值，也不将方向打乱对照冒充独立行为验证。",
        "", "四、主窗口的留出结果"]
    sub = summary[summary.region == "recognition"]
    for (evaluation, file), g in sub.groupby(["evaluation", "file"], sort=False):
        lines.append(f"[{evaluation} / 测试{file}]")
        for _, r in g.iterrows():
            lines.append(f"  {r.model:16s} RMSE={r.rmse:.4f}；相对训练趋势误差改进={100*r.skill_vs_train_trend:+.3f}%")
        h = gains[(gains.evaluation == evaluation) & (gains.file == file) & (gains.region == "recognition")]
        for _, r in h.iterrows():
            lines.append(f"  {r.comparison:22s} SSE减少={r.sse_reduction_pct:+.3f}%；改善试次{r.n_trials_improved}/{r.n_trials}；改善测试块{r.n_test_blocks_improved}/{r.n_test_blocks}")
    key = gains[(gains.region == "recognition") & (gains.comparison == "memory_added")]
    within = key[key.scope == "within_record"]
    cross = key[key.scope == "cross_subject"]
    lines += ["", f"主模型相对shape_target：记录内改善{int((within.sse_reduction_pct > 0).sum())}/2份；跨被试改善{int((cross.sse_reduction_pct > 0).sum())}/2个方向。"]
    if (key.sse_reduction_pct > 0).all():
        lines.append("当前固定设置在这四项比较中均有预测改善；仍只是两名被试上的探索性现象，不能据此确认记忆来源。")
    else:
        lines.append("新增候选记忆状态没有在记录内及跨被试验证中取得一致改善；不能写成记忆/海马机制已验证。")
    lines += ["", "五、记忆时间常数敏感性（主窗口）"]
    for _, r in sensitivity[sensitivity.region == "recognition"].iterrows():
        if r.model == "shape_memory":
            lines.append(f"  {r.variant:18s} {r.evaluation}/{r.file}: RMSE={r.rmse:.4f}, skill={100*r.skill_vs_train_trend:+.3f}%")
    lines += ["", "六、自动检查"]
    for _, r in checks.iterrows():
        lines.append(f"  {'PASS' if r.passed else 'FAIL'} {r['check']}: {r.value:.6g}")
    lines += ["PASS表示代码约束成立，不表示新增状态有统计显著性或真实解剖对应。",
        "", "七、三问完成范围",
        "问题一：仍按10–11的注入/强度实验描述保形能力和信号条件；本步不改变第一问结论。",
        "问题二：12给出形状驱动网络与候选方向表示；本步复用网络不代表其方向识别已可靠。",
        "问题三：已有可执行的视觉-维持-再激活宏观候选链条、真实点击前窗口、消融和跨被试验证。",
        "缺少可核验正确目标、错误/遗漏应答日志；不能从±1与±2同号推导正确率，也不能验证错误认知模型。",
        "仅三个额区电极、两名被试，不能唯一定位海马/LGN、确立认知因果机制或支持临床诊断。",
        "分析以03既定保留数据为条件；模型在已见数据上继续提出，属于探索性回顾分析。",
        "截止点由实际点击确定，预测是离线条件脑电重建，不是提前预测点击时刻或实时脑机接口性能。",
        "后续应按这些证据范围组织三问答案；任何强于数据支持的结论都需要额外独立观测。",
        "", "参考背景（本脚本为简化候选状态模型，不是以下模型逐式复现）：",
        "Daume et al. (2024), Nature 629:393–401. DOI:10.1038/s41586-024-07309-z.",
        "Compte et al. (2000), Cerebral Cortex 10:910–923. DOI:10.1093/cercor/10.9.910.",
        "视觉网络参考及固定参数见第12步报告。", "", f"运行耗时：{elapsed:.1f}秒。"]
    (out/"q3_shape_memory_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root/"results"/"13_q3_shape_memory_validation"
    out.mkdir(parents=True, exist_ok=True)
    started, p = time.perf_counter(), Parameters()
    q12 = module("12_q2_shape_driven_model.py", "q12_parent_of_13")
    q09 = module("09_q3_cognitive_joint_validation.py", "q09_parent_of_13")
    q08 = q12.import_validation()
    config = dict(parameters=asdict(p), primary_model="shape_memory", exploratory=True,
                  script_sha256=digest(__file__), parent09_sha256=digest(q09.__file__),
                  parent08_sha256=digest(q08.__file__), python=platform.python_version(), numpy=np.__version__,
                  event_inputs=["cue direction", "cue time", "target onset time", "click time for cutoff only"],
                  unused_event_values=["target sign", "click side", "behavioral correctness"],
                  validation="5 original 20-trial blocks with two-trial purge; A-to-B and B-to-A",
                  scaling="nominal training-only RMS reused by all variants", formal_significance_test=False)
    (out/"run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    with threadpool_limits(limits=1):
        print("步骤1/4：读取12的固定视觉网络，截取真实应答前信号……", flush=True)
        visual = load_visual(root, q12, p)
        records = {r["name"]: r for r in q08.load_data(root)}
        trials, eligibility, checks = prepare(root, records, q08, q09, visual, p)
        evaluations, manifest = make_evaluations(trials, records, q08)
        q08.save_csv(manifest, out/"split_manifest.csv")
        q08.save_csv(pd.DataFrame([dict(file=r["file"], trial_id=r["trial_id"], cue_direction="R" if r["cue"]==1 else "L",
                                         target_s=r["target_s"], cutoff_s=r["cutoff_s"], last_sample_s=r["time"][-1],
                                         n_samples=len(r["time"])) for r in trials]), out/"trial_windows.csv")
        (out/"eligibility.json").write_text(json.dumps(eligibility, ensure_ascii=False, indent=2), encoding="utf-8")
        config["inputs"] = {str(path.relative_to(root)): digest(path) for path in
                            [visual["source_file"], visual["config_file"], root/"data/processed/trial_info.csv",
                             *[root/"data/raw"/f"VisualCog{name}.mat" for name in FILES],
                             *[records[name]["path"] for name in FILES]]}
        config["parent12_sha256"] = visual["parent_script_sha256"]
        (out/"run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        print("步骤2/4：留块与双向跨被试预测，比较记忆状态及消融……", flush=True)
        metrics, coefficients, examples = evaluate(trials, evaluations, p)
        summary = summarize(metrics)
        gains = compare_models(metrics, summary)
        q08.save_csv(metrics, out/"heldout_trial_errors.csv")
        q08.save_csv(summary, out/"model_comparison.csv")
        q08.save_csv(gains, out/"memory_increment.csv")
        q08.save_csv(coefficients, out/"training_coefficients.csv")
        print("步骤3/4：时间常数敏感性、截止点和训练/测试隔离核验……", flush=True)
        sensitivity = [summary[summary.model.isin(["trend", "shape_memory"])].copy()]
        for variant in ("memory_tau_x0.8", "memory_tau_x1.2"):
            changed, _, _ = evaluate(trials, evaluations, p,
                                    models={"trend": ("nominal", 2), "shape_memory": (variant, 14)},
                                    variant_label=variant, collect_examples=False)
            sensitivity.append(summarize(changed))
        sensitivity = pd.concat(sensitivity, ignore_index=True)
        q08.save_csv(sensitivity, out/"memory_time_sensitivity.csv")
        more_checks, old = audit(root, trials, evaluations, records, q08, q09, visual, p, metrics)
        checks = pd.DataFrame(checks+more_checks)
        q08.save_csv(checks, out/"validation_checks.csv")
        q08.save_csv(old, out/"legacy09_same_trials.csv")
        print("步骤4/4：生成图表与中文报告……", flush=True)
        save_plots(summary, gains, examples, trials, out, q08)
        write_report(eligibility, summary, gains, sensitivity, checks, old, p,
                     time.perf_counter()-started, out)
        first = trials[0]
        np.savez_compressed(out/"fixed_example_states.npz", record_name=first["file"], trial_id=first["trial_id"],
                            **first["state"])
    print("\n完成。报告：", out/"q3_shape_memory_report.txt", flush=True)
    table = summary[summary.region == "recognition"]
    print(table.pivot(index=["scope", "file"], columns="model", values="rmse").round(4).to_string())
    print(gains[(gains.region == "recognition") & (gains.comparison == "memory_added")]
          [["evaluation", "file", "sse_reduction_pct", "n_trials_improved", "n_trials", "n_test_blocks_improved", "n_test_blocks"]].round(4).to_string(index=False))
    print("SSE减少为正表示改善；候选记忆状态的预测改善不等于验证真实海马来源。")


if __name__ == "__main__":
    main()
