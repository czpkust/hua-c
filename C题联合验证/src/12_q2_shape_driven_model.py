r"""问题二补充：镜像三角形 -> 空间特征 -> E/I群体 -> 三通道有效EEG。

运行：python -X utf8 .\src\12_q2_shape_driven_model.py
前置：03的4份ERP片段和08脚本。只写results/12_q2_shape_driven_model。
这是固定假设下的探索性候选模型，不是实际脑源定位或生理参数辨识。
默认1999次完整重拟合置换；所有L/R模型参数共享，只有空间输入不同。
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
import scipy
from scipy.ndimage import gaussian_filter
from scipy.optimize import root as solve_root
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.special import expit
from scipy.stats import beta
from threadpoolctl import threadpool_limits


SEED = 20260925
CHANNELS = ("Fz", "F3", "F4")
SOURCE_NAMES = tuple(f"x{x}_{y}" for x in ("left", "middle", "right") for y in ("outer", "center"))
MODEL_NAMES = ("shape_wc", "shape_blind", "no_feedback")
MODEL_COLORS = {"shape_wc": "#c14e36", "shape_blind": "#8c8c8c", "no_feedback": "#bb8a23",
                "old_basis_shared": "#267d9d", "old_basis_direction": "#81b7c5",
                "old_trend_shared": "#555555", "old_trend_direction": "#aaaaaa"}


@dataclass(frozen=True)
class Parameters:
    grid_size: int = 65
    dog_sigma_pixels: tuple = (1.0, 2.0)
    rf_radius: float = 0.30
    tau_g: float = 0.020
    tau_e: float = 0.025
    tau_i: float = 0.050
    tau_readout_1: float = 0.060
    tau_readout_2: float = 0.100
    w_ee: float = 8.0
    w_ei: float = 10.0
    w_ie: float = 10.0
    w_ii: float = 2.0
    lateral_gain: float = 0.8
    input_gain: float = 4.0
    threshold_e: float = 4.0
    threshold_i: float = 4.0
    delay_s: float = 0.035
    pulse_s: float = 0.200
    ridge: float = 0.100
    integration_substeps: int = 8


def require(condition, message):
    if not condition:
        raise ValueError(message)


def import_validation():
    path = Path(__file__).with_name("08_q2_mechanism_validation.py")
    if not path.is_file():
        raise FileNotFoundError(f"缺少既有验证脚本：{path}")
    spec = importlib.util.spec_from_file_location("q2_validation_08_for_12", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def spatial_inputs(p):
    """相同DoG/梯度方向能量/局部感受野；无人工L/R增益。"""
    coords = np.linspace(-1, 1, p.grid_size)
    xx, yy = np.meshgrid(coords, coords)
    right = ((xx >= -0.5) & (xx <= 0.7) & (np.abs(yy) <= 0.5 * (0.7 - xx))).astype(float)
    images = np.stack([right[:, ::-1], right])
    centers = np.array([(x, y, theta) for x in (-0.5, 0.0, 0.5)
                        for y in (-0.5, 0.0, 0.5) for theta in np.arange(4)*np.pi/4])
    feature, dogs = [], []
    for im in images:
        dog = gaussian_filter(im, p.dog_sigma_pixels[0])-gaussian_filter(im, p.dog_sigma_pixels[1])
        dogs.append(dog)
        gx = gaussian_filter(dog, 1.0, order=(0, 1))
        gy = gaussian_filter(dog, 1.0, order=(1, 0))
        vals = []
        for x, y, theta in centers:
            rf = np.exp(-((xx-x)**2+(yy-y)**2)/(2*p.rf_radius**2))
            energy = np.abs(np.cos(theta)*gx+np.sin(theta)*gy)
            vals.append(np.sum(rf*energy)/np.sum(rf))
        feature.append(vals)
    feature = np.asarray(feature)
    feature /= feature.max()  # 只用两幅预设图形，不用任何EEG或试次标签统计量。
    delta = centers[:, None, :2]-centers[None, :, :2]
    angle = centers[:, None, 2]-centers[None, :, 2]
    lateral = np.exp(-np.sum(delta**2, axis=2)/(2*0.5**2))*np.exp(2*np.cos(2*angle))
    np.fill_diagonal(lateral, 0)
    lateral /= lateral.sum(axis=1, keepdims=True)
    pooling = np.zeros((6, 36))
    for k, (x, y, theta) in enumerate(centers):
        ix = int(np.argmin(np.abs(np.array([-0.5, 0, 0.5])-x)))
        pooling[2*ix+int(y == 0), k] = 1
    pooling /= pooling.sum(axis=1, keepdims=True)
    reflected = np.array([(2-ix)*12+iy*4+((-ori) % 4)
                          for ix in range(3) for iy in range(3) for ori in range(4)])
    return dict(images=images, dog=np.asarray(dogs), features=feature, centers=centers,
                lateral=lateral, pooling=pooling, reflected=reflected)


def simulate(spatial, p, kind="shape_wc", substeps=None):
    """稳定平衡态启动；显式E/I反馈；RK4固定步长，无隐式限幅。"""
    fs = 256
    sub = p.integration_substeps if substeps is None else substeps
    dt = 1/(fs*sub)
    features = spatial["features"].copy()
    if kind == "shape_blind":
        features[:] = features.mean(axis=0)
    feedback = kind != "no_feedback"
    wee, wei, wii, lat = (p.w_ee, p.w_ei, p.w_ii, p.lateral_gain) if feedback else (0, 0, 0, 0)
    def equilibrium(v):
        e, i = v
        return [-e+expit((wee+lat)*e-wei*i-p.threshold_e),
                -i+expit(p.w_ie*e-wii*i-p.threshold_i)]
    sol = solve_root(equilibrium, [0.02, 0.02])
    require(sol.success and np.linalg.norm(equilibrium(sol.x)) < 1e-10, "静息平衡态求解失败。")
    e0, i0 = sol.x
    require(0 < e0 < 1 and 0 < i0 < 1, "静息态越界。")
    # 状态顺序：LGN样中继36、E36、I36、读出低通6、读出低通6。
    state = np.zeros((2, 120))
    state[:, 36:72], state[:, 72:108] = e0, i0
    lateral, pooling = spatial["lateral"], spatial["pooling"]
    def rhs(s, drive):
        g, e, i = s[:, :36], s[:, 36:72], s[:, 72:108]
        a, b = s[:, 108:114], s[:, 114:120]
        current = ((e-e0)-(i-i0)) @ pooling.T
        return np.concatenate([
            (-g+drive)/p.tau_g,
            (-e+expit(wee*e-wei*i+lat*(e@lateral.T)+p.input_gain*g-p.threshold_e))/p.tau_e,
            (-i+expit(p.w_ie*e-wii*i-p.threshold_i))/p.tau_i,
            (-a+current)/p.tau_readout_1,
            (-b+a)/p.tau_readout_2,
        ], axis=1)
    n = int(round(3*fs*sub))
    times = -.5+np.arange(n//sub+1)/fs
    saved = np.empty((2, len(times), 120))
    saved[:, 0] = state
    minimum, maximum = min(e0, i0), max(e0, i0)
    # 在脉冲起止处切开积分步，避免改变步长时移动刺激时刻。
    edges = (p.delay_s, p.delay_s+p.pulse_s)
    for j in range(n):
        left, right = -.5+j*dt, -.5+(j+1)*dt
        points = [left]+[edge for edge in edges if left < edge < right]+[right]
        for a, b in zip(points[:-1], points[1:]):
            h, mid = b-a, (a+b)/2
            drive = features if p.delay_s <= mid < p.delay_s+p.pulse_s else np.zeros_like(features)
            k1 = rhs(state, drive)
            k2 = rhs(state+.5*h*k1, drive)
            k3 = rhs(state+.5*h*k2, drive)
            k4 = rhs(state+h*k3, drive)
            state += h*(k1+2*k2+2*k3+k4)/6
        minimum = min(minimum, float(state[:, 36:108].min()))
        maximum = max(maximum, float(state[:, 36:108].max()))
        if (j+1) % sub == 0:
            saved[:, (j+1)//sub] = state
    require(np.isfinite(saved).all() and minimum >= 0 and maximum <= 1, "神经群状态不稳定或越界。")
    # 模型也经过03测量处理。长零背景只作用于基线已扣除的合成读出。
    source = saved[:, :, 114:120]
    padded = np.pad(source, ((0, 0), (10*fs, 10*fs), (0, 0)))
    sos = butter(4, [0.5, 30], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, padded, axis=1)[:, 10*fs:10*fs+len(times)]
    base = (times >= -.2) & (times < 0)
    filtered -= filtered[:, base].mean(axis=1, keepdims=True)
    mask = (times >= 0) & (times <= .9)
    z = filtered[:, mask]
    scale = np.sqrt(np.mean(z*z, axis=(0, 1)))
    require(np.all(scale > 1e-10), "合成读出全为零。")
    z = z/scale
    tm = times[mask]
    trend = np.column_stack([np.ones(len(tm)), (tm-tm.mean())/(np.ptp(tm)/2)])
    full = np.stack([np.column_stack([zd, trend]) for zd in z])
    design = full.reshape(-1, 8)
    penalty = np.diag([p.ridge]*6+[0.0, 0.0])
    projection = np.linalg.solve(design.T@design/len(design)+penalty, design.T/len(design))
    return dict(time=tm, sim_time=times, states=saved, source=source, source_filtered=filtered,
                scale=scale, z=z, full=full, trend=trend, projection=projection,
                condition=float(np.linalg.cond(design)), effective_df=float(np.trace(projection@design)),
                min_state=minimum, max_state=maximum, resting_e=float(e0), resting_i=float(i0))


def fit_mapping(train_wave, train_labels, sim):
    """只接收训练波形和训练标签。L/R条件均值等权，映射B和趋势共享。"""
    require(set(np.unique(train_labels)) == {0, 1}, "训练集缺少方向类别。")
    means = np.stack([train_wave[train_labels == d].mean(axis=0) for d in (0, 1)])
    coef = sim["projection"]@means.reshape(-1, 3)
    return coef, sim["full"]@coef


def remove_trend(wave, trend):
    return wave-np.einsum("tp,npc->ntc", trend, np.einsum("pt,ntc->npc", np.linalg.pinv(trend), wave))


def score_unknown(train_wave, train_y, test_wave, sim, raw_train=None):
    """未知方向时分别比较两个候选预测；接口不接收测试方向。"""
    coef, templates = fit_mapping(train_wave, train_y, sim)
    template = remove_trend(templates, sim["trend"])
    obs = remove_trend(test_wave, sim["trend"])
    noise_data = remove_trend(train_wave if raw_train is None else raw_train, sim["trend"])
    variance = noise_data.var(axis=0, ddof=1).mean(axis=0)
    variance = np.maximum(variance, max(float(np.max(variance))*1e-8, 1e-12))
    delta, middle = template[1]-template[0], (template[1]+template[0])/2
    contributions = np.sum((obs-middle)*delta/variance[None, None, :], axis=1)
    scores = contributions.sum(axis=1)
    return (scores > 0).astype(int), scores, coef, contributions


def prepare_records(records, v):
    splits, smooth, raw, manifests = {}, {}, {}, []
    for r in records:
        name = r["name"]
        splits[name], rows = v.make_block_splits(r)
        manifests.extend(rows)
        mask = (r["time"] >= 0) & (r["time"] <= .9)
        raw[name] = r["epochs"][:, :, mask].transpose(0, 2, 1)
        smooth[name] = savgol_filter(r["epochs"], 21, 3, axis=-1)[:, :, mask].transpose(0, 2, 1)
    return splits, smooth, raw, manifests


def reconstruction(records, splits, smooth, raw, simulations, v):
    rows, coefficients, examples = [], [], {}
    for r in records:
        name = r["name"]
        y = (r["directions"] == "R").astype(int)
        design = v.build_design(r["time"], r["fs"], v.SETTINGS[0])
        fitted = v.fit_single_trials(r, design, v.SETTINGS[0])
        old, _ = v.validate_reconstruction(r, design, fitted, splits[name], v.SETTINGS[0])
        for row in old:
            row["model"] = "old_"+row["model"]
        rows.extend(old)
        for fold, (tr, te, label) in enumerate(splits[name], 1):
            for model, sim in simulations.items():
                coef, prediction = fit_mapping(smooth[name][tr], y[tr], sim)
                for k, component in enumerate((*SOURCE_NAMES, "intercept", "linear_trend")):
                    for ci, channel in enumerate(CHANNELS):
                        coefficients.append(dict(file=name, fold=fold, model=model, component=component,
                                                 channel=channel, value=float(coef[k, ci])))
                for d in (0, 1):
                    te_d = te[y[te] == d]
                    observed = raw[name][te_d].mean(axis=0)
                    for ci, channel in enumerate((*CHANNELS, "joint")):
                        sl = slice(None) if channel == "joint" else slice(ci, ci+1)
                        rows.append(dict(variant="nominal", file=name, fold=fold, direction="LR"[d],
                                         model=model, channel=channel, n_train=len(tr), n_test=len(te_d),
                                         **v.score_curve(observed[:, sl], prediction[d, :, sl])))
                    if fold == 5 and model in MODEL_NAMES:
                        examples[(name, model, d)] = dict(observed=observed, predicted=prediction[d], n=len(te_d))
    return pd.DataFrame(rows), pd.DataFrame(coefficients), examples


def evaluation_sets(records, splits):
    meta = pd.concat([pd.DataFrame(dict(file=r["name"], subject=r["subject"], task=r["task"],
                                       trial_id=r["ids"], direction=r["directions"])) for r in records], ignore_index=True)
    evaluations = []
    for r in records:
        sel = np.flatnonzero(meta["file"].to_numpy() == r["name"])
        evaluations.append((f"within_{r['name']}", "within_record", sel, splits[r["name"]]))
    for task in ("Task-1", "Task-2", "Pooled"):
        sel = np.arange(len(meta)) if task == "Pooled" else np.flatnonzero(meta["task"].to_numpy() == task)
        subjects = meta.iloc[sel]["subject"].to_numpy()
        cross = [(np.flatnonzero(subjects != s), np.flatnonzero(subjects == s),
                  f"{'B' if s == 'A' else 'A'}_to_{s}") for s in ("A", "B")]
        evaluations.append((f"cross_{task}", "cross_subject", sel, cross))
    return meta, evaluations


def make_fast_decoder(train, raw, splits, sim):
    """固定线性时间算子可缓存；标签均值、全部映射每次置换重新估计。"""
    # 分类先投影掉每试次截距/趋势；这是固定算子，不估计测试集统计量。
    trend = sim["trend"]
    residual_design = remove_trend(sim["full"], trend)
    residual_raw = remove_trend(raw, trend)
    cache = []
    for tr, te, label in splits:
        require(not np.intersect1d(tr, te).size, "训练与测试重叠。")
        variance = residual_raw[tr].var(axis=0, ddof=1).mean(axis=0)
        variance = np.maximum(variance, max(float(variance.max())*1e-8, 1e-12))
        # 按线性性把固定投影应用于每试次：改变标签时重算均值即等价完整拟合。
        pt = sim["projection"].reshape(8, 2, len(sim["time"]))
        projected = np.einsum("kdt,ntc->ndkc", pt, train[tr])
        # 得分(delta*(Y-mid))可在8维空间精确计算，节省1999次置换时间。
        delta = residual_design[1]-residual_design[0]
        mid = (residual_design[1]+residual_design[0])/2
        cross = np.einsum("tk,ntc->nkc", delta, residual_raw[te])/variance[None, None, :]
        gram = delta.T@mid
        cache.append((tr, te, label, projected, cross, gram, variance))
    def predict(y):
        pred = np.full(len(y), -1, dtype=int)
        score = np.full(len(y), np.nan)
        fold = np.empty(len(y), dtype=object)
        for tr, te, label, projected, cross, gram, variance in cache:
            yy = y[tr]
            require(np.any(yy == 0) and np.any(yy == 1), "置换训练集缺失类别。")
            coef = projected[yy == 0, 0].mean(axis=0)+projected[yy == 1, 1].mean(axis=0)
            correction = np.sum(coef*(gram@coef)/variance[None, :])
            score[te] = np.einsum("nkc,kc->n", cross, coef)-correction
            pred[te] = (score[te] > 0).astype(int)
            fold[te] = label
        require(np.all(pred >= 0) and np.isfinite(score).all(), "测试覆盖不完整。")
        return pred, score, fold
    return predict


def classify(records, splits, smooth, raw, sim, permutations, v, out):
    meta, evaluations = evaluation_sets(records, splits)
    train_all = np.concatenate([smooth[r["name"]] for r in records])
    raw_all = np.concatenate([raw[r["name"]] for r in records])
    y_all = (meta["direction"].to_numpy() == "R").astype(int)
    summary, predictions, folds, nulls, audit = [], [], [], [], []
    for name, scope, sel, local in evaluations:
        print(f"  {name}：共享映射拟合及{permutations}次完整置换……", flush=True)
        x, rr, y = train_all[sel], raw_all[sel], y_all[sel]
        mm = meta.iloc[sel].reset_index(drop=True)
        coverage = np.zeros(len(y), dtype=int)
        for tr, te, _ in local:
            coverage[te] += 1
        require(np.all(coverage == 1), "分类试次应恰好留出一次。")
        decoder = make_fast_decoder(x, rr, local, sim)
        pred, score, fold = decoder(y)
        features_oof = np.empty((len(y), 3))
        # 用直接波形拟合逐折核验缓存等价；不做标签依赖的分数翻转或阈值校准。
        for tr, te, label in local:
            pp, ss, _, contributions = score_unknown(x[tr], y[tr], rr[te], sim, raw_train=rr[tr])
            features_oof[te] = contributions
            err = float(np.max(np.abs(score[te]-ss)))
            require(np.array_equal(pred[te], pp) and np.allclose(score[te], ss, rtol=1e-9, atol=1e-9),
                    "快速置换实现与直接重拟合不一致。")
            altered_y = y.copy()
            altered_y[te] = 1-altered_y[te]
            # 仅检查本折；跨折中同一试次可能属于其他折的训练集。
            pp2, ss2, _, _ = score_unknown(x[tr], altered_y[tr], rr[te], sim, raw_train=rr[tr])
            require(np.array_equal(pp, pp2) and np.array_equal(ss, ss2), "测试标签影响了本折预测。")
            audit.append(dict(evaluation=name, fold=label, fast_direct_max_error=err,
                              test_labels_unused=True, n_train=len(tr), n_test=len(te)))
            folds.append(dict(evaluation=name, scope=scope, fold=label,
                              **v.classification_metrics(y[te], pred[te], score[te])))
        observed = v.classification_metrics(y, pred, score)
        result = mm.copy()
        result["evaluation"], result["scope"], result["fold"] = name, scope, fold
        result["predicted_direction"], result["score_R_minus_L"] = np.where(pred, "R", "L"), score
        for ci, channel in enumerate(CHANNELS):
            result[f"feature_{channel}"] = features_oof[:, ci]
        predictions.append(result)
        groups = [np.asarray(g) for g in mm.groupby(["file", (mm["trial_id"]-1)//20], sort=True).indices.values()]
        rng = np.random.default_rng(v.seed_for(name))  # 与08相同的每项置换种子及标签序列。
        null = np.empty(permutations)
        for k in range(permutations):
            yp = y.copy()
            for g in groups:
                yp[g] = rng.permutation(y[g])
            pp, perm_score, _ = decoder(yp)
            if k == 0:
                tr, te, label = local[0]
                direct_pred, direct_score, _, _ = score_unknown(x[tr], yp[tr], rr[te], sim, raw_train=rr[tr])
                require(np.array_equal(pp[te], direct_pred) and np.allclose(perm_score[te], direct_score, rtol=1e-9, atol=1e-9),
                        "置换标签后快速求解与直接重拟合不一致。")
            null[k] = .5*(np.mean(pp[yp == 0] == 0)+np.mean(pp[yp == 1] == 1))
            if (k+1) % 500 == 0 or k+1 == permutations:
                print(f"    置换 {k+1}/{permutations}", flush=True)
        exceed = int(np.sum(null >= observed["balanced_accuracy"]-1e-12))
        low = float(beta.ppf(.025, exceed, permutations-exceed+1)) if exceed else 0.
        high = float(beta.ppf(.975, exceed+1, permutations-exceed)) if exceed < permutations else 1.
        summary.append(dict(evaluation=name, scope=scope, n_folds=len(local), **observed,
                            permutation_p=(1+exceed)/(1+permutations), n_permutations=permutations,
                            null_exceedances=exceed, p_mc_95_low=low, p_mc_95_high=high,
                            null_ba_median=float(np.median(null)), null_ba_p025=float(np.quantile(null, .025)),
                            null_ba_p975=float(np.quantile(null, .975))))
        nulls.extend(dict(evaluation=name, permutation=k+1, balanced_accuracy=float(a)) for k, a in enumerate(null))
    summary = pd.DataFrame(summary)
    summary["bh_fdr_q_7_tests"] = v.fdr_bh(summary["permutation_p"].to_numpy())
    v.save_csv(summary, out/"shape_classification_summary.csv")
    v.save_csv(pd.concat(predictions, ignore_index=True), out/"classification_predictions.csv")
    feature_cols = ["evaluation", "scope", "fold", "file", "subject", "task", "trial_id", "direction",
                    "feature_Fz", "feature_F3", "feature_F4", "score_R_minus_L"]
    v.save_csv(pd.concat(predictions, ignore_index=True)[feature_cols], out/"direction_features_oof.csv")
    v.save_csv(pd.DataFrame(folds), out/"classification_folds.csv")
    v.save_csv(pd.DataFrame(nulls), out/"permutation_null_scores.csv")
    v.save_csv(pd.DataFrame(audit), out/"prediction_audit.csv")
    return summary


def model_checks(spatial, simulations, p):
    full, blind, nofb = (simulations[n] for n in MODEL_NAMES)
    fine = simulate(spatial, p, substeps=2*p.integration_substeps)
    mirror = np.array([4, 5, 2, 3, 0, 1])
    numeric = np.linalg.norm(full["source"]-fine["source"])/np.linalg.norm(fine["source"])
    checks = [
        ("mirror_pixels_exact", float(np.max(np.abs(spatial["images"][0]-spatial["images"][1, :, ::-1]))), 1e-12),
        ("mirror_spatial_features", float(np.max(np.abs(spatial["features"][0]-spatial["features"][1, spatial["reflected"]]))), 1e-10),
        ("mirror_dynamics", float(np.max(np.abs(full["source"][0]-full["source"][1][:, mirror]))), 1e-10),
        ("symmetric_readout_cancels_LR", float(np.max(np.abs(full["z"][0].sum(axis=1)-full["z"][1].sum(axis=1)))), 1e-10),
        ("shape_blind_LR_equal", float(np.max(np.abs(blind["full"][0]-blind["full"][1]))), 1e-12),
        ("half_step_relative_error", float(numeric), 1e-5),
    ]
    rows = [dict(check=n, value=a, upper_limit=b, passed=bool(a <= b)) for n, a, b in checks]
    changed = float(np.linalg.norm(full["source"]-nofb["source"])/np.linalg.norm(full["source"]))
    rows.append(dict(check="feedback_changes_simulated_response", value=changed, upper_limit=np.nan, passed=bool(changed > 1e-4)))
    tail = float(np.max(np.abs(full["source"][:, -1]))/np.max(np.abs(full["source"])))
    rows.append(dict(check="simulation_tail_relative_amplitude", value=tail, upper_limit=.001, passed=bool(tail < .001)))
    df = pd.DataFrame(rows)
    require(df["passed"].all(), f"模型自检未通过：\n{df.to_string(index=False)}")
    return df


def save_model(spatial, simulations, p, out, v):
    features = []
    for d in (0, 1):
        for k, (x, y, theta) in enumerate(spatial["centers"]):
            features.append(dict(direction="LR"[d], unit=k, x=x, y=y, orientation_deg=theta*180/np.pi,
                                 feature=spatial["features"][d, k]))
    v.save_csv(pd.DataFrame(features), out/"spatial_features.csv")
    trajectories = []
    info = []
    for model, sim in simulations.items():
        info.append(dict(model=model, condition_number=sim["condition"], effective_df=sim["effective_df"],
                         min_state=sim["min_state"], max_state=sim["max_state"],
                         resting_e=sim["resting_e"], resting_i=sim["resting_i"]))
        for d in (0, 1):
            table = pd.DataFrame(sim["z"][d], columns=SOURCE_NAMES)
            table.insert(0, "time_s", sim["time"])
            table.insert(0, "direction", "LR"[d])
            table.insert(0, "model", model)
            trajectories.append(table)
        np.savez_compressed(out/f"simulation_{model}.npz", time_s=sim["sim_time"],
                            state=sim["states"], source=sim["source"], filtered_source=sim["source_filtered"],
                            scale=sim["scale"], fit_time_s=sim["time"], design=sim["full"])
    v.save_csv(pd.concat(trajectories, ignore_index=True), out/"model_trajectories.csv")
    v.save_csv(pd.DataFrame(info), out/"model_conditioning.csv")


def plot_results(spatial, simulations, summary, classification, examples, records, out, v):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained")
    for d in (0, 1):
        axes[d, 0].imshow(spatial["images"][d], cmap="gray_r", origin="lower", extent=(-1, 1, -1, 1))
        axes[d, 0].set_title(f"{'LR'[d]}: assumed triangle")
        axes[d, 1].imshow(spatial["features"][d].reshape(3, 3, 4).sum(axis=2).T,
                          origin="lower", cmap="viridis", vmin=0,
                          vmax=spatial["features"].reshape(2, 3, 3, 4).sum(axis=3).max())
        axes[d, 1].set_title("Spatial energy (sum over orientations)")
        sim = simulations["shape_wc"]
        for k, label in enumerate(SOURCE_NAMES):
            axes[d, 2].plot(sim["time"], sim["z"][d, :, k], label=label)
        axes[d, 2].set_title("Processed, scaled readout pools")
        axes[d, 2].set_xlabel("Time (s)")
    axes[1, 2].legend(fontsize=7, ncol=2)
    fig.suptitle("Shared parameters: direction changes only the spatial input")
    v.save_figure(fig, out, "spatial_model.png")

    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    names = [r["name"] for r in records]
    models = ("old_trend_shared", "old_basis_shared", "shape_blind", "no_feedback", "shape_wc")
    for j, model in enumerate(models):
        val = summary[summary["model"] == model].set_index("file").loc[names, "r2"]
        bars = ax.bar(np.arange(4)+(j-2)*.17, val, width=.165, color=MODEL_COLORS[model], label=model)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.axhline(0, color="black", lw=.7)
    ax.set_xticks(np.arange(4), names)
    ax.set_ylabel("Held-out block-mean R-squared")
    ax.set_title("Same test blocks and waveforms; all five folds included")
    ax.legend(fontsize=9, ncol=3, loc="upper center", bbox_to_anchor=(.5, 1.25))
    ax.grid(axis="y", alpha=.2)
    ax.margins(y=.2)
    v.save_figure(fig, out, "blocked_model_comparison.png")

    fig, axes = plt.subplots(4, 3, figsize=(12, 12), sharex=True, layout="constrained")
    for row, name in enumerate(names):
        for ci, ch in enumerate(CHANNELS):
            ax = axes[row, ci]
            for d, color in ((0, "#3366cc"), (1, "#dd482f")):
                case = examples[(name, "shape_wc", d)]
                ax.plot(simulations["shape_wc"]["time"], case["observed"][:, ci], color=color, lw=1, label=f"{'LR'[d]} observed")
                ax.plot(simulations["shape_wc"]["time"], case["predicted"][:, ci], color=color, lw=1.5, ls="--", label=f"{'LR'[d]} predicted")
            ax.grid(alpha=.2)
            if row == 0:
                ax.set_title(ch)
            if ci == 0:
                ax.set_ylabel(name+"\nInput amplitude units")
            if row == 3:
                ax.set_xlabel("Time from cue (s)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.suptitle("Fixed illustration: trials 81-100 held out; train on accepted trials 1-78")
    v.save_figure(fig, out, "held_out_curves.png")

    fig, ax = plt.subplots(figsize=(12, 5), layout="constrained")
    xx = np.arange(len(classification))
    ax.bar(xx, classification["balanced_accuracy"], color="#c14e36", width=.6)
    ax.axhline(.5, color="gray", ls="--")
    for j, r in classification.iterrows():
        ax.text(j, .94, f"BA={r.balanced_accuracy:.3f}\np={r.permutation_p:.3f}\nq={r.bh_fdr_q_7_tests:.3f}", ha="center", va="top", fontsize=9)
    ax.set_xticks(xx, classification["evaluation"].str.replace("within_", "").str.replace("cross_", "Cross "), rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Unknown-direction prediction: shared forward map refitted in every permutation")
    ax.grid(axis="y", alpha=.2)
    v.save_figure(fig, out, "shape_classification.png")


def write_report(records, p, checks, summary, classification, sensitivity, elapsed, permutations, out):
    lines = ["问题二补充：空间形状驱动的神经群与EEG有效观测模型", "="*70,
        "范围：补足形状输入、显式反馈、共享参数映射；不是已证实的脑源解释。",
        "性质：在已分析过的两名被试上继续探索；不得当作新独立验证集或确证性检验。",
        "", "一、模型与假设",
        "输入：65×65理想实心镜像三角形；相同面积、对比度。原刺激逐像素图未提供，尺寸为假设。",
        "空间特征：中心-周边DoG -> 4方向梯度幅值 -> 3×3高斯感受野，共36维。",
        "此设计借鉴相对空间配置思想，不是COSFIRE原论文的完整复现。",
        "G为LGN样低通中继；36组E/I群体有自兴奋、抑制反馈和相邻群体耦合。",
        "tau_g G'=-G+f(shape)*pulse(t);",
        "tau_e E'=-E+sigmoid(8E-10I+0.8WE+4G-4);",
        "tau_i I'=-I+sigmoid(10E-2I-4).",
        "读出：E/I相对静息态差值，汇聚为6个视野功能池，再串联60/100ms低通。",
        "模型读出同样做0.5–30Hz四阶零相位带通和提示前200ms基线校正。",
        "拟合0–0.9s；每池按两种合成图形的RMS缩放，缩放不使用EEG。",
        "y_c(t|d)=b_c+a_c*t+sum_k B_kc*z_k(t|d)。B、b、a在L/R之间完全共享。",
        "令P交换左右功能池，则z_L=z_R*P，故y_R-y_L=z_R*(I-P)*B；",
        "如果PB=B（左右池观测权重对称），即使空间活动不同也会抵消为零头皮差异。",
        "B为有效混合系数；6池是视野功能分组，不是6个已定位的解剖脑区。",
        "模拟参数预先在脚本固定，未根据本次测试成绩优化；具体数值见run_config.json。",
        "36群体网络不等于36个可由三个电极辨识的真实脑源，亦不证实LGN或某皮层的活动。",
        "", "二、训练、对照和验证",
        f"使用03固定保留试次共{sum(len(r['ids']) for r in records)}；不新增剔除，不读取行为作答标签。",
        "每文件原编号20次一块，共5折；训练额外排除测试块两侧各2次。与08相同。",
        "训练波形额外21点三阶SG平滑；测试目标一直是03波形，等同08名义设置。",
        "训练L/R均值等权；最小化||D*coef-Y||²/(2T)+0.1||B||²，趋势不罚。",
        "shape_wc：完整共享形状驱动模型。",
        "shape_blind：输入换为L/R特征的平均，保留同样36群体/6池/参数量。",
        "no_feedback：去掉E自激、I到E抑制、I自抑和群间耦合，仅留G->E->I前馈。",
        "old_*：重新执行08名义模型；参数量、正则化与新模型不同，作为既有结果基准。",
        "shape_blind/no_feedback用于重建对照；未从三个模型中挑分类成绩最好者。",
        "R²汇总所有留出块×方向×通道SSE/SST，SST按每条件/通道的时间均值中心化。",
        "条件重建知道输入图形方向，不能据此证明未知方向可被识别。",
        "分类只用shape_wc，逐试次比较两个候选模板；固定去趋势，按训练数据通道方差加权。",
        "3维方向特征phi_c=sum_t[(Y_c-mid_c)*(template_R_c-template_L_c)]/var_train_c；",
        "仅用训练折拟合模板；三个通道特征相加即分类得分，见direction_features_oof.csv。",
        "测试标签不参与模板选择、阈值、权重、符号或参数估计；得分不是校准的概率。",
        "分块内分类4项；A->B与B->A合并的跨被试分类3项（Task-1/Task-2/Pooled）。",
        "Pooled仅在训练被试的两个任务上拟合一套共享映射，不访问测试被试统计量。",
        f"每项{permutations}次文件×原20试次块内置换，每次完整重算L/R均值和共享映射。",
        "固定线性算子和不依赖标签的训练方差可缓存；已逐折核验与直接拟合数值等价。",
        "p=(1+置换BAC>=实测BAC次数)/(1+置换次数)；BH-FDR仅覆盖本次7项。",
        "未校正历史06/08/12之间的所有探索；跨折合并AUC为描述量，不作额外显著性结论。",
        "p_mc区间仅为置换尾概率的Monte Carlo误差区间，不是准确率的总体置信区间。",
        "", "三、留出块重建（不截断负R²）"]
    for r in records:
        name = r["name"]
        g = summary[summary.file == name].set_index("model")
        lines.append(name)
        for model in ("old_trend_shared", "old_basis_shared", "old_basis_direction", *MODEL_NAMES):
            lines.append(f"  {model:22s} R²={g.loc[model, 'r2']:+.4f}  RMSE={g.loc[model, 'rmse']:.4f}")
        lines.append(f"  完整模型相对去空间输入 ΔR²={g.loc['shape_wc','r2']-g.loc['shape_blind','r2']:+.4f}")
        lines.append(f"  完整模型相对去反馈 ΔR²={g.loc['shape_wc','r2']-g.loc['no_feedback','r2']:+.4f}")
    pivot = summary.pivot(index="file", columns="model", values="r2")
    lines += [f"完整模型优于旧共享时间基函数：{int((pivot.shape_wc > pivot.old_basis_shared).sum())}/4份记录；",
              f"优于去空间输入对照：{int((pivot.shape_wc > pivot.shape_blind).sum())}/4；",
              f"优于去反馈对照：{int((pivot.shape_wc > pivot.no_feedback).sum())}/4。",
              "这些是同一数据上的描述性比较，不构成对重叠训练折的独立样本显著性检验。"]
    lines += ["", "四、未知方向分类"]
    for _, r in classification.iterrows():
        lines.append(f"  {r.evaluation:19s} n={r.n:3d} BAC={r.balanced_accuracy:.4f} AUC={r.auc:.4f} p={r.permutation_p:.4f} q={r.bh_fdr_q_7_tests:.4f}")
    significant = classification[classification.bh_fdr_q_7_tests < .05]
    if len(significant):
        lines.append("本次7项中FDR<0.05："+", ".join(significant.evaluation)+"；仍属探索结果，不能推广到人群。")
    else:
        lines.append("本次7项均未达到FDR<0.05；未获得稳定未知方向识别的证据。")
    lines += ["", "五、固定参数敏感性（不挑最优替换主模型）",
              "同时将五个时间常数乘0.8/1.2；保持其他参数、划分、训练规则不变；仅重建对照。"]
    for _, r in sensitivity.iterrows():
        lines.append(f"  {r.variant:14s} {r.file}: R²={r.r2:+.4f}")
    lines += ["", "六、程序检查"]
    for _, r in checks.iterrows():
        lines.append(f"  {'PASS' if r.passed else 'FAIL'} {r['check']}: {r.value:.6g}")
    lines += ["  PASS 训练/测试互斥、每试次留出一次、快速置换与直接拟合等价、测试标签不进入本折预测。",
              "PASS只说明程序约束成立，不能代替模型效能或生理正确性的证据。",
              "", "七、能够回答的内容与保留界限",
              "现在有可执行的形状到神经群再到EEG的数学链条、数值解、消融和独立留出评分。",
              "这补充第二问的候选计算机制；是否解释实测左右差异必须由以上误差和分类结果判断。",
              "共同参数与镜像约束防止手动给左右类别设不同神经增益，但不保证候选网络就是真实机制。",
              "shape_blind有完全相同的镜像读出列，映射可能不可辨识；岭解可计算不代表各池增益可唯一解释。",
              "三导联/两被试不能支持唯一脑源反演、精确解剖定位、疾病诊断或人群推广。",
              "既有连续预处理/质量筛选在全记录上执行，本次仍以既定保留试次为条件；非全流程盲测。",
              "分块训练可以使用后续试次，不是严格未来预测；局部块置换也依赖块内可交换性假设。",
              "本步骤不增加逐试次行为正确答案，也不替代第三问认知指标的外部验证。",
              "", "参考（方程为受启发的简化模型，不是原文逐式复现）：",
              "Wilson & Cowan (1972), Biophysical Journal 12:1–24. DOI:10.1016/S0006-3495(72)86068-5.",
              "Azzopardi & Petkov (2014), Frontiers in Computational Neuroscience 8:80. DOI:10.3389/fncom.2014.00080.",
              "", f"耗时：{elapsed:.1f}秒；随机种子沿用08每项确定性置换序列。"]
    (out/"q2_shape_model_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--permutations", type=int, default=1999)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    require(args.permutations >= 19, "置换至少19次；正式结果使用默认1999次。")
    start = time.perf_counter()
    root = args.project_root.resolve()
    out = args.output_dir.resolve() if args.output_dir else root/"results"/"12_q2_shape_driven_model"
    out.mkdir(parents=True, exist_ok=True)
    v, p = import_validation(), Parameters()
    config = dict(parameters=asdict(p), n_permutations=args.permutations, exploratory=True,
                  python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  validation_script_sha256=hashlib.sha256(Path(v.__file__).read_bytes()).hexdigest(),
                  primary_model="shape_wc", block_size=20, purge_trials=2,
                  permutation_seed_rule="08.seed_for(evaluation); 08.SEED=20260924",
                  parameter_status="fixed candidate assumptions; not fitted biological parameters")
    # 在实测验证前保存配置，所有分析均从这一固定配置产生。
    (out/"run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    with threadpool_limits(limits=1):
        print("步骤1/4：构造镜像图形、共享空间特征与显式E/I动力学……", flush=True)
        spatial = spatial_inputs(p)
        simulations = {name: simulate(spatial, p, name) for name in MODEL_NAMES}
        checks = model_checks(spatial, simulations, p)
        v.save_csv(checks, out/"model_checks.csv")
        save_model(spatial, simulations, p, out, v)
        records = v.load_data(root)
        config["inputs"] = [{"file": r["name"], "sha256": r["sha256"], "n_trials": len(r["ids"])} for r in records]
        (out/"run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        splits, smooth, raw, manifest = prepare_records(records, v)
        v.save_csv(pd.DataFrame(manifest), out/"split_manifest.csv")
        print("步骤2/4：同一分块上的旧模型、新模型与消融比较……", flush=True)
        metrics, coeff, examples = reconstruction(records, splits, smooth, raw, simulations, v)
        summary = v.aggregate_reconstruction(metrics)
        v.save_csv(metrics, out/"reconstruction_by_fold.csv")
        v.save_csv(summary, out/"reconstruction_summary.csv")
        v.save_csv(coeff, out/"training_mapping_coefficients.csv")
        sensitivity = [summary[summary.model == "shape_wc"].copy()]
        for factor in (.8, 1.2):
            pp = replace(p, tau_g=p.tau_g*factor, tau_e=p.tau_e*factor, tau_i=p.tau_i*factor,
                         tau_readout_1=p.tau_readout_1*factor, tau_readout_2=p.tau_readout_2*factor)
            ss = simulate(spatial, pp)
            mm, _, _ = reconstruction(records, splits, smooth, raw, {"shape_wc": ss}, v)
            table = v.aggregate_reconstruction(mm)
            table = table[table.model == "shape_wc"].copy()
            table["variant"] = f"all_tau_x{factor}"
            sensitivity.append(table)
        sensitivity = pd.concat(sensitivity, ignore_index=True)
        v.save_csv(sensitivity, out/"time_constant_sensitivity.csv")
        print("步骤3/4：未知方向识别、跨被试验证与完整置换……", flush=True)
        classification = classify(records, splits, smooth, raw, simulations["shape_wc"], args.permutations, v, out)
        print("步骤4/4：生成图表与中文报告……", flush=True)
        plot_results(spatial, simulations, summary, classification, examples, records, out, v)
        write_report(records, p, checks, summary, classification, sensitivity,
                     time.perf_counter()-start, args.permutations, out)
    print("\n完成。报告：", out/"q2_shape_model_report.txt", flush=True)
    print(summary.pivot(index="file", columns="model", values="r2").round(4).to_string())
    print(classification[["evaluation", "balanced_accuracy", "auc", "permutation_p", "bh_fdr_q_7_tests"]].round(4).to_string(index=False))
    print("PASS不代表识别显著。请结合共享空间对照、反馈消融和FDR解释结果。")


if __name__ == "__main__":
    main()
