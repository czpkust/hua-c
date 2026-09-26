"""C题问题2：对07的四阶段时间基函数模型做独立验证。

保存为 C:\\C\\src\\08_q2_mechanism_validation.py，在PyCharm中直接运行。
读取03生成的 data/processed/erp_epochs/*_erp_epochs.npz。
所有结果写入 results/08_q2_mechanism_validation/，不修改01—07的文件。

需要 numpy scipy pandas matplotlib scikit-learn（与06相同的环境）。
默认1999次置换，每次都重新训练分类器；普通CPU即可，不调用PyTorch/GPU。
命令行可选：--project-root C:\\C --permutations 4999

验证边界：
1. 07实际实现的是四组串联低通时间基函数，不含显式反馈回路。
   本脚本使用stage_1—stage_4，不能将它们当成实测LGN或皮层源。
2. 原试验编号每20次为一个连续测试块；训练集排除测试块及相邻2次试验。
   前面测试块的训练集可以包含后续试验，因此不是严格的未来预测。
3. 四种重建模型共同预测留出块的L/R平均波形；方向模型使用已知条件标签。
   识别未知方向的能力另用12维特征分类验证，禁止用重建R²代替分类证据。
4. 固定预处理、参数和筛选来自既有分析；这是探索性审计，不是盲测确认。
5. 仅两名被试。结果描述这两人的记录，不支持人群层面的泛化结论。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    import scipy
    from scipy.signal import lfilter, savgol_filter
    from scipy.stats import beta
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import sklearn
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
        accuracy_score, confusion_matrix,
    )
    from threadpoolctl import threadpool_limits
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"缺少依赖：{exc.name}。请在当前项目的Terminal中运行：\n"
        "python -m pip install numpy scipy pandas matplotlib scikit-learn threadpoolctl"
    ) from exc


SEED = 20260924
FILES = ("A_Task-1", "A_Task-2", "B_Task-1", "B_Task-2")
CHANNELS = ("Fz", "F3", "F4")
MODEL_NAMES = ("trend_shared", "trend_direction", "basis_shared", "basis_direction")
MODEL_LABELS = ("Trend / shared", "Trend / direction", "4 bases / shared", "4 bases / direction")
COLORS = ("#b3b3b3", "#727272", "#2b8cbe", "#df7a32")
BLOCK_SIZE = 20
PURGE_TRIALS = 2
FIT_WINDOW = (0.0, 0.90)
FEATURE_NAMES = tuple(f"stage_{k}/{ch}" for k in range(1, 5) for ch in CHANNELS)


@dataclass(frozen=True)
class Setting:
    name: str
    tau_scale: float = 1.0
    late_scale: float = 1.0
    ridge_alpha: float = 1.0
    linear_trend: bool = True
    smooth_s: float = 0.080


# 全部列出并报告，不根据测试成绩挑选最优参数。
SETTINGS = (
    Setting("nominal"),
    Setting("all_tau_x0.8", tau_scale=0.8),
    Setting("all_tau_x1.2", tau_scale=1.2),
    Setting("late_tau_x0.8", late_scale=0.8),
    Setting("late_tau_x1.2", late_scale=1.2),
    Setting("ridge_0.1", ridge_alpha=0.1),
    Setting("ridge_10", ridge_alpha=10.0),
    Setting("no_linear_trend", linear_trend=False),
    Setting("no_extra_smoothing", smooth_s=0.0),
)


def seed_for(name: str) -> int:
    return (SEED + zlib.crc32(name.encode("utf-8"))) % (2**32 - 1)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.9g")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_data(root: Path) -> list[dict]:
    records = []
    for name in FILES:
        path = root / "data" / "processed" / "erp_epochs" / f"{name}_erp_epochs.npz"
        if not path.is_file():
            raise FileNotFoundError(f"找不到：{path}\n请确认03已成功运行，且本脚本保存在项目src文件夹。")
        with np.load(path, allow_pickle=False) as a:
            e = np.asarray(a["epochs_accepted"], dtype=float)
            original_ids = np.asarray(a["accepted_trial_id"])
            ids = original_ids.astype(int)
            d = np.asarray(a["cue_direction_accepted"]).astype(str)
            t = np.asarray(a["time_s"], dtype=float)
            fs = float(np.asarray(a["sample_rate_hz"]).squeeze())
            ch = tuple(np.asarray(a["channel_labels"]).astype(str).tolist())
            band = np.asarray(a["filter_band_hz"], dtype=float)
        require(e.ndim == 3 and e.shape[1] == 3, f"{name}：片段必须是n×3×时间点。")
        require(ids.ndim == d.ndim == 1 and len(ids) == len(d) == len(e), f"{name}：索引长度不一致。")
        require(np.array_equal(ids, original_ids), f"{name}：试验编号必须为整数。")
        require(len(set(ids)) == len(ids) and np.all((ids >= 1) & (ids <= 100)), f"{name}：试验编号应唯一且在1—100内。")
        require(set(d) == {"L", "R"}, f"{name}：需要L和R两类提示。")
        require(ch == CHANNELS and np.isclose(fs, 256), f"{name}：需要Fz/F3/F4顺序、256 Hz。")
        require(band.shape == (2,) and np.allclose(band, [0.5, 30]), f"{name}：不是03的0.5—30 Hz数据。")
        require(t.ndim == 1 and e.shape[2] == len(t) and len(t) == 384, f"{name}：时间轴应为384点。")
        require(np.allclose(t, -0.5 + np.arange(384) / fs), f"{name}：时间轴不是[-0.5,1.0)秒。")
        require(np.isfinite(e).all(), f"{name}：仍有NaN或Inf。")
        order = np.argsort(ids)
        records.append(dict(
            name=name, subject=name[0], task=name[2:], epochs=e[order],
            ids=ids[order], directions=d[order], time=t, fs=fs, path=path,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ))
    return records


def make_block_splits(record: dict) -> tuple[list[tuple], list[dict]]:
    ids, d = record["ids"], record["directions"]
    splits, manifest = [], []
    coverage = np.zeros(len(ids), dtype=int)
    for block in range(5):
        low, high = 1 + BLOCK_SIZE * block, BLOCK_SIZE * (block + 1)
        te = np.flatnonzero((ids >= low) & (ids <= high))
        tr = np.flatnonzero((ids < low - PURGE_TRIALS) | (ids > high + PURGE_TRIALS))
        gap = np.setdiff1d(np.arange(len(ids)), np.concatenate([tr, te]))
        require(len(te) > 0 and not np.intersect1d(tr, te).size, "分块为空或训练/测试索引重叠。")
        require(np.min(np.abs(ids[tr, None] - ids[te][None, :])) > PURGE_TRIALS, "相邻试验排除失败。")
        for direction in ("L", "R"):
            require(np.sum(d[tr] == direction) >= 10, f"{record['name']}块{block+1}训练类{direction}不足10次。")
            require(np.sum(d[te] == direction) >= 2, f"{record['name']}块{block+1}测试类{direction}不足2次。")
        coverage[te] += 1
        splits.append((tr, te, f"block_{block + 1}"))
        manifest.append(dict(
            file=record["name"], fold=block+1, original_block_start=low,
            original_block_end=high, gap_trials=PURGE_TRIALS,
            train_ids=";".join(map(str, ids[tr])), test_ids=";".join(map(str, ids[te])),
            purged_ids=";".join(map(str, ids[gap])), n_train=len(tr), n_test=len(te),
            train_L=int(np.sum(d[tr] == "L")), train_R=int(np.sum(d[tr] == "R")),
            test_L=int(np.sum(d[te] == "L")), test_R=int(np.sum(d[te] == "R")),
        ))
    require(np.all(coverage == 1), "每个保留试验必须恰好作为一次测试试验。")
    return splits, manifest


def build_design(t: np.ndarray, fs: float, setting: Setting) -> dict:
    x = np.zeros_like(t)
    x[np.argmin(np.abs(t - 0.035))] = fs
    states = []
    taus = np.array([0.015, 0.035, 0.050, 0.150]) * setting.tau_scale
    taus[-1] *= setting.late_scale
    for tau, order in zip(taus, (2, 3, 4, 3)):
        a = np.exp(-1.0 / (fs * tau))
        for _ in range(order):
            x = lfilter([1.0 - a], [1.0, -a], x)
        # 后续环节使用未归一化状态；单位峰值仅用于设计矩阵。
        states.append(x / np.max(np.abs(x)))
    states = np.column_stack(states)
    mask = (t >= FIT_WINDOW[0]) & (t <= FIT_WINDOW[1])
    tm = t[mask]
    nuisance = [np.ones(len(tm))]
    if setting.linear_trend:
        nuisance.append((tm - tm.mean()) / (np.ptp(tm) / 2))
    trend = np.column_stack(nuisance)
    full = np.column_stack([states[mask], trend])
    penalty = np.diag([setting.ridge_alpha]*4 + [0.0]*trend.shape[1])
    projection = np.linalg.solve(full.T @ full + penalty, full.T)
    return dict(mask=mask, time=tm, full=full, trend=trend, projection=projection,
                trend_projection=np.linalg.pinv(trend),
                peaks=t[np.argmax(states, axis=0)], condition_number=float(np.linalg.cond(full)))


def fit_single_trials(record: dict, design: dict, setting: Setting) -> dict:
    original = record["epochs"]
    smooth = original
    if setting.smooth_s > 0:
        window = int(round(setting.smooth_s * record["fs"]))
        window += int(window % 2 == 0)
        smooth = savgol_filter(original, window_length=max(window, 5), polyorder=3, axis=-1)
    fit_data = smooth[:, :, design["mask"]]
    coeff = np.einsum("pt,nct->npc", design["projection"], fit_data)
    trend_coeff = np.einsum("pt,nct->npc", design["trend_projection"], fit_data)
    features = coeff[:, :4, :].reshape(len(coeff), 12)
    require(np.isfinite(features).all(), f"{record['name']}：12维特征含非有限值。")
    # 所有参数组合共同预测03输出的波形，测试目标不随08的平滑参数改变。
    target = original[:, :, design["mask"]].transpose(0, 2, 1)
    return dict(coeff=coeff, trend_coeff=trend_coeff, features=features, target=target)


def score_curve(observed: np.ndarray, predicted: np.ndarray) -> dict:
    sse = float(np.sum((observed - predicted)**2))
    sst = float(np.sum((observed - observed.mean(axis=0, keepdims=True))**2))
    r2 = 1.0 - sse/sst if sst > 1e-20 else np.nan
    a, b = observed.ravel(), predicted.ravel()
    correlation = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 1e-12 and np.std(b) > 1e-12 else np.nan
    return dict(sse=sse, sst=sst, n_values=observed.size, r2=r2,
                rmse=float(np.sqrt(sse/observed.size)), correlation=correlation)


def validate_reconstruction(record: dict, design: dict, fitted: dict, splits: list, setting: Setting) -> tuple:
    rows, examples = [], {}
    d = record["directions"]
    for fold, (tr, te, _) in enumerate(splits, 1):
        examples[fold] = {}
        for direction in ("L", "R"):
            te_d = te[d[te] == direction]
            observed = fitted["target"][te_d].mean(axis=0)
            predictions = {}
            for model in MODEL_NAMES:
                train_idx = tr[d[tr] == direction] if model.endswith("_direction") else tr
                is_basis = model.startswith("basis")
                coefs = fitted["coeff"] if is_basis else fitted["trend_coeff"]
                matrix = design["full"] if is_basis else design["trend"]
                predicted = matrix @ coefs[train_idx].mean(axis=0)
                predictions[model] = predicted
                for ch_index, channel in enumerate((*CHANNELS, "joint")):
                    sl = slice(None) if channel == "joint" else slice(ch_index, ch_index+1)
                    rows.append(dict(
                        variant=setting.name, file=record["name"], fold=fold,
                        direction=direction, model=model, channel=channel,
                        n_train=len(train_idx), n_test=len(te_d),
                        **score_curve(observed[:, sl], predicted[:, sl]),
                    ))
            examples[fold][direction] = dict(observed=observed, predictions=predictions, n=len(te_d))
    return rows, examples


def aggregate_reconstruction(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    joint = metrics[metrics["channel"] == "joint"]
    for (variant, file, model), g in joint.groupby(["variant", "file", "model"], sort=False):
        sse, sst, nv = g["sse"].sum(), g["sst"].sum(), g["n_values"].sum()
        rows.append(dict(variant=variant, file=file, model=model,
                         r2=1.0-sse/sst if sst > 1e-20 else np.nan,
                         rmse=np.sqrt(sse/nv), sse=sse, sst=sst,
                         n_values=nv, n_condition_blocks=len(g)))
    return pd.DataFrame(rows)


def paired_gains(summary: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        ("basis_added_shared", "basis_shared", "trend_shared"),
        ("direction_added_basis", "basis_direction", "basis_shared"),
        ("basis_added_direction", "basis_direction", "trend_direction"),
    )
    rows = []
    for (variant, file), group in summary.groupby(["variant", "file"], sort=False):
        g = group.set_index("model")
        for name, extended, reference in comparisons:
            rows.append(dict(variant=variant, file=file, comparison=name,
                             extended_model=extended, reference_model=reference,
                             delta_r2=g.loc[extended, "r2"]-g.loc[reference, "r2"],
                             sse_reduction_pct=100*(1-g.loc[extended, "sse"]/g.loc[reference, "sse"])))
    return pd.DataFrame(rows)


def hedges_g(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    df = len(x)+len(y)-2
    pooled = ((len(x)-1)*x.var(axis=0, ddof=1)+(len(y)-1)*y.var(axis=0, ddof=1))/df
    return (1-3/(4*df-1))*(y.mean(axis=0)-x.mean(axis=0))/np.sqrt(np.maximum(pooled, 1e-20))


def feature_effects(record: dict, fitted: dict, setting: Setting) -> list[dict]:
    d, x = record["directions"], fitted["features"]
    effects = hedges_g(x[d == "L"], x[d == "R"])
    return [dict(variant=setting.name, file=record["name"], feature=feature, hedges_g_R_minus_L=float(g))
            for feature, g in zip(FEATURE_NAMES, effects)]


def effect_stability(effects: pd.DataFrame) -> pd.DataFrame:
    p = effects.pivot(index=["file", "feature"], columns="variant", values="hedges_g_R_minus_L")
    reference = p["nominal"].to_numpy()
    rows = []
    for setting in SETTINGS:
        values = p[setting.name].to_numpy()
        away_from_zero = np.abs(reference) >= 0.2
        rows.append(dict(
            variant=setting.name, n_effects=len(reference),
            correlation=float(np.corrcoef(reference, values)[0, 1]),
            sign_agreement=float(np.mean(np.sign(reference) == np.sign(values))),
            n_nominal_abs_g_ge_0p2=int(away_from_zero.sum()),
            sign_agreement_abs_g_ge_0p2=float(np.mean(
                np.sign(reference[away_from_zero]) == np.sign(values[away_from_zero]))) if away_from_zero.any() else np.nan,
        ))
    return pd.DataFrame(rows)


def classification_metrics(y: np.ndarray, pred: np.ndarray, prob: np.ndarray) -> dict:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return dict(
        n=len(y), balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        accuracy=float(accuracy_score(y, pred)), auc=float(roc_auc_score(y, prob)),
        mcc=float(matthews_corrcoef(y, pred)),
        true_L_pred_L=int(cm[0, 0]), true_L_pred_R=int(cm[0, 1]),
        true_R_pred_L=int(cm[1, 0]), true_R_pred_R=int(cm[1, 1]),
    )


def out_of_fold_prediction(x: np.ndarray, y: np.ndarray, splits: list) -> tuple:
    pred = np.full(len(y), -1, dtype=int)
    prob = np.full(len(y), np.nan)
    fold_names = np.empty(len(y), dtype=object)
    for tr, te, label in splits:
        # scaler和分类器每折都创建/拟合；任何训练统计量均不使用测试数据。
        estimator = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", solver="liblinear",
                               max_iter=2000, random_state=SEED),
        )
        estimator.fit(x[tr], y[tr])
        prob[te] = estimator.predict_proba(x[te])[:, 1]
        pred[te] = estimator.predict(x[te])
        fold_names[te] = label
    require(np.all(pred >= 0) and np.isfinite(prob).all(), "分类测试覆盖不完整。")
    return pred, prob, fold_names


def fdr_bh(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    adj = values[order]*len(values)/np.arange(1, len(values)+1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty_like(adj)
    out[order] = np.clip(adj, 0, 1)
    return out


def validate_classification(records: list, fitted: dict, local_splits: dict, permutations: int, out: Path) -> pd.DataFrame:
    metadata = pd.concat([
        pd.DataFrame(dict(file=r["name"], subject=r["subject"], task=r["task"],
                          trial_id=r["ids"], direction=r["directions"]))
        for r in records
    ], ignore_index=True)
    all_x = np.concatenate([fitted[r["name"]]["features"] for r in records])
    y_all = (metadata["direction"].to_numpy() == "R").astype(int)
    feature_table = pd.concat([metadata, pd.DataFrame(all_x, columns=FEATURE_NAMES)], axis=1)
    save_csv(feature_table, out / "mechanism_12d_features.csv")
    evaluations = []
    for r in records:
        sel = np.flatnonzero(metadata["file"].to_numpy() == r["name"])
        evaluations.append((f"within_{r['name']}", "within_record", sel, local_splits[r["name"]]))
    for task in ("Task-1", "Task-2", "Pooled"):
        sel = np.arange(len(metadata)) if task == "Pooled" else np.flatnonzero(metadata["task"].to_numpy() == task)
        subjects = metadata.iloc[sel]["subject"].to_numpy()
        splits = []
        for subject in ("A", "B"):
            tr, te = np.flatnonzero(subjects != subject), np.flatnonzero(subjects == subject)
            require(not set(subjects[tr]).intersection(subjects[te]), "跨被试训练/测试混入同一人。")
            splits.append((tr, te, f"{'B' if subject == 'A' else 'A'}_to_{subject}"))
        evaluations.append((f"cross_{task}", "cross_subject", sel, splits))

    summary, folds, predictions, null_rows = [], [], [], []
    for name, scope, sel, splits in evaluations:
        print(f"  {name}：拟合及{permutations}次完整置换……", flush=True)
        x, y = all_x[sel], y_all[sel]
        meta = metadata.iloc[sel].reset_index(drop=True)
        coverage = np.zeros(len(y), dtype=int)
        for tr, te, _ in splits:
            require(not np.intersect1d(tr, te).size, "分类训练/测试交叉。")
            coverage[te] += 1
        require(np.all(coverage == 1), "分类每个试验应恰好预测一次。")
        pred, prob, fold_names = out_of_fold_prediction(x, y, splits)
        observed = classification_metrics(y, pred, prob)
        for _, te, label in splits:
            folds.append(dict(evaluation=name, scope=scope, fold=label,
                              **classification_metrics(y[te], pred[te], prob[te])))
        result = meta.copy()
        result["evaluation"], result["scope"] = name, scope
        result["fold"] = fold_names
        result["predicted_direction"] = np.where(pred == 1, "R", "L")
        result["probability_R"] = prob
        predictions.append(result)

        # 原始20次试验块内置换，保留各文件/局部块的类别计数。
        groups = [np.asarray(indices) for indices in meta.groupby(
            ["file", (meta["trial_id"]-1)//BLOCK_SIZE], sort=True).indices.values()]
        rng = np.random.default_rng(seed_for(name))
        null = np.empty(permutations)
        for k in range(permutations):
            yp = y.copy()
            for group in groups:
                yp[group] = rng.permutation(y[group])
            pp, _, _ = out_of_fold_prediction(x, yp, splits)
            null[k] = balanced_accuracy_score(yp, pp)
            if (k+1) % 100 == 0 or k+1 == permutations:
                print(f"    置换 {k+1}/{permutations}", flush=True)
        exceedances = int(np.sum(null >= observed["balanced_accuracy"]-1e-12))
        p_value = (1+exceedances)/(permutations+1)
        # 对置换尾概率的点态Clopper-Pearson区间；不是模型准确率的置信区间。
        mc_low = float(beta.ppf(0.025, exceedances, permutations-exceedances+1)) if exceedances else 0.0
        mc_high = float(beta.ppf(0.975, exceedances+1, permutations-exceedances)) if exceedances < permutations else 1.0
        summary.append(dict(evaluation=name, scope=scope, n_features=12, n_folds=len(splits),
                            **observed, permutation_p=float(p_value), n_permutations=permutations,
                            null_exceedances=exceedances, p_mc_95_low=mc_low, p_mc_95_high=mc_high,
                            null_ba_median=float(np.median(null)),
                            null_ba_p025=float(np.quantile(null, 0.025)),
                            null_ba_p975=float(np.quantile(null, 0.975))))
        null_rows.extend(dict(evaluation=name, permutation=i+1, balanced_accuracy=float(v)) for i, v in enumerate(null))
    summary_df = pd.DataFrame(summary)
    summary_df["bh_fdr_q_7_tests"] = fdr_bh(summary_df["permutation_p"].to_numpy())
    save_csv(summary_df, out / "mechanism_classification_summary.csv")
    save_csv(pd.DataFrame(folds), out / "classification_folds.csv")
    save_csv(pd.concat(predictions, ignore_index=True), out / "classification_predictions.csv")
    save_csv(pd.DataFrame(null_rows), out / "permutation_null_scores.csv")
    return summary_df


def save_figure(fig: plt.Figure, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_models(summary: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.8), layout="constrained")
    x = np.arange(4)
    for j, model in enumerate(MODEL_NAMES):
        values = summary[summary["model"] == model].set_index("file").loc[list(FILES), "r2"].to_numpy()
        bars = ax.bar(x+(j-1.5)*0.2, values, width=0.19, label=MODEL_LABELS[j], color=COLORS[j])
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f.replace("_", " ") for f in FILES])
    ax.set_ylabel("Held-out block-mean R-squared")
    ax.set_title("Chronological block validation: four matched models")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, fontsize=9)
    ax.margins(y=0.2)
    ax.grid(axis="y", alpha=0.2)
    fig.supxlabel("R-squared = 1 - total SSE / total SST; all five test blocks; no population confidence intervals", fontsize=9)
    save_figure(fig, out, "blocked_model_comparison.png")


def plot_fit_examples(examples: dict, design: dict, out: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), sharex=True, layout="constrained")
    for row, file in enumerate(FILES):
        case = examples[file][5]  # 固定展示最后一个测试块，不能挑拟合最好的一折。
        for ci, ch in enumerate(CHANNELS):
            ax = axes[row, ci]
            texts = []
            for direction, color in (("L", "#3366cc"), ("R", "#dd482f")):
                entry = case[direction]
                obs = entry["observed"][:, ci]
                pred = entry["predictions"]["basis_direction"][:, ci]
                ax.plot(design["time"], obs, color=color, lw=1.3, label=f"{direction} test mean (n={entry['n']})")
                ax.plot(design["time"], pred, color=color, lw=1.8, ls="--", label=f"{direction} direction model")
                r2 = score_curve(obs[:, None], pred[:, None])["r2"]
                texts.append(f"{direction} R2={r2:.2f}")
            shared = case["L"]["predictions"]["basis_shared"][:, ci]
            ax.plot(design["time"], shared, color="#555555", ls=":", lw=1.5, label="Shared model")
            ax.text(0.02, 0.98, "; ".join(texts), transform=ax.transAxes, va="top", fontsize=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
            ax.axhline(0, color="grey", lw=0.6)
            ax.grid(alpha=0.18)
            if row == 0:
                ax.set_title(ch)
            if ci == 0:
                ax.set_ylabel(file.replace("_", " ")+"\nAmplitude (input units)")
            if row == 3:
                ax.set_xlabel("Time from cue onset (s)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # 每文件样本数不同，图例只展示线型；各面板n另用行首标注。
    labels = ["L held-out mean", "L direction model", "R held-out mean", "R direction model", "Shared model"]
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=9)
    for row, file in enumerate(FILES):
        case = examples[file][5]
        axes[row, 0].text(0.02, 0.89, f"Test n: L={case['L']['n']}, R={case['R']['n']}",
                         transform=axes[row, 0].transAxes, va="top", fontsize=8)
    fig.suptitle("Fixed illustration: original trials 81-100 held out\nTrained on accepted trials 1-78; R2 shown separately for each channel", fontsize=14)
    save_figure(fig, out, "blocked_fit_curves.png")


def plot_sensitivity(summary: pd.DataFrame, gains: pd.DataFrame, out: Path) -> None:
    names = [s.name for s in SETTINGS]
    a = summary[summary["model"] == "basis_shared"].pivot(index="file", columns="variant", values="r2").loc[list(FILES), names]
    b = gains[gains["comparison"] == "direction_added_basis"].pivot(index="file", columns="variant", values="delta_r2").loc[list(FILES), names]
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), layout="constrained")
    for ax, data, title in zip(axes, (a, b), ("Shared four-basis model: held-out R2", "Added direction parameters: change in held-out R2")):
        limit = max(float(np.nanmax(np.abs(data.to_numpy()))), 0.01)
        im = ax.imshow(data, cmap="RdBu_r", vmin=-limit, vmax=limit, aspect="auto")
        for i in range(4):
            for j in range(len(names)):
                v = float(data.iloc[i, j])
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(v) > 0.6*limit else "black")
        ax.set_xticks(np.arange(len(names)), names, rotation=25, ha="right", fontsize=9)
        ax.set_yticks(np.arange(4), [f.replace("_", " ") for f in FILES])
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.9)
    fig.suptitle("Fixed-parameter sensitivity: same splits and same test waveforms for every setting", fontsize=14)
    save_figure(fig, out, "parameter_sensitivity.png")


def plot_classification(summary: pd.DataFrame, out: Path) -> None:
    folds = pd.read_csv(out / "classification_folds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout="constrained")
    for ax, scope, title in zip(axes, ("within_record", "cross_subject"), ("Within-record held-out blocks", "Bidirectional cross-subject tests")):
        df = summary[summary["scope"] == scope].reset_index(drop=True)
        x = np.arange(len(df))
        ax.bar(x, df["balanced_accuracy"], color="#4685ac", width=0.55)
        ax.axhline(0.5, color="#555555", ls="--", label="Chance reference = 0.5")
        for j, row in df.iterrows():
            vals = folds[folds["evaluation"] == row["evaluation"]]["balanced_accuracy"].to_numpy()
            ax.scatter(j+np.linspace(-0.16, 0.16, len(vals)), vals, s=28, facecolors="white", edgecolors="black", zorder=3)
            ax.text(j, 0.94, f"BA={row['balanced_accuracy']:.3f}\np={row['permutation_p']:.3f}\nq={row['bh_fdr_q_7_tests']:.3f}",
                    ha="center", va="top", fontsize=9)
        labels = df["evaluation"].str.replace("within_", "", regex=False).str.replace("cross_", "", regex=False).str.replace("_", " ", regex=False)
        ax.set_xticks(x, labels, fontsize=9)
        ax.set_ylim(0, 1.02)
        ax.set_title(title)
        ax.set_ylabel("Balanced accuracy")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Independent 12-feature classifier: scaler and classifier refitted in every permutation", fontsize=13)
    fig.supxlabel("Bars: combined held-out predictions. Dots: individual folds (not confidence intervals). q: BH-FDR over 7 tests.", fontsize=9)
    save_figure(fig, out, "mechanism_classification.png")


def write_report(records: list, nominal: pd.DataFrame, gains: pd.DataFrame,
                 classification: pd.DataFrame, sensitivity: pd.DataFrame,
                 stability: pd.DataFrame, design: dict, permutations: int, elapsed: float, out: Path) -> None:
    lines = [
        "C题问题2：07四阶段时间基函数模型验证报告", "="*64,
        "性质：探索性验证。PASS只表示程序/数据检查通过，不表示模型正确或统计显著。",
        "", "一、输入与验证范围",
        f"文件4个；03保留试验共{sum(len(r['ids']) for r in records)}次。",
    ]
    for r in records:
        lines.append(f"  {r['name']}: n={len(r['ids'])}, L={np.sum(r['directions']=='L')}, R={np.sum(r['directions']=='R')}")
    lines += [
        "只使用03输出的Fz、F3、F4；不读取Decon、点击事件或行为指标作为预测特征。",
        "固定基函数的系数来自单试验，不跨试验计算，不使用方向标签。",
        "原编号1—20、21—40、41—60、61—80、81—100分别作为测试块。",
        "训练集为其余块，但测试块两侧各2个原始编号的试验额外排除。",
        "每个保留试验恰好测试一次；前四折可用后续试验训练，不属于严格未来预测。",
        "仅两名被试，不能将356次试验当作356名独立被试。",
        "03的连续滤波和质量筛选已在全记录上完成；本验证以这份既定保留数据为条件，",
        "不是从原始记录开始全流程独立的盲测，2次试验间隔也不能保证消除所有时间依赖。",
        "", "二、同一测试目标上的四种重建模型",
        "trend_shared：截距+线性趋势，L/R共享训练均值系数。",
        "trend_direction：同样趋势模型，但L/R分别估计训练均值系数。",
        "basis_shared：4条固定基函数+截距+线性趋势，L/R共享系数。",
        "basis_direction：同样4条基函数模型，但L/R分别估计系数。",
        "nominal与07使用同一基函数和岭参数；本报告统一使用算术均值，不混用Huber汇总。",
        "训练试验默认额外做21点三阶SG平滑；所有设置的测试目标始终是03输出波形的算术均值。",
        "R²=1-各测试块/方向/通道SSE之和÷对应SST之和；SST按各测试条件和通道中心化。",
        "R²可以为负，意味着不及用测试条件的时间均值作解释基准；不截断负值。",
        "相关系数仅辅助描述形状；不能代替误差或R²，未对重叠训练折做独立样本显著性检验。",
        "注意：测试均值包含已知方向标签，因此条件重建不是未知方向分类。", "",
    ]
    for file in FILES:
        g = nominal[nominal["file"] == file].set_index("model")
        lines.append(file)
        for model in MODEL_NAMES:
            lines.append(f"  {model:18s}: R²={g.loc[model,'r2']:+.4f}, RMSE={g.loc[model,'rmse']:.4f}")
        h = gains[(gains["variant"] == "nominal") & (gains["file"] == file)].set_index("comparison")
        lines.append(f"  共享基函数相对共享趋势增益：ΔR²={h.loc['basis_added_shared','delta_r2']:+.4f}")
        lines.append(f"  方向参数相对共享基函数增益：ΔR²={h.loc['direction_added_basis','delta_r2']:+.4f}")
    positive_basis = int((gains.query("variant == 'nominal' and comparison == 'basis_added_shared'")["delta_r2"] > 0).sum())
    positive_direction = int((gains.query("variant == 'nominal' and comparison == 'direction_added_basis'")["delta_r2"] > 0).sum())
    lines += [
        "", f"名义设置下，基函数的共享重建增益为正：{positive_basis}/4份记录；方向参数增益为正：{positive_direction}/4。",
        "这些是预测误差比较，不是方向效应的显著性判据。",
        "blocked_fit_curves.png固定展示最后一个测试块81—100；训练试验为1—78中的保留试验。",
        "图中的R²按每个通道分别计算；全5折结果见blocked_reconstruction_metrics.csv。",
        "", "三、真正独立的12维方向分类",
        "12维=4条时间基函数×3个通道的系数；不含截距/线性趋势，不使用06的139维分类结果。",
        "固定分类器：训练集StandardScaler + LogisticRegression(C=0.1, class_weight=balanced)。",
        "脚本固定列定4个记录内分块评估、Task-1/Task-2/合并任务的3个双向跨被试评估。",
        "每折重新拟合标准化和分类器，没有使用测试成绩选特征、调C或选择参数组合。",
        "跨被试合并成绩由A→B和B→A的全部留出预测计算；分方向成绩见classification_folds.csv。",
        f"{permutations}次标签置换：每文件的每20原始试验块内打乱L/R，每次重跑全部训练/测试折。",
        "p=(1+置换BA≥观察BA的次数)/(1+置换次数)，7个BA检验统一BH-FDR。",
        "置换以块内标签可交换为假设，保留局部类别数；未声称完全排除序列相关。",
        "置换零分布的分位数不是模型准确率的置信区间。", "",
    ]
    for row in classification.to_dict("records"):
        lines.append(f"  {row['evaluation']}: BA={row['balanced_accuracy']:.4f}, AUC={row['auc']:.4f}, p={row['permutation_p']:.4f}, q={row['bh_fdr_q_7_tests']:.4f}")
        lines.append(f"    超越计数={int(row['null_exceedances'])}/{permutations}；置换尾概率的蒙特卡洛95%区间=[{row['p_mc_95_low']:.4f}, {row['p_mc_95_high']:.4f}]")
    significant = classification[classification["bh_fdr_q_7_tests"] < 0.05]
    if significant.empty:
        lines.append("结论：上述7个探索性分类检验均未达到BH-FDR q<0.05，当前验证未建立可靠方向识别证据。")
    else:
        lines.append("q<0.05的探索性评估："+", ".join(significant["evaluation"])+"；仍需独立数据确认。")
    lines += [
        "上述区间只量化有限次置换造成的抽样误差，不是准确率、q值或人群泛化能力的区间。",
        "临界p/q值不宜作二元定论；正式结论需结合效应大小、稳定性及独立验证。",
        "不显著不等于证明不存在方向信息；不得通过反复换参数只报告最高分类成绩。",
        "q只校正本脚本的7个检验，未覆盖之前所有探索与模型尝试。",
        "", "四、固定参数敏感性",
        "同时报告名义设置、全部时间常数×0.8/1.2、末阶段时间常数×0.8/1.2、",
        "岭参数0.1/10、去线性趋势、去08额外平滑，共9种；不选择测试最优者作为新主模型。",
        "所有组合保持相同测试块、剔除规则和测试波形，因此误差可直接比较。",
        "除名义设置外未重新搜索分类器；敏感性用于重建误差和48个方向效应的描述。",
    ]
    for file in FILES:
        shared = sensitivity[(sensitivity["file"] == file) & (sensitivity["model"] == "basis_shared")]["r2"]
        delta = gains[(gains["file"] == file) & (gains["comparison"] == "direction_added_basis")]["delta_r2"]
        lines.append(f"  {file}: 共享R²范围[{shared.min():+.4f}, {shared.max():+.4f}]，方向ΔR²范围[{delta.min():+.4f}, {delta.max():+.4f}]")
    lines.append("48个效应量=4文件×12系数；Hedges g为R减L，仅描述，不据此筛选分类特征：")
    for row in stability.to_dict("records"):
        lines.append(f"  {row['variant']}: r={row['correlation']:.3f}, 符号一致率={row['sign_agreement']:.1%}")
    lines += [
        "接近0的效应易变号；另报告名义|g|≥0.2的符号一致率，但该阈值不是显著性阈值。",
        "", "五、对生理机制解释的限制",
        "07的滤波器逐级串联，没有实现显式反馈状态或双皮层相互作用；本脚本也没有新增该机制。",
        "stage_1—stage_4是预设时间响应，不能凭三个头皮通道唯一识别LGN、具体皮层源或连接强度。",
        "名义基函数峰时(ms)："+", ".join(f"{x*1000:.1f}" for x in design["peaks"])+"，由设定常数决定，并非本实验测得的传导潜伏期。",
        f"名义设计矩阵条件数={design['condition_number']:.2f}；基函数相关时，单个系数解释尤其需要谨慎。",
        "本程序只能回答该受约束时间模型的预测表现与参数敏感性，不能证明某种生理机制成立。",
        "报告中使用输入幅值单位，不在缺少原始单位确认时自动写成微伏。",
        "", "六、程序与数据检查",
        "PASS  仅载入03的Fz/F3/F4保留片段，256 Hz，0.5—30 Hz。",
        "PASS  试验编号唯一；全部波形及12维特征有限；L/R标签合法。",
        "PASS  分块内每个试验恰好测试一次；训练/测试不重叠，邻近2次试验排除。",
        "PASS  跨被试每折训练与测试被试完全分离。",
        "PASS  分类器标准化仅在每折训练集拟合；置换中重新训练。",
        "PASS  9种敏感性设置测试目标一致；所有设置均报告。",
        "PASS  本程序不写入data目录，也不改写01—07的结果。",
        "试验总数与先前356次记录"+("一致。" if sum(len(r["ids"]) for r in records) == 356 else "不同，请核实03是否重新运行过。"),
        f"耗时：{elapsed:.1f}秒；随机种子：{SEED}。完整参数/版本/输入SHA256见run_config.json。",
        "", "实现参考（官方文档）",
        "https://scikit-learn.org/stable/common_pitfalls.html",
        "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.permutation_test_score.html",
        "注：本脚本自定义局部块内置换与固定折，未直接调用permutation_test_score。",
    ]
    (out / "q2_mechanism_validation_report.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="C题问题2：四阶段时间基函数模型独立验证")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--permutations", type=int, default=1999)
    args = parser.parse_args()
    require(args.permutations >= 99, "置换至少99次；正式报告建议默认1999次或更多。")
    started = time.perf_counter()
    root = args.project_root.expanduser().resolve()
    out = root / "results" / "08_q2_mechanism_validation"
    print(f"项目路径：{root}\n输出路径：{out}", flush=True)
    records = load_data(root)
    out.mkdir(parents=True, exist_ok=True)
    print(f"已读取{len(records)}份记录，共{sum(len(r['ids']) for r in records)}次保留试验。", flush=True)
    local_splits, manifests = {}, []
    for r in records:
        splits, rows = make_block_splits(r)
        local_splits[r["name"]] = splits
        manifests.extend(rows)
    save_csv(pd.DataFrame(manifests), out / "split_manifest.csv")

    all_metrics, effects, nominal_fitted, examples = [], [], {}, {}
    nominal_design = None
    with threadpool_limits(limits=1):
        print("步骤1/3：四模型分块对照及9种参数敏感性……", flush=True)
        for setting in SETTINGS:
            print(f"  参数组合：{setting.name}", flush=True)
            for record in records:
                design = build_design(record["time"], record["fs"], setting)
                fitted = fit_single_trials(record, design, setting)
                rows, example = validate_reconstruction(record, design, fitted, local_splits[record["name"]], setting)
                all_metrics.extend(rows)
                effects.extend(feature_effects(record, fitted, setting))
                if setting.name == "nominal":
                    nominal_design = design
                    nominal_fitted[record["name"]] = fitted
                    examples[record["name"]] = example
        metrics = pd.DataFrame(all_metrics)
        require(len(metrics) == 9*4*5*2*4*4, "重建评估行数不符。")
        summary = aggregate_reconstruction(metrics)
        nominal = summary[summary["variant"] == "nominal"].copy()
        gains = paired_gains(summary)
        effects_df = pd.DataFrame(effects)
        stability = effect_stability(effects_df)
        save_csv(metrics[metrics["variant"] == "nominal"], out / "blocked_reconstruction_metrics.csv")
        save_csv(nominal, out / "model_comparison.csv")
        save_csv(gains[gains["variant"] == "nominal"], out / "paired_model_gains.csv")
        save_csv(summary, out / "sensitivity_summary.csv")
        save_csv(gains, out / "sensitivity_gains.csv")
        save_csv(effects_df, out / "sensitivity_effects.csv")
        save_csv(stability, out / "sensitivity_feature_stability.csv")
        print("步骤2/3：12维特征分类与完整重新训练的置换检验……", flush=True)
        classification = validate_classification(records, nominal_fitted, local_splits, args.permutations, out)

    print("步骤3/3：生成图表和中文报告……", flush=True)
    plot_models(nominal, out)
    plot_fit_examples(examples, nominal_design, out)
    plot_sensitivity(summary, gains, out)
    plot_classification(classification, out)
    elapsed = time.perf_counter()-started
    config = dict(
        script="08_q2_mechanism_validation.py", seed=SEED,
        project_root=str(root), n_permutations=args.permutations,
        elapsed_seconds=elapsed, fit_window_s=FIT_WINDOW, block_size=BLOCK_SIZE,
        purge_original_trials_each_side=PURGE_TRIALS,
        basis_delay_s=0.035, nominal_taus_s=[0.015, 0.035, 0.050, 0.150],
        filter_orders=[2, 3, 4, 3], basis_normalization="unit peak on full epoch",
        settings=[asdict(s) for s in SETTINGS], feature_names=FEATURE_NAMES,
        targets="03 preprocessed epochs; no extra smoothing of validation targets",
        labels="cue direction: L=0, R=1", classifier="StandardScaler + LogisticRegression",
        classifier_parameters=dict(C=0.1, class_weight="balanced", solver="liblinear", max_iter=2000),
        permutation="shuffle labels within each file and original 20-trial block; refit all folds",
        fdr_family="7 prespecified-in-this-script balanced accuracy tests; prior explorations excluded",
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                      pandas=pd.__version__, matplotlib=matplotlib.__version__, sklearn=sklearn.__version__),
        inputs=[dict(file=str(r["path"]), sha256=r["sha256"], accepted_n=len(r["ids"])) for r in records],
    )
    (out / "run_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    write_report(records, nominal, gains, classification, summary, stability,
                 nominal_design, args.permutations, elapsed, out)
    print("\n验证完成。", flush=True)
    print(f"中文报告：{out / 'q2_mechanism_validation_report.txt'}")
    print(f"四模型对照：{out / 'blocked_model_comparison.png'}")
    print(f"12维分类检验：{out / 'mechanism_classification.png'}")
    print(f"参数敏感性：{out / 'parameter_sensitivity.png'}")
    print(f"耗时：{elapsed:.1f}秒。请先读报告；PASS并不代表分类显著。")


if __name__ == "__main__":
    main()
