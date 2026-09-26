"""用真实试次验证行为标记未知，以及点击以后信号不会进入09的观测。"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]
path = ROOT / "src" / "09_q3_cognitive_joint_validation.py"
spec = importlib.util.spec_from_file_location("c_question_q3_test", path)
q3 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = q3
spec.loader.exec_module(q3)

events = pd.read_csv(ROOT / "data" / "processed" / "trial_info.csv")
assert len(events) == 400 and events.correct_inferred.isna().all()
assert (events.behavior_accuracy_status == "unknown_no_verified_ground_truth").all()
for archive in sorted((ROOT / "data" / "processed" / "erp_epochs").glob("*.npz")):
    with np.load(archive, allow_pickle=False) as a:
        assert len(a["all_trial_id"]) == 100
        assert (a["behavior_correct_inferred_all"] == -1).all()

q2 = q3.load_q2_module()
record = next(r for r in q2.load_data(ROOT) if r["name"] == "A_Task-2")
original, _ = q3.prepare_file(ROOT, "A_Task-2", record, q2)
case = original[0]
click = int(events[(events.file_name == "VisualCogA_Task-2.mat") &
                   (events.trial_id == case["trial_id"])].iloc[0].response_sample)
mat_path = ROOT / "data" / "raw" / "VisualCogA_Task-2.mat"
modified_mat = loadmat(mat_path)
modified_mat["data"] = modified_mat["data"].copy()
modified_mat["data"][:3, click:click+40] = 123456.0
real_load = q3.loadmat


def injected_loader(p):
    return modified_mat if Path(p).name == mat_path.name else real_load(p)


q3.loadmat = injected_loader
modified, _ = q3.prepare_file(ROOT, "A_Task-2", record, q2)
comparison = next(r for r in modified if r["trial_id"] == case["trial_id"])
assert np.array_equal(comparison["eeg"], case["eeg"])
assert np.array_equal(comparison["design"], case["design"])
print("PASS: 400 个试次的行为准确性保持未知；点击时及其后的人为脉冲不改变09的截止前试次。")
