# 第13步：视觉—记忆维持—目标后再激活的联动验证

本补丁直接连接第12步的形状驱动视觉网络，增加两个候选记忆痕迹和两个目标后再激活状态，检验它们能否改善应答前EEG的留出预测。代码、参考结果和中文报告均已提供。

本次运行的主要结论是：**新增状态没有稳定改善预测，不能写成已经验证海马或正确/错误认知机制。** 这是一套可以复现和检验的候选模型，模型结构比原09更完整，实证支持仍然有限。

## 在自己的电脑上运行

将 `C题_问题三_认知联动模型补丁.zip` 保存到“下载”文件夹，在现有 `(pytorch)` PowerShell 中依次执行：

```powershell
Expand-Archive -LiteralPath "$env:USERPROFILE\Downloads\C题_问题三_认知联动模型补丁.zip" -DestinationPath 'C:\C\C题联合验证' -Force
cd 'C:\C\C题联合验证'
python -X utf8 .\src\13_q3_shape_memory_validation.py
```

运行完打开报告：

```powershell
notepad .\results\13_q3_shape_memory_validation\q3_shape_memory_report.txt
```

也可显示在终端：

```powershell
Get-Content .\results\13_q3_shape_memory_validation\q3_shape_memory_report.txt -Encoding UTF8
```

沿用现有依赖和CPU环境，不需要安装新包或重新跑01–12。本步骤新增 `src/13_q3_shape_memory_validation.py`，运行结果写入 `results/13_q3_shape_memory_validation`。

压缩包内 `reference_results/13_q3_shape_memory_validation` 是此次参考运行结果；你自己电脑的新输出在 `results` 目录。补丁不包含MAT数据，不覆盖01–12的代码或输出。

前置文件：

- `src/08_q2_mechanism_validation.py`、`09_q3_cognitive_joint_validation.py`、`12_q2_shape_driven_model.py`。
- `data/processed/trial_info.csv` 和03生成的4份ERP片段。
- `data/raw/VisualCogA_Task-2.mat`、`VisualCogB_Task-2.mat`。
- `results/12_q2_shape_driven_model/run_config.json`、`simulation_shape_wc.npz`。

如果你修改过12的脚本，却没有重跑12，13会提示其脚本与模拟输出版本不一致。这时应重新运行12，不能删除版本检查来绕过依赖。

## 数据窗口

继续沿用09的窗口及筛选：从提示前0.5秒开始，只读到真实±2点击前约0.1秒；在这段截断数据内执行0.5–30 Hz带通，再做基线校正。滤波不访问截止点及以后的数据。

Task-2最终使用A的88次、B的89次，共177次试验。Task-1没有独立点击标记，因此不伪造其应答前100 ms终点。选定窗口仍可能包含较早的运动准备活动。

真实点击时刻只用于离线设定窗口终点；模型不使用点击方向、目标标记正负号、所谓“正确率”来估计脑电。这个任务是已知事件时刻的条件脑电预测，并非实时提前预测点击。

## 与第12步怎样连接

直接读取12保存的神经群状态，保持三角形空间特征、36组E/I网络参数、6个提示视觉功能池不变。代码核对12脚本与其输出的指纹。

目标输入使用相同神经网络，仅把外部输入延长到3秒；将两个图形的响应平均为不带方向的目标探测输入。所有实际分析的目标后窗口均短于固定模拟长度。这是目标持续显示的候选简化，不代表已知真实双目标的逐像素布局。

由固定图形原型对视觉群体响应加权，得到两路编码输入 `u_L(t), u_R(t)`，再建立：

```text
tau_M dM_j/dt = -M_j + u_j(t)
tau_C dC_j/dt = -C_j + M_j(t) g_target(t),  j = L, R
```

其中 `M` 表示提示形状的候选维持痕迹，`C` 表示目标门控的候选再激活。固定 `tau_M=1.5 s`、`tau_C=0.25 s`，这些是模型假设，不是从三导联辨识出的真实记忆时间常数。

脑电观测模型为：

```text
EEG = 截距 + 线性趋势
    + 6个提示视觉池的线性混合
    + 2个目标状态的线性混合
    + 2个记忆痕迹M的线性混合
    + 2个再激活状态C的线性混合
```

潜在观测分量与EEG接受相同的截止点内滤波；混合权重仅由训练试次估计。各折由名义模型的训练数据计算一套RMS缩放，并对所有对照和敏感性设置复用，使“缩短记忆时间导致衰减”不会被每个对照独立归一化抹去。

这构成视觉输入和认知候选状态之间的聚合动力学链条。由于实际目标图形与正确答案日志缺失，此次只检验维持与再激活，没有把它冒充真实形状匹配判断、错误决策或海马定位。

## 对照与评价

| 模型 | 组成 |
| --- | --- |
| `trend` | 训练截距与线性趋势 |
| `shape_visual` | 再加6个提示视觉池 |
| `shape_target` | 再加2个目标状态 |
| `memory_only` | 再加2个维持状态M |
| `shape_memory` | 再加2个再激活状态C，固定主模型 |
| `memory_blind` | 将记忆编码取方向平均，视觉提示输入保持原样 |
| `short_memory` | 记忆时间常数缩短到0.15秒，其余不变 |

沿用原编号20次一块的5折，训练额外排除测试块两侧2次；另执行A训练B测试、B训练A测试。参数不根据测试成绩选取。

主评价区间为目标出现后50 ms至应答前截止点。同时报告早期视觉段、等待段和整个应答前窗口。长窗口包含更多采样点，对拟合与汇总SSE的权重相应更大。

主指标包括RMSE和误差减少比例。报告中 `skill_vs_train_trend = 1 - SSE_model / SSE_trend` 是相对可训练趋势基线的误差改进分数，**不是普通R²**。CSV另列按试次/通道时间均值中心化的描述性R²，它也不能与12条件平均波形的R²直接比较。

只有两名被试，训练折还互相重叠，因此不将五折当成五名独立被试来做显著性检验。本步骤没有声称记忆效应“显著”，也没有将打乱方向对应关系当作独立行为验证。

## 参考运行结果

下面比较完整记忆模型 `shape_memory` 与视觉加目标基线 `shape_target`。**SSE减少为正表示改善，为负表示误差增加。**

| 验证 | 测试记录 | SSE减少 | 改善试次数 |
| --- | --- | ---: | ---: |
| A记录内留块 | A项目2 | −0.3496% | 32/88 |
| B记录内留块 | B项目2 | +0.0109% | 48/89 |
| B训练→A测试 | A项目2 | −0.5540% | 39/88 |
| A训练→B测试 | B项目2 | −0.0615% | 52/89 |

记录内只有1/2份有微小改善，跨被试两个方向都没有改善。改善试次多于一半也不保证总误差减少，因为每次误差变化幅度不同。

这组结果支持保留候选模型与消融过程，但不支持“新增记忆环节稳定提高预测”或“已验证海马来源”。记忆时间常数取1.2、1.5、1.8秒的全部敏感性结果都保存了，没有挑选最优结果替换主模型。

本轮程序检查已通过：

- 与09相同的试次窗口、筛选和训练趋势基线。
- 大幅修改截止点后的原始数据，预处理结果不变。
- 改变测试试次脑电和设计矩阵，不影响本折训练参数或缩放。
- 改变目标编码和点击方向，不影响候选模型设计。
- 记忆输入满足镜像约束，未滤波的再激活状态在目标出现前为零。

`PASS`只表示这些约束成立，不代表模型机制得到实证确认。

## 输出文件

| 文件 | 内容 |
| --- | --- |
| `q3_shape_memory_report.txt` | 中文模型说明、结果、限制 |
| `model_comparison.csv` | 所有模型、窗口、验证方向的汇总 |
| `memory_increment.csv` | 记忆、再激活、方向信息、维持时长的对照 |
| `heldout_trial_errors.csv` | 逐试次留出误差 |
| `memory_time_sensitivity.csv` | 记忆时间常数敏感性 |
| `training_coefficients.csv` | 训练折混合系数、RMS和有效自由度 |
| `trial_windows.csv`、`split_manifest.csv` | 截取窗口和训练/测试划分 |
| `validation_checks.csv` | 程序检查结果 |
| `legacy09_same_trials.csv` | 同一试次上复算09的结果 |
| `cognitive_model_comparison.png` | 主窗口各模型与趋势基线的比较 |
| `memory_increment_by_phase.png` | 不同阶段的新增记忆误差变化 |
| `heldout_preclick_examples.png` | 固定展示最后测试块的首个保留试次 |
| `candidate_state_dynamics.png` | 模拟维持与再激活时程，不是真实脑源图 |
| `fixed_example_states.npz` | 示例潜在状态数值 |
| `run_config.json`、`eligibility.json` | 固定配置、输入指纹、排除原因 |

## 复现后如何组织三问答案

| 问题 | 可以提供的内容 | 仍不能声称的内容 |
| --- | --- | --- |
| 第一问 | 预处理、响应拟合、方向检验、注入回收和信号强度诊断 | 所有伪影已去除、真实图形差异已可靠保留 |
| 第二问 | 图形→空间输入→E/I网络→EEG的候选机制、方向特征和留出验证 | 可靠左右识别、真实LGN或皮层源已定位 |
| 第三问 | 与第二问连接的维持/再激活模型、应答前窗口、消融与跨被试比较 | 海马机制已确认、正确/错误/遗漏应答已验证 |

本步复现后可以转入三问模型方程、结果图表及结论范围的整理。现有计算链条已经可执行，但“提出了候选模型”和“模型解释得到验证”应分开表达，不能把程序跑通等同于题目所有要求都已获得实证支持。

## 参考背景

- Daume et al. (2024), *Control of working memory by phase–amplitude coupling of human hippocampal neurons*, Nature 629:393–401. [作者机构公开页面](https://authors.library.caltech.edu/records/es8hd-ftd63)。该研究包含人类单神经元及theta–gamma耦合证据；本补丁没有复现该机制，也不把额区EEG当作海马单神经元记录。
- Compte et al. (2000), *Synaptic Mechanisms and Network Dynamics Underlying Spatial Working Memory in a Cortical Network Model*, Cerebral Cortex 10:910–923. [论文](https://doi.org/10.1093/cercor/10.9.910)。本补丁借鉴维持状态的建模动机，使用的是简化漏泄状态模型，并非原文完整持续活动网络。
- 视觉网络方程和参考依据见第12步报告。
