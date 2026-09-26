# 问题一补充验证的运行方法

本补丁依赖原有 `C:\C\C题联合验证` 中已经完成的 01、03、04 步及四份 MAT 数据。只增加 `src/10_q1_injection_recovery.py`，结果写到 `results/10_q1_injection_recovery/`，不会重跑 08 的置换检验。

将本 ZIP 的内容解压到 `C:\C\C题联合验证` 后，打开已经激活 `pytorch` 的 PowerShell：

```powershell
cd 'C:\C\C题联合验证'
python -X utf8 .\src\10_q1_injection_recovery.py
Get-Content .\results\10_q1_injection_recovery\q1_injection_report.txt -Encoding UTF8
```

输出包括逐轮指标 `injection_repetitions.csv`、汇总 `injection_summary.csv`、中文报告 `q1_injection_report.txt`、复现参数 `run_config.json` 和两张图。ZIP 内也附有我们试跑时生成的参考结果；在你的电脑运行后将按本地 MAT 重新生成。

解读时，NRMSE 越小越好，NRMSE 超过 1 表示单次合成方向对比的恢复误差大于已知对比自身的能量。试跑中当前流程虽降低了相对于原始试次算术均值的误差，但四份数据的 NRMSE 中位数仍大于 1，不足以证明真实左右三角形可分。设备自带 Decon 不支持重新注入同一模拟响应；不能用其既有输出假装同等条件的注入实验。
