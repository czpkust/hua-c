# 问题一信号强度诊断

本补丁依赖已经安装并运行的 `src/10_q1_injection_recovery.py` 和01、03、04的文件。把ZIP内容解压到现有项目根目录 `C:\C\C题联合验证`，然后运行：

```powershell
cd 'C:\C\C题联合验证'
python -X utf8 .\src\11_q1_signal_strength.py
Get-Content .\results\11_q1_signal_strength\q1_signal_strength_report.txt -Encoding UTF8
```

脚本保持预处理不变，完整比较0.4、0.8、1.6、3.2倍基线标准差的合成注入强度，使用相同随机分组。0.8倍必须复现已有10号结果，否则程序停止并提示检查版本和输入。结果写入新目录 `results/11_q1_signal_strength`。

ZIP附带参考报告、逐轮数据、汇总表、无背景形状失真表和强度曲线。提高合成幅值是诊断方法，不代表真实数据质量得到改善，也不能替代真实左右方向的验证。这个实验是根据10号结果开展的探索性补充，不能写作独立数据验证。
