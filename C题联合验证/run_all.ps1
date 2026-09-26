# 在已激活的 Python 环境中依次运行 C 题的 01–09 阶段；任何阶段失败即停止。
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $steps = @(
        '01_parse_trials.py',
        '02_raw_qc.py',
        '03_preprocess_erp.py',
        '04_erp_curves.py',
        '05_sensitivity_analysis.py',
        '06_q2_feature_model.py',
        '07_q2_mechanism_model.py',
        '08_q2_mechanism_validation.py',
        '09_q3_cognitive_joint_validation.py'
    )
    foreach ($step in $steps) {
        Write-Host "正在运行 $step ..."
        & python -X utf8 (Join-Path 'src' $step)
        if ($LASTEXITCODE -ne 0) { throw "$step 运行失败，退出码 $LASTEXITCODE" }
    }
    Write-Host '全部运行完成。请阅读 results/08_* 和 results/09_* 下的报告。'
} finally {
    Pop-Location
}
