@echo off
REM ============================================================
REM 一键重新生成全部论文矢量图 (PDF)
REM 使用方法: 双击运行，或在命令行中 cd 到 F:\APIN1 后运行此脚本
REM ============================================================
setlocal enabledelayedexpansion

set ROOT=F:\APIN1
set PYTHON=python
set ERROR_COUNT=0

echo.
echo ========================================
echo  重新生成论文全图 (矢量PDF格式)
echo ========================================
echo.
echo [检查] Python ...
%PYTHON% --version 2>nul
if %errorlevel% neq 0 (
    echo [错误] 找不到 python，请先安装 Python 并加入 PATH
    pause
    exit /b 1
)

echo [检查] 依赖包 ...
%PYTHON% -c "import matplotlib, seaborn, pandas" 2>nul
if %errorlevel% neq 0 (
    echo [安装] 正在安装 matplotlib seaborn pandas ...
    pip install matplotlib seaborn pandas --break-system-packages 2>nul
    if %errorlevel% neq 0 (
        pip install matplotlib seaborn pandas 2>nul
    )
)

cd /d "%ROOT%"
if %errorlevel% neq 0 (
    echo [错误] 无法进入 F:\APIN1
    pause
    exit /b 1
)

REM ---- 清理旧PDF文件 ----
echo.
echo [清理] 删除旧 PDF ...
del "%ROOT%\figures\main\btc_*.pdf" 2>nul
del "%ROOT%\figures\main\framework_increment_*.pdf" 2>nul

REM ---- 1. plot_publication_figures (3张图) ----
echo.
echo ========================================
echo  [1/4] 主结果图: temporal backbone + TCN tradeoff + drawdown deconfounding
echo ========================================
%PYTHON% -m src.evaluation.plot_publication_figures --summary_dir experiments/summary --output_dir figures/main
if %errorlevel% neq 0 (
    echo [警告] plot_publication_figures 返回错误码 %errorlevel%
    set /a ERROR_COUNT+=1
) else (
    echo [完成] plot_publication_figures
)

REM ---- 2. plot_semantic_focus (3张图) ----
echo.
echo ========================================
echo  [2/4] 语义图: overall + return by period + risk by period
echo ========================================
%PYTHON% -m src.evaluation.plot_semantic_focus --summary_dir experiments/summary/btc_risk_state_semantics --output_dir figures/main
if %errorlevel% neq 0 (
    echo [警告] plot_semantic_focus 返回错误码 %errorlevel%
    set /a ERROR_COUNT+=1
) else (
    echo [完成] plot_semantic_focus
)

REM ---- 3. plot_significance_robustness (1张图) ----
echo.
echo ========================================
echo  [3/4] 显著性稳健性: nominal vs block bootstrap
echo ========================================
%PYTHON% -m src.evaluation.plot_significance_robustness --input_path experiments/revision/btc_risk_state_significance_with_histgb/risk_significance_summary.csv --output_dir figures/main
if %errorlevel% neq 0 (
    echo [警告] plot_significance_robustness 返回错误码 %errorlevel%
    set /a ERROR_COUNT+=1
) else (
    echo [完成] plot_significance_robustness
)

REM ---- 4. plot_framework_increment (1张图) ----
echo.
echo ========================================
echo  [4/4] 框架增量: semantic regret heatmap
echo ========================================
%PYTHON% -m src.evaluation.plot_framework_increment --input_path experiments/summary/evaluation_framework_increment/framework_increment_summary.csv --output_dir figures/main
if %errorlevel% neq 0 (
    echo [警告] plot_framework_increment 返回错误码 %errorlevel%
    set /a ERROR_COUNT+=1
) else (
    echo [完成] plot_framework_increment
)

REM ---- 验证 ----
echo.
echo ========================================
echo  验证生成的 PDF 文件
echo ========================================
set EXPECTED=8
set FOUND=0
for %%f in (
    "%ROOT%\figures\main\btc_temporal_backbone_comparison_extended.pdf"
    "%ROOT%\figures\main\btc_gru_vs_tcn96_tradeoff.pdf"
    "%ROOT%\figures\main\btc_drawdown_deconfounding_comparison.pdf"
    "%ROOT%\figures\main\btc_semantics_overall_main.pdf"
    "%ROOT%\figures\main\btc_return_semantics_by_period_main.pdf"
    "%ROOT%\figures\main\btc_risk_semantics_by_period_main.pdf"
    "%ROOT%\figures\main\btc_significance_nominal_vs_block_bootstrap.pdf"
    "%ROOT%\figures\main\framework_increment_semantic_regret.pdf"
) do (
    if exist %%f (
        echo   [OK] %%~nxf
        set /a FOUND+=1
    ) else (
        echo   [缺失] %%~nxf
    )
)

echo.
if !FOUND! equ !EXPECTED! (
    echo ========================================
    echo  全部 %EXPECTED% 张矢量图已生成！可以编译 LaTeX 了。
    echo ========================================
) else (
    echo ========================================
    echo  已生成 !FOUND! / %EXPECTED% 张图 (有 %error_LEVEL% 个错误)。
    echo ========================================
)

echo.
echo 图像已生成到 figures\main
echo.
pause
