@echo off
setlocal

set PYTHON=%1
if "%PYTHON%"=="" set PYTHON=python

set ROOT=F:\APIN1
cd /d "%ROOT%"

set WINDOW_PATH=data\processed\windows_improved\1h\default\BTCUSDT_win48.parquet
set LABEL_PATH=data\labels_improved\default\BTCUSDT_labels.parquet

echo [1/2] Training BTC GRU with validation macro-F1 checkpoint selection...
%PYTHON% -m src.models.main.train_main ^
  --input_path "%WINDOW_PATH%" ^
  --label_path "%LABEL_PATH%" ^
  --output_dir "experiments\improved\main\gru_btc_checkpoint_macro" ^
  --model_type gru ^
  --checkpoint_selection valid_macro_f1
if errorlevel 1 exit /b 1

echo [2/2] Training BTC GRU with semantic-audit checkpoint selection...
%PYTHON% -m src.models.main.train_main ^
  --input_path "%WINDOW_PATH%" ^
  --label_path "%LABEL_PATH%" ^
  --output_dir "experiments\improved\main\gru_btc_checkpoint_semantic" ^
  --model_type gru ^
  --checkpoint_selection semantic_audit
if errorlevel 1 exit /b 1

echo [done] semantic checkpoint comparison runs finished.
