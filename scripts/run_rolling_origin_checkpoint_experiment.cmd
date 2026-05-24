@echo off
setlocal

set PYTHON=%1
if "%PYTHON%"=="" set PYTHON=python

set ROOT=F:\APIN1
cd /d "%ROOT%"

set WINDOW_PATH=data\processed\windows_improved\1h\default\BTCUSDT_win48.parquet
set LABEL_PATH=data\labels_improved\default\BTCUSDT_labels.parquet

echo [window 2024][1/2] macro checkpoint selection...
%PYTHON% -m src.models.main.train_main ^
  --input_path "%WINDOW_PATH%" ^
  --label_path "%LABEL_PATH%" ^
  --output_dir "experiments\improved\main\gru_btc_roll2024_macro" ^
  --model_type gru ^
  --train_end 2022-12-31 ^
  --valid_end 2023-12-31 ^
  --test_end 2024-12-31 ^
  --checkpoint_selection valid_macro_f1
if errorlevel 1 exit /b 1

echo [window 2024][2/2] semantic checkpoint selection...
%PYTHON% -m src.models.main.train_main ^
  --input_path "%WINDOW_PATH%" ^
  --label_path "%LABEL_PATH%" ^
  --output_dir "experiments\improved\main\gru_btc_roll2024_semantic" ^
  --model_type gru ^
  --train_end 2022-12-31 ^
  --valid_end 2023-12-31 ^
  --test_end 2024-12-31 ^
  --checkpoint_selection semantic_audit
if errorlevel 1 exit /b 1

echo [done] rolling-origin 2024 checkpoint comparison finished.
