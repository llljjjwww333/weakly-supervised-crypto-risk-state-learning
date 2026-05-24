@echo off
setlocal

set PYTHON=%1
if "%PYTHON%"=="" set PYTHON=python

set ROOT=F:\APIN1
cd /d "%ROOT%"

set WINDOW_PATH=data\processed\windows_improved\1h\default\BTCUSDT_win48.parquet
set LABEL_PATH=data\labels_improved\default\BTCUSDT_labels.parquet

echo [1/1] Training BTC Transformer baseline...
%PYTHON% -m src.models.main.train_main ^
  --input_path "%WINDOW_PATH%" ^
  --label_path "%LABEL_PATH%" ^
  --output_dir "experiments\improved\main\transformer_btc" ^
  --model_type transformer ^
  --hidden_dim 64 ^
  --num_layers 2 ^
  --dropout 0.1 ^
  --transformer_heads 4 ^
  --transformer_ff_dim 256 ^
  --load_batch_size 128 ^
  --checkpoint_selection valid_macro_f1
if errorlevel 1 exit /b 1

echo [done] BTC Transformer run finished.
