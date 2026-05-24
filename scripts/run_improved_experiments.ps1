param(
    [string]$Python = "python",
    [string]$Root = "D:\APIN"
)

$ErrorActionPreference = "Stop"
Set-Location $Root

$featureRoot = "data\processed\features\1h"
$labelRoot = "data\labels_improved"
$windowRoot = "data\processed\windows_improved\1h"
$experimentRoot = "experiments\improved"

# 1. Weak labels: default and drawdown-free variants.
& $Python -m src.features.build_labels `
  --input_dir $featureRoot `
  --output_dir "$labelRoot\default"

& $Python -m src.features.build_labels `
  --input_dir $featureRoot `
  --output_dir "$labelRoot\no_drawdown_rule" `
  --disable_drawdown_filter

# 2. Windows: default overlap, stride-48 overlap control, and no-drawdown-feature variants.
& $Python -m src.features.build_windows `
  --input_dir "$labelRoot\default" `
  --output_dir "$windowRoot\default" `
  --window 48 `
  --stride 1

& $Python -m src.features.build_windows `
  --input_dir "$labelRoot\default" `
  --output_dir "$windowRoot\stride48" `
  --window 48 `
  --stride 48

& $Python -m src.features.build_windows `
  --input_dir "$labelRoot\default" `
  --output_dir "$windowRoot\no_drawdown_feature" `
  --window 48 `
  --stride 1 `
  --exclude_features max_drawdown_72

# 3. Main GRU reruns with the revised imbalance handling.
& $Python -m src.models.main.train_main `
  --input_path "$windowRoot\default\BTCUSDT_win48.parquet" `
  --output_dir "$experimentRoot\main\gru_btc_empirical_balance" `
  --model_type gru `
  --balance_mode empirical `
  --ce_class_weight_mode inverse_freq

& $Python -m src.models.main.train_main `
  --input_path "$windowRoot\default\BTCUSDT_win48.parquet" `
  --output_dir "$experimentRoot\main\gru_btc_no_continuity_empirical_balance" `
  --model_type gru `
  --continuity_weight 0 `
  --balance_mode empirical `
  --ce_class_weight_mode inverse_freq

& $Python -m src.models.main.train_main `
  --input_path "$windowRoot\stride48\BTCUSDT_win48_s48.parquet" `
  --output_dir "$experimentRoot\main\gru_btc_stride48_no_continuity" `
  --model_type gru `
  --continuity_weight 0 `
  --balance_mode empirical `
  --ce_class_weight_mode inverse_freq

& $Python -m src.models.main.train_main `
  --input_path "$windowRoot\no_drawdown_feature\BTCUSDT_win48.parquet" `
  --output_dir "$experimentRoot\main\gru_btc_no_drawdown_feature" `
  --model_type gru `
  --balance_mode empirical `
  --ce_class_weight_mode inverse_freq

# 4. Static baselines for the drawdown deconfounding check.
& $Python -m src.models.baselines.run_logreg `
  --input_path "$labelRoot\default\BTCUSDT_labels.parquet" `
  --output_dir "$experimentRoot\baselines\logreg_btc_no_drawdown_feature" `
  --exclude_features max_drawdown_72

& $Python -m src.models.baselines.run_hmm `
  --input_path "$labelRoot\default\BTCUSDT_labels.parquet" `
  --output_dir "$experimentRoot\baselines\hmm_btc_no_drawdown_feature" `
  --exclude_features max_drawdown_72
