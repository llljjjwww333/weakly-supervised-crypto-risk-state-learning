from __future__ import annotations

DEFAULT_FEATURE_COLUMNS = [
    "log_return_1",
    "log_return_4",
    "log_return_24",
    "rolling_vol_24",
    "rolling_vol_72",
    "high_low_range",
    "open_close_change",
    "volume_zscore_24",
    "quote_volume_zscore_24",
    "trade_count_zscore_24",
    "ema_gap_12_48",
    "ema_gap_24_72",
    "rolling_skew_24",
    "rolling_kurt_24",
    "up_bar_ratio_24",
    "down_bar_ratio_24",
    "max_drawdown_72",
    "trend_strength_24",
    "taker_buy_ratio",
    "volume_price_corr_24",
]


def parse_feature_list(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return values or None


def resolve_feature_columns(
    include_features: list[str] | None = None,
    exclude_features: list[str] | None = None,
) -> list[str]:
    if include_features is None:
        columns = list(DEFAULT_FEATURE_COLUMNS)
    else:
        unknown = [feature for feature in include_features if feature not in DEFAULT_FEATURE_COLUMNS]
        if unknown:
            raise ValueError(f"Unknown included features: {unknown}")
        columns = list(dict.fromkeys(include_features))

    if exclude_features:
        unknown = [feature for feature in exclude_features if feature not in DEFAULT_FEATURE_COLUMNS]
        if unknown:
            raise ValueError(f"Unknown excluded features: {unknown}")
        exclude_set = set(exclude_features)
        columns = [feature for feature in columns if feature not in exclude_set]

    if not columns:
        raise ValueError("Feature selection removed every feature.")
    return columns
