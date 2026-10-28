"""Configuration for the Kalshi BTC 15-minute strategy.

Every tunable number in the spec lives here so the backtester, the paper
trader and (eventually) live trading all read the same values.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

# Live order routing needs BOTH the config flag and this environment variable.
LIVE_TRADING_ENV_VAR = "KALSHI_LIVE_TRADING"
LIVE_TRADING_ENV_VALUE = "I_ACCEPT_REAL_MONEY_RISK"


@dataclass(frozen=True)
class EntryConfig:
    """Asymmetrical Bollinger entry model, computed on 1-second BTC closes."""

    fast_ma: int = 50            # MA50
    slow_ma: int = 250           # MA250 (also the Bollinger middle)
    long_ma: int = 500           # MA500 (computed and logged, not used in the entry rule)
    band_sd_mult: float = 2.0    # Upper/Lower = MA250 +/- 2 * SD250
    breach_z: float = 2.0        # lower breach when Z <= -2, upper breach when Z >= +2
    breach_window_s: int = 120   # golden cross must occur within this many seconds of a lower breach
    yes_price_min: float = 0.30
    yes_price_max: float = 0.70
    # Which Kalshi price the 0.30-0.70 filter is applied to: "ask" (what you would pay),
    # "last" (last trade, as in the Oct 26 exploration) or "mid".
    yes_price_field: str = "ask"
    min_seconds_remaining: int = 1   # contract must not be expired at the signal
    ddof: int = 0                    # 0 = population SD (standard Bollinger), 1 = sample SD


@dataclass(frozen=True)
class ExitMethod:
    """One exit rule. kind="hold" holds to settlement; kind="bollinger" uses a trailing stop.

    The trailing stop runs on `candle_seconds` candles with a `lookback`-candle
    Bollinger Band. It activates once BTC >= upper band (if require_activation),
    then exits after BTC closes below the middle band for `confirm_seconds`
    consecutively.
    """

    name: str
    kind: str = "bollinger"
    candle_seconds: int = 5
    lookback: int = 20
    sd_mult: float = 2.0
    confirm_seconds: int = 60
    require_activation: bool = True


# The four methods compared in section 7 of the spec. The "original" 250-second
# exit is a middle-band exit on 1-second closes; it is not gated on touching the
# upper band (set require_activation=True in a config file to gate it).
DEFAULT_EXIT_METHODS: tuple[ExitMethod, ...] = (
    ExitMethod(name="hold_to_settlement", kind="hold"),
    ExitMethod(name="orig_250s_mid_30s", candle_seconds=1, lookback=250, confirm_seconds=30,
               require_activation=False),
    ExitMethod(name="bb20x5s_60s", candle_seconds=5, lookback=20, confirm_seconds=60),
    ExitMethod(name="bb20x1m_60s", candle_seconds=60, lookback=20, confirm_seconds=60),
)


@dataclass(frozen=True)
class TradingConfig:
    quantity: int = 100                  # contracts per signal (100 for the historical test)
    max_open_positions: int = 1          # simultaneous open positions per strategy engine
    max_entries_per_contract: int = 1    # duplicate-entry guard per Kalshi market
    entry_slippage: float = 0.02         # buy limit = signal-time price + this
    latency_s: int = 1                   # orders fill against the quote this many seconds later
    quote_max_age_s: int = 10            # ignore Kalshi quotes older than this
    allow_partial_fills: bool = True     # buy what the book offers if depth < quantity
    fee_rate: float = 0.07               # Kalshi taker fee: ceil(rate * C * P * (1 - P)) to the cent
    live_exit_method: str = "bb20x5s_60s"
    live_trading_enabled: bool = False   # must stay False until explicitly enabled


@dataclass(frozen=True)
class SettlementConfig:
    window_s: int = 60          # BRTI rolling average over the final 60 seconds
    # YES if avg(index over last window) >= strike (">=") or > strike (">").
    # Confirm against the contract rules of the series you trade.
    yes_if: str = ">="


@dataclass(frozen=True)
class KalshiConfig:
    api_base: str = "https://api.elections.kalshi.com/trade-api/v2"
    series_ticker: str = "KXBTC15M"
    orderbook_depth: int = 10
    timeout_s: float = 0.8


@dataclass(frozen=True)
class Config:
    entry: EntryConfig = field(default_factory=EntryConfig)
    exit_methods: tuple[ExitMethod, ...] = DEFAULT_EXIT_METHODS
    trading: TradingConfig = field(default_factory=TradingConfig)
    settlement: SettlementConfig = field(default_factory=SettlementConfig)
    kalshi: KalshiConfig = field(default_factory=KalshiConfig)

    def exit_method(self, name: str) -> ExitMethod:
        for m in self.exit_methods:
            if m.name == name:
                return m
        raise KeyError(f"unknown exit method {name!r}; have {[m.name for m in self.exit_methods]}")


def _apply(dc, overrides: dict):
    names = {f.name for f in fields(dc)}
    unknown = set(overrides) - names
    if unknown:
        raise ValueError(f"unknown {type(dc).__name__} keys: {sorted(unknown)}")
    return replace(dc, **overrides)


def load_config(path: str | Path | None = None) -> Config:
    """Load a TOML config; anything not set keeps its default."""
    cfg = Config()
    if path is None:
        return cfg
    data = tomllib.loads(Path(path).read_text())
    kwargs = {}
    for section, cls_default in (("entry", cfg.entry), ("trading", cfg.trading),
                                 ("settlement", cfg.settlement), ("kalshi", cfg.kalshi)):
        if section in data:
            kwargs[section] = _apply(cls_default, data[section])
    if "exit_methods" in data:
        kwargs["exit_methods"] = tuple(ExitMethod(**m) for m in data["exit_methods"])
    return replace(cfg, **kwargs)


def live_trading_allowed(cfg: Config) -> bool:
    """True only when the config flag AND the environment variable are both set."""
    return (cfg.trading.live_trading_enabled
            and os.environ.get(LIVE_TRADING_ENV_VAR) == LIVE_TRADING_ENV_VALUE)
