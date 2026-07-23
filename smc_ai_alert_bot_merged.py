"""
smc_ai_alert_bot_merged.py
---------------------------
Single-file merge of: smc_core.py + setup_detector.py + snapshot.py + smc_ai_alert_bot.py

Standalone bot: OKX (public OHLCV, no key needed) -> SMC structure engine (4h bias +
15m timing) -> setup detector -> chart snapshot -> Claude-written rationale ->
Telegram alert.

Env vars required:
    TELEGRAM_BOT_TOKEN
    TELEGRAM_CHAT_ID
    ANTHROPIC_API_KEY
Optional:
    SMC_SYMBOL            (default "BTC/USDT:USDT" -- OKX perp ccxt symbol)
    SMC_POLL_SECONDS       (default 900 -- how often to check for new setups)
    SMC_STATE_DB           (default "smc_alert_state.sqlite")

Merged from originals:
    - smc_core.py       -- LuxAlgo-style SMC structure engine (BOS/CHoCH, OBs, FVGs, EQH/EQL)
    - setup_detector.py -- premium/discount fade + OB/FVG continuation setup logic
    - snapshot.py       -- mplfinance chart rendering for alerts
    - smc_ai_alert_bot.py -- runner: polls OKX, detects setups, calls Claude, sends Telegram alert
"""

# ============================================================
# Shared imports (deduplicated across all merged modules)
# ============================================================
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional
import os
import time
import sqlite3
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import ccxt
import requests
import mplfinance as mpf
import matplotlib.pyplot as plt


# ============================================================
# SECTION 1: smc_core.py  (SMC structure engine)
# ============================================================


class Bias(IntEnum):
    BEARISH = -1
    NONE = 0
    BULLISH = 1


class Leg(IntEnum):
    BEARISH_LEG = 0
    BULLISH_LEG = 1


@dataclass
class SMCConfig:
    swing_length: int = 50          # bars used to confirm a swing pivot
    internal_length: int = 5        # bars used to confirm an internal pivot
    eqhl_length: int = 3            # bars used to confirm equal high/low
    eqhl_threshold: float = 0.1     # ATR fraction tolerance for equal high/low
    atr_period: int = 200
    ob_max_stored: int = 100        # how many order blocks to retain in history
    ob_mitigation: str = "highlow"  # "close" or "highlow"
    fvg_auto_threshold: bool = True
    premium_discount_band: float = 0.05  # fraction of range for premium/discount/equilibrium bands


@dataclass
class Pivot:
    current_level: float = np.nan
    last_level: float = np.nan
    crossed: bool = False
    bar_time: pd.Timestamp = None
    bar_index: int = None


@dataclass
class OrderBlock:
    bar_high: float
    bar_low: float
    bar_time: pd.Timestamp
    bias: int   # Bias.BULLISH / Bias.BEARISH
    mitigated: bool = False


@dataclass
class FVG:
    top: float
    bottom: float
    bias: int
    bar_time: pd.Timestamp
    filled: bool = False


@dataclass
class PremiumDiscountZones:
    """Mirrors drawPremiumDiscountZones: top/bottom bands sized as a fraction of the
    trailing swing range, plus a band centered on the midpoint (equilibrium)."""
    range_top: float
    range_bottom: float
    premium_bottom: float     # premium zone spans [premium_bottom, range_top]
    discount_top: float       # discount zone spans [range_bottom, discount_top]
    equilibrium_mid: float
    equilibrium_top: float
    equilibrium_bottom: float


@dataclass
class StructureEvent:
    bar_index: int
    bar_time: pd.Timestamp
    scope: str          # "swing" or "internal"
    kind: str            # "BOS" or "CHoCH"
    direction: int        # Bias.BULLISH / Bias.BEARISH
    level: float


@dataclass
class SMCState:
    """Everything the engine needs to keep across bars — mirrors the Pine `var` state."""
    swing_high: Pivot = field(default_factory=Pivot)
    swing_low: Pivot = field(default_factory=Pivot)
    internal_high: Pivot = field(default_factory=Pivot)
    internal_low: Pivot = field(default_factory=Pivot)
    equal_high: Pivot = field(default_factory=Pivot)
    equal_low: Pivot = field(default_factory=Pivot)
    swing_trend: int = Bias.NONE
    internal_trend: int = Bias.NONE
    trailing_top: float = -np.inf
    trailing_bottom: float = np.inf
    trailing_top_time: pd.Timestamp = None
    trailing_bottom_time: pd.Timestamp = None
    swing_obs: list = field(default_factory=list)
    internal_obs: list = field(default_factory=list)
    fvgs: list = field(default_factory=list)
    events: list = field(default_factory=list)
    equal_highs: list = field(default_factory=list)
    equal_lows: list = field(default_factory=list)


class SMCEngine:
    """
    Feed it an OHLCV dataframe (columns: open, high, low, close, indexed by UTC time)
    and it returns a fully-annotated copy plus a SMCState with live order blocks / FVGs
    / structure events, matching what LuxAlgo would show on the last closed candle.
    """

    def __init__(self, config: SMCConfig = None):
        self.cfg = config or SMCConfig()

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _atr(df: pd.DataFrame, period: int) -> pd.Series:
        high, low, close = df["high"], df["low"], df["close"]
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    def _legs(self, df: pd.DataFrame, size: int) -> np.ndarray:
        """
        Reproduces `leg(size)`: a bar starts a new bullish leg once a low `size`
        bars back breaks below the rolling `size`-bar lowest, and a new bearish leg
        once a high `size` bars back breaks above the rolling `size`-bar highest.
        """
        high, low = df["high"].values, df["low"].values
        n = len(df)
        roll_high = df["high"].rolling(size).max().values
        roll_low = df["low"].rolling(size).min().values

        legs = np.zeros(n, dtype=int)
        cur = Leg.BEARISH_LEG
        for i in range(n):
            if i < size or np.isnan(roll_high[i]) or np.isnan(roll_low[i]):
                legs[i] = cur
                continue
            new_leg_high = high[i - size] > roll_high[i]
            new_leg_low = low[i - size] < roll_low[i]
            if new_leg_high:
                cur = Leg.BEARISH_LEG
            elif new_leg_low:
                cur = Leg.BULLISH_LEG
            legs[i] = cur
        return legs

    # -------------------------------------------------------- structure logic
    def _update_pivot_at_bar(self, df, legs, size, i, state: SMCState,
                              equal_high_low=False, internal=False):
        """
        Mirrors one call of getCurrentStructure(size, equalHighLow, internal) for bar i.
        Must be called in bar order (i increasing) so pivot state reflects only bars
        up to and including i -- this is what makes it behave like Pine's per-bar model.
        """
        if i < size or legs[i] == legs[i - 1]:
            return  # no new leg started at this bar
        idx = df.index
        high, low = df["high"].values, df["low"].values
        pivot_low = legs[i] == Leg.BULLISH_LEG
        src_i = i - size  # the confirmed pivot bar

        if pivot_low:
            pivot = state.equal_low if equal_high_low else (
                state.internal_low if internal else state.swing_low)
            level = low[src_i]
            if equal_high_low and not np.isnan(pivot.current_level):
                atr_i = self._atr_cache[i]
                if abs(pivot.current_level - level) < self.cfg.eqhl_threshold * atr_i:
                    state.equal_lows.append((idx[src_i], level))
            pivot.last_level = pivot.current_level
            pivot.current_level = level
            pivot.crossed = False
            pivot.bar_time = idx[src_i]
            pivot.bar_index = src_i
            if not equal_high_low and not internal:
                state.trailing_bottom = level
                state.trailing_bottom_time = idx[src_i]
        else:
            pivot = state.equal_high if equal_high_low else (
                state.internal_high if internal else state.swing_high)
            level = high[src_i]
            if equal_high_low and not np.isnan(pivot.current_level):
                atr_i = self._atr_cache[i]
                if abs(pivot.current_level - level) < self.cfg.eqhl_threshold * atr_i:
                    state.equal_highs.append((idx[src_i], level))
            pivot.last_level = pivot.current_level
            pivot.current_level = level
            pivot.crossed = False
            pivot.bar_time = idx[src_i]
            pivot.bar_index = src_i
            if not equal_high_low and not internal:
                state.trailing_top = level
                state.trailing_top_time = idx[src_i]

    def _store_order_block(self, df, pivot: Pivot, up_to_i: int, internal: bool,
                            bias: int, state: SMCState):
        lo, hi = pivot.bar_index, up_to_i
        window = df.iloc[lo:hi + 1]
        if window.empty:
            return
        if bias == Bias.BEARISH:
            ob_bar = window["high"].idxmax()
        else:
            ob_bar = window["low"].idxmin()
        row = df.loc[ob_bar]
        ob = OrderBlock(bar_high=row["high"], bar_low=row["low"], bar_time=ob_bar, bias=bias)
        bucket = state.internal_obs if internal else state.swing_obs
        bucket.insert(0, ob)
        if len(bucket) > self.cfg.ob_max_stored:
            bucket.pop()

    def _mitigate_order_blocks(self, price_high, price_low, price_close, i, state: SMCState,
                                internal: bool):
        bucket = state.internal_obs if internal else state.swing_obs
        mitigation_high = price_close if self.cfg.ob_mitigation == "close" else price_high
        mitigation_low = price_close if self.cfg.ob_mitigation == "close" else price_low
        for ob in bucket:
            if ob.mitigated:
                continue
            if ob.bias == Bias.BEARISH and mitigation_high > ob.bar_high:
                ob.mitigated = True
            elif ob.bias == Bias.BULLISH and mitigation_low < ob.bar_low:
                ob.mitigated = True

    def _display_structure(self, df, i, state: SMCState, internal: bool):
        """Mirrors displayStructure(internal): checks BOS/CHoCH via crossover/crossunder."""
        close = df["close"].values
        idx = df.index
        pivot_top = state.internal_high if internal else state.swing_high
        pivot_bot = state.internal_low if internal else state.swing_low
        trend_attr = "internal_trend" if internal else "swing_trend"

        # bullish break (crossover close above pivot top)
        if (not np.isnan(pivot_top.current_level) and not pivot_top.crossed
                and i > 0 and close[i - 1] <= pivot_top.current_level < close[i]):
            kind = "CHoCH" if getattr(state, trend_attr) == Bias.BEARISH else "BOS"
            pivot_top.crossed = True
            setattr(state, trend_attr, Bias.BULLISH)
            state.events.append(StructureEvent(i, idx[i], "internal" if internal else "swing",
                                                kind, Bias.BULLISH, pivot_top.current_level))
            self._store_order_block(df, pivot_top, i, internal, Bias.BULLISH, state)

        # bearish break (crossunder close below pivot bottom)
        if (not np.isnan(pivot_bot.current_level) and not pivot_bot.crossed
                and i > 0 and close[i - 1] >= pivot_bot.current_level > close[i]):
            kind = "CHoCH" if getattr(state, trend_attr) == Bias.BULLISH else "BOS"
            pivot_bot.crossed = True
            setattr(state, trend_attr, Bias.BEARISH)
            state.events.append(StructureEvent(i, idx[i], "internal" if internal else "swing",
                                                kind, Bias.BEARISH, pivot_bot.current_level))
            self._store_order_block(df, pivot_bot, i, internal, Bias.BEARISH, state)

    def _fvgs(self, df, state: SMCState):
        """Single-timeframe 3-candle FVG detection (no MTF lookahead)."""
        high, low, close, open_ = (df["high"].values, df["low"].values,
                                    df["close"].values, df["open"].values)
        idx = df.index
        n = len(df)
        threshold_series = None
        if self.cfg.fvg_auto_threshold:
            bar_delta_pct = (close - open_) / (open_ * 100)
            cum = np.cumsum(np.abs(bar_delta_pct))
            with np.errstate(divide="ignore", invalid="ignore"):
                threshold_series = np.where(np.arange(n) > 0, cum / np.arange(1, n + 1) * 2, 0)
        for i in range(2, n):
            bar_delta_pct = (close[i - 1] - open_[i - 1]) / (open_[i - 1] * 100)
            thr = threshold_series[i] if self.cfg.fvg_auto_threshold else 0
            bullish = (low[i] > high[i - 2] and close[i - 1] > high[i - 2] and bar_delta_pct > thr)
            bearish = (high[i] < low[i - 2] and close[i - 1] < low[i - 2] and -bar_delta_pct > thr)
            if bullish:
                state.fvgs.insert(0, FVG(top=low[i], bottom=high[i - 2], bias=Bias.BULLISH,
                                          bar_time=idx[i - 1]))
            if bearish:
                state.fvgs.insert(0, FVG(top=high[i - 2], bottom=low[i], bias=Bias.BEARISH,
                                          bar_time=idx[i - 1]))
        # mark filled
        for fvg in state.fvgs:
            sub = df[df.index > fvg.bar_time]
            if fvg.bias == Bias.BULLISH and (sub["low"] < fvg.bottom).any():
                fvg.filled = True
            elif fvg.bias == Bias.BEARISH and (sub["high"] > fvg.top).any():
                fvg.filled = True

    # ------------------------------------------------------------------ run
    def run(self, df: pd.DataFrame) -> SMCState:
        """
        df must have columns: open, high, low, close (index = datetime, ascending, UTC).
        Returns the populated SMCState after processing every bar in order.
        """
        df = df.copy()
        state = SMCState()
        self._atr_cache = self._atr(df, self.cfg.atr_period).values

        swing_legs = self._legs(df, self.cfg.swing_length)
        internal_legs = self._legs(df, self.cfg.internal_length)
        eqhl_legs = self._legs(df, self.cfg.eqhl_length)

        high, low, close = df["high"].values, df["low"].values, df["close"].values
        for i in range(len(df)):
            # 1. update pivots for any leg that just confirmed at this bar
            self._update_pivot_at_bar(df, swing_legs, self.cfg.swing_length, i, state,
                                       equal_high_low=False, internal=False)
            self._update_pivot_at_bar(df, internal_legs, self.cfg.internal_length, i, state,
                                       equal_high_low=False, internal=True)
            self._update_pivot_at_bar(df, eqhl_legs, self.cfg.eqhl_length, i, state,
                                       equal_high_low=True, internal=False)
            # 2. check for structure breaks against the (possibly just-updated) pivots
            self._display_structure(df, i, state, internal=True)
            self._display_structure(df, i, state, internal=False)
            # 3. mitigate any order blocks price has now traded through
            self._mitigate_order_blocks(high[i], low[i], close[i], i, state, internal=True)
            self._mitigate_order_blocks(high[i], low[i], close[i], i, state, internal=False)
            # 4. trailing extremes (strong/weak high-low)
            state.trailing_top = max(state.trailing_top, high[i])
            state.trailing_bottom = min(state.trailing_bottom, low[i])

        self._fvgs(df, state)
        return state

    @staticmethod
    def premium_discount_zones(state: SMCState, band: float = 0.05) -> "PremiumDiscountZones":
        """
        Mirrors the original indicator's drawPremiumDiscountZones math exactly:
        premium = top band-fraction of the trailing swing range, discount = bottom
        band-fraction, equilibrium = a band-fraction-wide zone centered on the midpoint.
        """
        top, bottom = state.trailing_top, state.trailing_bottom
        mid = (top + bottom) / 2
        half = band / 2
        return PremiumDiscountZones(
            range_top=top,
            range_bottom=bottom,
            premium_bottom=(1 - band) * top + band * bottom,
            discount_top=(1 - band) * bottom + band * top,
            equilibrium_mid=mid,
            equilibrium_top=(0.5 + half) * top + (0.5 - half) * bottom,
            equilibrium_bottom=(0.5 + half) * bottom + (0.5 - half) * top,
        )

    # --------------------------------------------------------------- helpers
    @staticmethod
    def active_order_blocks(state: SMCState, internal=False):
        bucket = state.internal_obs if internal else state.swing_obs
        return [ob for ob in bucket if not ob.mitigated]

    @staticmethod
    def active_fvgs(state: SMCState):
        return [f for f in state.fvgs if not f.filled]

    @staticmethod
    def last_events(state: SMCState, n=5):
        return state.events[-n:]


# ============================================================
# SECTION 2: setup_detector.py  (trade setup logic)
# ============================================================


@dataclass
class TradeSetup:
    symbol: str
    direction: int          # Bias.BULLISH / Bias.BEARISH
    strategy: str            # "premium_discount_fade" or "ob_continuation"
    htf_timeframe: str
    ltf_timeframe: str
    trigger_zone_kind: str  # "premium", "discount", "order_block", "fvg"
    zone_top: float
    zone_bottom: float
    entry: float
    stop_loss: float
    take_profit: float
    rr: float
    htf_trend: int
    ltf_last_event: str
    current_price: float
    bar_time: pd.Timestamp


def _rr(entry, stop, target) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return round(reward / risk, 2) if risk > 0 else 0.0


def _nearest_swing_ob_beyond(engine: SMCEngine, state: SMCState, level: float,
                              direction_beyond: str):
    """
    direction_beyond="above": nearest swing OB whose top sits above `level`.
    direction_beyond="below": nearest swing OB whose bottom sits below `level`.
    Returns the OB (closest to `level`), or None.
    """
    obs = engine.active_order_blocks(state, internal=False)
    if direction_beyond == "above":
        candidates = [ob for ob in obs if ob.bar_high > level]
        return min(candidates, key=lambda ob: ob.bar_high) if candidates else None
    else:
        candidates = [ob for ob in obs if ob.bar_low < level]
        return max(candidates, key=lambda ob: ob.bar_low) if candidates else None


def _nearest_unfilled_fvg_midpoint(engine: SMCEngine, state: SMCState,
                                    near_level: float, tolerance: float) -> Optional[float]:
    """Find an unfilled FVG whose 50% level sits within `tolerance` of near_level."""
    best = None
    best_dist = tolerance
    for f in engine.active_fvgs(state):
        mid = (f.top + f.bottom) / 2
        dist = abs(mid - near_level)
        if dist <= best_dist:
            best, best_dist = mid, dist
    return best


# ------------------------------------------------------- primary: premium/discount fade
def find_premium_discount_setup(symbol: str,
                                 htf_df: pd.DataFrame, htf_state: SMCState, htf_tf: str,
                                 ltf_df: pd.DataFrame, ltf_state: SMCState, ltf_tf: str,
                                 engine: SMCEngine,
                                 sl_buffer_atr_mult: float = 0.25,
                                 fvg_equilibrium_tolerance_pct: float = 0.15,
                                 min_rr: float = 1.5) -> Optional[TradeSetup]:
    """
    Checks the HTF range for a premium/discount fade setup. Uses HTF structure for the
    range/zones (that's what the examples use -- 2h and weekly), and the LTF close as
    the live price for confluence/timing.
    """
    if htf_state.trailing_top is None or htf_state.trailing_bottom is None:
        return None

    pdz = engine.premium_discount_zones(htf_state)
    current_price = ltf_df["close"].iloc[-1]
    bar_time = ltf_df.index[-1]
    atr = engine._atr(htf_df, engine.cfg.atr_period).iloc[-1]

    last_event = ltf_state.events[-1] if ltf_state.events else None
    last_event_desc = f"{last_event.kind} {last_event.scope}" if last_event else "none"

    # --- SHORT: price trading in the premium zone (fade back toward equilibrium)
    if current_price >= pdz.premium_bottom:
        entry = current_price
        beyond_ob = _nearest_swing_ob_beyond(engine, htf_state, pdz.range_top, "above")
        stop_loss = max(pdz.range_top, beyond_ob.bar_high if beyond_ob else 0) \
            + sl_buffer_atr_mult * atr
        take_profit = _nearest_unfilled_fvg_midpoint(
            engine, htf_state, pdz.equilibrium_mid,
            tolerance=fvg_equilibrium_tolerance_pct / 100 * (pdz.range_top - pdz.range_bottom)
        ) or pdz.equilibrium_mid

        rr = _rr(entry, stop_loss, take_profit)
        if rr >= min_rr:
            return TradeSetup(
                symbol=symbol, direction=Bias.BEARISH, strategy="premium_discount_fade",
                htf_timeframe=htf_tf, ltf_timeframe=ltf_tf, trigger_zone_kind="premium",
                zone_top=pdz.range_top, zone_bottom=pdz.premium_bottom,
                entry=entry, stop_loss=stop_loss, take_profit=take_profit, rr=rr,
                htf_trend=htf_state.swing_trend, ltf_last_event=last_event_desc,
                current_price=current_price, bar_time=bar_time,
            )

    # --- LONG: price trading in the discount zone (fade back toward equilibrium)
    if current_price <= pdz.discount_top:
        entry = current_price
        beyond_ob = _nearest_swing_ob_beyond(engine, htf_state, pdz.range_bottom, "below")
        stop_loss = min(pdz.range_bottom, beyond_ob.bar_low if beyond_ob else float("inf")) \
            - sl_buffer_atr_mult * atr
        take_profit = _nearest_unfilled_fvg_midpoint(
            engine, htf_state, pdz.equilibrium_mid,
            tolerance=fvg_equilibrium_tolerance_pct / 100 * (pdz.range_top - pdz.range_bottom)
        ) or pdz.equilibrium_mid

        rr = _rr(entry, stop_loss, take_profit)
        if rr >= min_rr:
            return TradeSetup(
                symbol=symbol, direction=Bias.BULLISH, strategy="premium_discount_fade",
                htf_timeframe=htf_tf, ltf_timeframe=ltf_tf, trigger_zone_kind="discount",
                zone_top=pdz.discount_top, zone_bottom=pdz.range_bottom,
                entry=entry, stop_loss=stop_loss, take_profit=take_profit, rr=rr,
                htf_trend=htf_state.swing_trend, ltf_last_event=last_event_desc,
                current_price=current_price, bar_time=bar_time,
            )

    return None


# ------------------------------------------------------- secondary: OB/FVG continuation
def find_continuation_setup(symbol: str,
                             htf_df: pd.DataFrame, htf_state: SMCState, htf_tf: str,
                             ltf_df: pd.DataFrame, ltf_state: SMCState, ltf_tf: str,
                             engine: SMCEngine,
                             sl_buffer_atr_mult: float = 0.15) -> Optional[TradeSetup]:
    """
    Price returning into an unmitigated LTF order block / FVG that agrees with the HTF
    swing trend direction. Kept as a secondary confirmation model -- trend continuation
    rather than the fade strategy above.
    """
    if htf_state.swing_trend == Bias.NONE:
        return None

    htf_bias = htf_state.swing_trend
    current_price = ltf_df["close"].iloc[-1]
    bar_time = ltf_df.index[-1]
    atr = engine._atr(ltf_df, engine.cfg.atr_period).iloc[-1]

    candidates = []
    for ob in engine.active_order_blocks(ltf_state, internal=True):
        candidates.append(("order_block", ob.bar_low, ob.bar_high, ob.bias))
    for ob in engine.active_order_blocks(ltf_state, internal=False):
        candidates.append(("order_block", ob.bar_low, ob.bar_high, ob.bias))
    for f in engine.active_fvgs(ltf_state):
        candidates.append(("fvg", f.bottom, f.top, f.bias))

    for kind, bottom, top, bias in candidates:
        if bias != htf_bias or not (bottom <= current_price <= top):
            continue

        direction = bias
        entry = current_price
        if direction == Bias.BULLISH:
            stop_loss = bottom - sl_buffer_atr_mult * atr
            take_profit = htf_state.trailing_top
        else:
            stop_loss = top + sl_buffer_atr_mult * atr
            take_profit = htf_state.trailing_bottom

        rr = _rr(entry, stop_loss, take_profit)
        if rr <= 0:
            continue

        last_event = ltf_state.events[-1] if ltf_state.events else None
        last_event_desc = f"{last_event.kind} {last_event.scope}" if last_event else "none"

        return TradeSetup(
            symbol=symbol, direction=direction, strategy="ob_continuation",
            htf_timeframe=htf_tf, ltf_timeframe=ltf_tf, trigger_zone_kind=kind,
            zone_top=top, zone_bottom=bottom, entry=entry, stop_loss=stop_loss,
            take_profit=take_profit, rr=rr, htf_trend=htf_bias,
            ltf_last_event=last_event_desc, current_price=current_price, bar_time=bar_time,
        )

    return None


def find_setup(symbol: str,
               htf_df: pd.DataFrame, htf_state: SMCState, htf_tf: str,
               ltf_df: pd.DataFrame, ltf_state: SMCState, ltf_tf: str,
               engine: SMCEngine) -> Optional[TradeSetup]:
    """Tries the premium/discount fade first (the primary documented strategy);
    falls back to OB/FVG continuation if no fade setup is active."""
    setup = find_premium_discount_setup(symbol, htf_df, htf_state, htf_tf,
                                         ltf_df, ltf_state, ltf_tf, engine)
    if setup:
        return setup
    return find_continuation_setup(symbol, htf_df, htf_state, htf_tf,
                                    ltf_df, ltf_state, ltf_tf, engine)


# ============================================================
# SECTION 3: snapshot.py  (chart rendering)
# ============================================================


def render_setup_chart(df, state: SMCState, setup: TradeSetup, out_path: str,
                        lookback_bars: int = 150):
    plot_df = df.tail(lookback_bars).copy()
    plot_df.index.name = "Date"

    addplots = []
    fig, axlist = mpf.plot(
        plot_df,
        type="candle",
        style="nightclouds",
        volume=False,
        returnfig=True,
        figsize=(11, 6.5),
        title=f"\n{setup.symbol}  {setup.ltf_timeframe} (HTF bias: {setup.htf_timeframe})",
    )
    ax = axlist[0]
    x0, x1 = ax.get_xlim()

    # highlight the trigger zone (order block / FVG)
    zone_color = "#1848cc" if setup.direction == Bias.BULLISH else "#b22833"
    ax.axhspan(setup.zone_bottom, setup.zone_top, xmin=0, xmax=1,
               color=zone_color, alpha=0.25,
               label=f"{setup.trigger_zone_kind.replace('_', ' ')} zone")

    # entry / SL / TP lines
    ax.axhline(setup.entry, color="#e8eaed", linestyle="--", linewidth=1.2)
    ax.text(x1, setup.entry, f" Entry {setup.entry:,.1f}", color="#e8eaed",
            va="center", fontsize=9)

    ax.axhline(setup.stop_loss, color="#F23645", linestyle="--", linewidth=1.2)
    ax.text(x1, setup.stop_loss, f" SL {setup.stop_loss:,.1f}", color="#F23645",
            va="center", fontsize=9)

    ax.axhline(setup.take_profit, color="#089981", linestyle="--", linewidth=1.2)
    ax.text(x1, setup.take_profit, f" TP {setup.take_profit:,.1f}", color="#089981",
            va="center", fontsize=9)

    direction_word = "LONG" if setup.direction == Bias.BULLISH else "SHORT"
    ax.set_title(
        f"{setup.symbol} — {direction_word} setup ({setup.trigger_zone_kind}) "
        f"RR {setup.rr}  |  HTF {setup.htf_timeframe} trend: "
        f"{'UP' if setup.htf_trend == Bias.BULLISH else 'DOWN'}",
        fontsize=11, color="#e8eaed"
    )

    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0a0e14")
    plt.close(fig)
    return out_path


# ============================================================
# SECTION 4: smc_ai_alert_bot.py  (runner)
# ============================================================


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("smc_ai_alert_bot")

SYMBOL = os.getenv("SMC_SYMBOL", "BTC/USDT:USDT")
HTF = "4h"
LTF = "15m"
POLL_SECONDS = int(os.getenv("SMC_POLL_SECONDS", "900"))
STATE_DB = os.getenv("SMC_STATE_DB", "smc_alert_state.sqlite")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


# --------------------------------------------------------------------------- state
def init_db():
    con = sqlite3.connect(STATE_DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS sent_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT, direction INTEGER, zone_kind TEXT,
            zone_top REAL, zone_bottom REAL, bar_time TEXT,
            sent_at TEXT
        )
    """)
    con.commit()
    return con


def already_alerted(con, setup: TradeSetup) -> bool:
    row = con.execute(
        "SELECT 1 FROM sent_alerts WHERE symbol=? AND direction=? AND zone_kind=? "
        "AND ABS(zone_top-?) < 1 AND ABS(zone_bottom-?) < 1",
        (setup.symbol, int(setup.direction), setup.trigger_zone_kind,
         setup.zone_top, setup.zone_bottom)
    ).fetchone()
    return row is not None


def record_alert(con, setup: TradeSetup):
    con.execute(
        "INSERT INTO sent_alerts (symbol, direction, zone_kind, zone_top, zone_bottom, "
        "bar_time, sent_at) VALUES (?,?,?,?,?,?,?)",
        (setup.symbol, int(setup.direction), setup.trigger_zone_kind,
         setup.zone_top, setup.zone_bottom, str(setup.bar_time),
         datetime.now(timezone.utc).isoformat())
    )
    con.commit()


# --------------------------------------------------------------------------- data
def fetch_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts")
    # drop the still-forming candle so structure only evaluates closed bars
    return df.iloc[:-1]


# ----------------------------------------------------------------------------- LLM
def get_ai_rationale(setup: TradeSetup) -> str:
    """
    Asks Claude for a short, structured technical readout of the setup. This is
    descriptive market-structure commentary based on the data you already computed --
    not a prediction and not financial advice, and the prompt says so explicitly so
    the model doesn't try to hedge or refuse.
    """
    if not ANTHROPIC_API_KEY:
        return "(no ANTHROPIC_API_KEY set -- skipping AI rationale)"

    direction_word = "long" if setup.direction == Bias.BULLISH else "short"
    if setup.strategy == "premium_discount_fade":
        strategy_desc = (
            f"a premium/discount fade: price is trading in the {setup.trigger_zone_kind} "
            f"zone of the {setup.htf_timeframe} swing range ({setup.zone_bottom:.1f}-{setup.zone_top:.1f}), "
            f"targeting a reversion toward equilibrium (or the 50% level of a nearby unfilled FVG)"
        )
    else:
        strategy_desc = (
            f"a trend continuation: price has returned into an unmitigated "
            f"{setup.trigger_zone_kind.replace('_', ' ')} on the {setup.ltf_timeframe} chart "
            f"that agrees with the {setup.htf_timeframe} swing trend"
        )

    prompt = f"""You are annotating a systematic SMC (smart money concepts) trade alert
for an experienced independent trader's own private Telegram feed. All the technical
levels below were already computed algorithmically -- your job is only to write a
tight 3-4 sentence readout explaining the structural logic in plain language, and one
sentence noting the main invalidation risk. Do not add a confidence score, do not tell
the reader whether to take the trade, do not add disclaimers -- they already know this
is not financial advice.

Setup type: {strategy_desc}

Setup data:
- Symbol: {setup.symbol}
- Direction: {direction_word}
- HTF ({setup.htf_timeframe}) trend: {'bullish' if setup.htf_trend == Bias.BULLISH else 'bearish'}
- Trigger zone ({setup.trigger_zone_kind}): {setup.zone_bottom:.1f}-{setup.zone_top:.1f}
- Most recent LTF structure event: {setup.ltf_last_event}
- Current price: {setup.current_price:.1f}
- Proposed entry: {setup.entry:.1f}
- Stop loss: {setup.stop_loss:.1f}
- Take profit: {setup.take_profit:.1f}
- R:R: {setup.rr}
"""
    try:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-sonnet-4-6",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()
    except Exception as e:
        log.warning(f"AI rationale call failed: {e}")
        return "(AI rationale unavailable -- see levels above)"


# ------------------------------------------------------------------------ telegram
TELEGRAM_CAPTION_LIMIT = 1024   # sendPhoto caption hard limit
TELEGRAM_MESSAGE_LIMIT = 4096   # sendMessage hard limit


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    # leave room for an ellipsis marker
    return text[: max(0, limit - 1)].rstrip() + "…"


def send_telegram_alert(setup: TradeSetup, rationale: str, chart_path: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.warning("Telegram not configured -- printing alert instead:")
        print(rationale)
        return

    direction_word = "🟢 LONG" if setup.direction == Bias.BULLISH else "🔴 SHORT"
    strategy_label = "Premium/Discount Fade" if setup.strategy == "premium_discount_fade" \
        else "OB/FVG Continuation"

    # Levels-only caption goes on the photo -- keep this short and predictable so it
    # never risks tripping the 1024-char sendPhoto caption limit, regardless of how
    # long the AI rationale ends up being.
    caption = (
        f"*{setup.symbol}* — {direction_word} setup ({strategy_label})\n"
        f"HTF {setup.htf_timeframe} bias: {'UP' if setup.htf_trend == Bias.BULLISH else 'DOWN'}  "
        f"| Zone: {setup.trigger_zone_kind.replace('_', ' ')}\n\n"
        f"Entry: `{setup.entry:,.1f}`\n"
        f"Stop: `{setup.stop_loss:,.1f}`\n"
        f"Target: `{setup.take_profit:,.1f}`\n"
        f"R:R: `{setup.rr}`"
    )
    caption = _truncate(caption, TELEGRAM_CAPTION_LIMIT)

    photo_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    with open(chart_path, "rb") as f:
        r = requests.post(
            photo_url,
            data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": f},
            timeout=30,
        )
    if not r.ok:
        log.error(f"Telegram photo send failed: {r.status_code} {r.text}")
        return
    log.info("Telegram photo alert sent")

    # Rationale as a separate follow-up message -- 4096-char budget, still truncated
    # defensively in case Claude ever returns something unusually long.
    if rationale:
        message_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        message_text = _truncate(rationale, TELEGRAM_MESSAGE_LIMIT)
        r2 = requests.post(
            message_url,
            data={"chat_id": TELEGRAM_CHAT_ID, "text": message_text},
            timeout=30,
        )
        if not r2.ok:
            log.error(f"Telegram rationale send failed: {r2.status_code} {r2.text}")
        else:
            log.info("Telegram rationale message sent")


# ---------------------------------------------------------------------------- core
def run_once(exchange, engine: SMCEngine, con):
    htf_df = fetch_ohlcv(exchange, SYMBOL, HTF)
    ltf_df = fetch_ohlcv(exchange, SYMBOL, LTF)

    htf_state = engine.run(htf_df)
    ltf_state = engine.run(ltf_df)

    setup = find_setup(SYMBOL, htf_df, htf_state, HTF, ltf_df, ltf_state, LTF, engine)
    if setup is None:
        log.info("No setup right now.")
        return

    if already_alerted(con, setup):
        log.info("Setup already alerted, skipping.")
        return

    log.info(f"New setup: {setup}")
    chart_path = f"/tmp/smc_setup_{int(time.time())}.png"
    render_setup_chart(ltf_df, ltf_state, setup, chart_path)
    rationale = get_ai_rationale(setup)
    send_telegram_alert(setup, rationale, chart_path)
    record_alert(con, setup)


def main():
    exchange = ccxt.okx({"enableRateLimit": True})
    engine = SMCEngine(SMCConfig(swing_length=50, internal_length=5))
    con = init_db()

    log.info(f"Starting smc_ai_alert_bot for {SYMBOL} (HTF={HTF}, LTF={LTF}, "
             f"poll={POLL_SECONDS}s)")
    while True:
        try:
            run_once(exchange, engine, con)
        except Exception as e:
            log.exception(f"run_once failed: {e}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
