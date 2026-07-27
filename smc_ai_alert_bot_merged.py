"""
SMC Premium/Discount Zone Monitor
----------------------------------
Tracks Daily + 4H premium/discount zones (LuxAlgo SMC concept, reimplemented
in Python off live OKX data), checks for HTF/LTF alignment, and has an LLM
turn the raw state into a Telegram-ready trade brief.

On startup it runs one full analysis immediately and posts the current
state. After that it polls on an interval and only posts again when the
zone/alignment actually changes (or a configurable heartbeat elapses), so
you're not getting spammed every 15 minutes with the same "still in daily
discount" message.

Env vars (matches the names used in smc_ai_alert_bot_merged.py so you can
reuse the same Railway env group without adding duplicates):
  SMC_SYMBOL              default 'BTC/USDT:USDT'  (same ccxt perp symbol)
  TELEGRAM_BOT_TOKEN      required to actually send
  TELEGRAM_CHAT_ID        required to actually send
  ANTHROPIC_API_KEY       optional — falls back to a templated summary if unset
  SMC_POLL_SECONDS        default 900  (15 min)

Optional, unique to this script (no equivalent in the merged bot):
  ANTHROPIC_MODEL         default 'claude-sonnet-5'
  SWING_LENGTH_DAILY      default 50   (bars to confirm a swing pivot, matches
                          the merged bot's cfg.swing_length default)
  SWING_LENGTH_4H         default 50
  PD_BAND                 default 0.05 (premium/discount edge fraction, matches
                          the merged bot's cfg.premium_discount_band default)
  OB_LOOKBACK             default 150  (bars scanned for order blocks)
  BREAKER_Z_LEN           default 100  (z-score window for breaker impulse detection)
  BREAKER_Z_THRESHOLD     default 4.0  (z-score crossover level that flags an impulse)
  BREAKER_MAX_AGE         default 500  (bars before an unmitigated breaker/OB candidate expires)
  BREAKER_ZONE_BAND       default 0.5  (which-half split for filtering breakers — see note
                          below; deliberately NOT the same as PD_BAND)
  HEARTBEAT_HOURS         default 6    (post even with no change after this long)
  SMC_PD_STATE_FILE       default 'pd_zone_state.json'  (own file — separate
                          from SMC_STATE_DB, which is the other bot's sqlite
                          alert log and a different format)
  CHART_BARS              default 120  (candles shown on each snapshot)
  CHART_DIR               default 'charts'

Also needs: mplfinance, matplotlib (on top of ccxt/pandas/requests).
"""

import os
import json
import time
import logging

import ccxt
import numpy as np
import pandas as pd
import requests
import mplfinance as mpf
import matplotlib.pyplot as plt

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pd_zones")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SYMBOL = os.getenv("SMC_SYMBOL", "BTC/USDT:USDT")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5")

SWING_LENGTH_DAILY = int(os.getenv("SWING_LENGTH_DAILY", 50))
SWING_LENGTH_4H = int(os.getenv("SWING_LENGTH_4H", 50))
PD_BAND = float(os.getenv("PD_BAND", 0.05))

OB_LOOKBACK = int(os.getenv("OB_LOOKBACK", 150))
BREAKER_Z_LEN = int(os.getenv("BREAKER_Z_LEN", 100))
BREAKER_Z_THRESHOLD = float(os.getenv("BREAKER_Z_THRESHOLD", 4.0))
BREAKER_MAX_AGE = int(os.getenv("BREAKER_MAX_AGE", 500))
BREAKER_ZONE_BAND = float(os.getenv("BREAKER_ZONE_BAND", 0.5))
POLL_SECONDS = int(os.getenv("SMC_POLL_SECONDS", 900))
HEARTBEAT_HOURS = float(os.getenv("HEARTBEAT_HOURS", 6))
STATE_FILE = os.getenv("SMC_PD_STATE_FILE", "pd_zone_state.json")

CHART_BARS = int(os.getenv("CHART_BARS", 120))
CHART_DIR = os.getenv("CHART_DIR", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

exchange = ccxt.okx({"enableRateLimit": True})


# ---------------------------------------------------------------------------
# DATA
# ---------------------------------------------------------------------------
def fetch_ohlcv(symbol, timeframe, limit=500):
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df


# ---------------------------------------------------------------------------
# SWING STRUCTURE -> TRAILING RANGE -> PREMIUM/DISCOUNT ZONE
#
# Faithful port of LuxAlgo's leg()/getCurrentStructure()/updateTrailingExtremes,
# same algorithm your smc_ai_alert_bot_merged.py's SMCEngine uses. The dealing
# range does NOT reset both sides on every BOS/CHoCH — it only re-anchors the
# side that just confirmed a new pivot (top on a new bearish leg, bottom on a
# new bullish leg), while the other side keeps expanding outward with each
# bar's high/low until it gets its own reversal confirmation. That's what
# makes the range expand until price eventually breaks and confirms both
# sides. An earlier version of this script used a symmetric left/right
# fractal per timeframe instead, which doesn't reproduce that expand-until-
# broken behavior and was reading the wrong side as premium/discount.
# ---------------------------------------------------------------------------
def compute_legs(df, size):
    """Reproduces `leg(size)`: BEARISH_LEG (0) while trending down, BULLISH_LEG (1)
    while trending up, flipping when a `size`-bars-back extreme survives the most
    recent `size`-bar rolling window unbroken."""
    high, low = df["high"].values, df["low"].values
    n = len(df)
    roll_high = df["high"].rolling(size).max().values
    roll_low = df["low"].rolling(size).min().values

    legs = np.zeros(n, dtype=int)
    cur = 0  # BEARISH_LEG
    for i in range(n):
        if i < size or np.isnan(roll_high[i]) or np.isnan(roll_low[i]):
            legs[i] = cur
            continue
        new_leg_high = high[i - size] > roll_high[i]
        new_leg_low = low[i - size] < roll_low[i]
        if new_leg_high:
            cur = 0
        elif new_leg_low:
            cur = 1
        legs[i] = cur
    return legs


def compute_structure(df, length, fib_level=0.71):
    """
    Walks the dataframe bar-by-bar maintaining:
      - trailing top/bottom (the dealing range) exactly like the indicator:
        re-anchor the side whose leg just confirmed, then let both sides
        expand to the current bar's high/low.
      - the swing trend bias (BULLISH/BEARISH), by tracking the last confirmed
        swing-high and swing-low pivot levels and flagging the first close
        that crosses each one (mirrors displayStructure()'s BOS/CHoCH check).
        This uses the fixed confirmed-pivot level, not the ever-expanding
        trailing top/bottom, since that's what the real indicator crosses
        against.
      - strong/weak labels off that bias: in a bearish trend the top is the
        Strong High (the origin of the down-leg, unlikely to break) and the
        bottom is the Weak Low (formed on the same push, expected to get
        swept before a real reversal); a bullish trend flips that — Strong
        Low, Weak High.
      - two fib levels anchored to the dealing range: fib_short sits `fib_level`
        (0.71) of the way up from the bottom (premium-side sell zone),
        fib_long sits `fib_level` of the way down from the top (discount-side
        buy zone). These prices don't depend on bias — bias only labels which
        one is the with-trend trade and which is the counter-trend fade, so
        both directions are always available. Recomputed fresh from the live
        top/bottom every call, so they re-anchor automatically whenever the
        range expands or resets.
    Returns the state as of the most recent bar.
    """
    n = len(df)
    high, low, close = df["high"].values, df["low"].values, df["close"].values
    ts = df["ts"].values
    legs = compute_legs(df, length)

    trailing_top = float(high[0])
    trailing_bottom = float(low[0])
    top_time = ts[0]
    bottom_time = ts[0]

    swing_high_level, swing_high_crossed = None, False
    swing_low_level, swing_low_crossed = None, False
    bias = None  # None until the first confirmed BOS/CHoCH, then "BULLISH"/"BEARISH"

    for i in range(n):
        if i >= length and legs[i] != legs[i - 1]:
            src_i = i - length
            if legs[i] == 1:  # new bullish leg -> swing low just confirmed
                trailing_bottom = float(low[src_i])
                bottom_time = ts[src_i]
                swing_low_level = float(low[src_i])
                swing_low_crossed = False
            else:  # new bearish leg -> swing high just confirmed
                trailing_top = float(high[src_i])
                top_time = ts[src_i]
                swing_high_level = float(high[src_i])
                swing_high_crossed = False

        if high[i] >= trailing_top:
            trailing_top = float(high[i])
            top_time = ts[i]
        if low[i] <= trailing_bottom:
            trailing_bottom = float(low[i])
            bottom_time = ts[i]

        if swing_high_level is not None and not swing_high_crossed and close[i] > swing_high_level:
            bias = "BULLISH"
            swing_high_crossed = True
        if swing_low_level is not None and not swing_low_crossed and close[i] < swing_low_level:
            bias = "BEARISH"
            swing_low_crossed = True

    if bias == "BEARISH":
        top_label, bottom_label = "Strong High", "Weak Low"
    elif bias == "BULLISH":
        top_label, bottom_label = "Weak High", "Strong Low"
    else:
        top_label, bottom_label = "High", "Low"

    # The two fib prices are always the same two numbers regardless of bias —
    # fib_short sits `fib_level` of the way up from bottom (premium-side, the
    # sell zone), fib_long sits `fib_level` of the way down from top
    # (discount-side, the buy zone). Bias doesn't move these prices, it only
    # decides which one is "with the trend" vs "fading it": in a downtrend
    # fib_short is the with-trend (short) entry and fib_long is the counter-
    # trend (long) entry; an uptrend flips that. Both are always returned so
    # you can trade either direction, not just the one bias favors.
    rng = trailing_top - trailing_bottom
    fib_short = trailing_bottom + fib_level * rng        # premium-side / sell zone
    fib_long = trailing_bottom + (1 - fib_level) * rng   # discount-side / buy zone

    if bias == "BEARISH":
        with_trend_dir, with_trend_fib = "SHORT", fib_short
        counter_dir, counter_fib = "LONG", fib_long
    elif bias == "BULLISH":
        with_trend_dir, with_trend_fib = "LONG", fib_long
        counter_dir, counter_fib = "SHORT", fib_short
    else:
        with_trend_dir = with_trend_fib = counter_dir = counter_fib = None

    return {
        "top": trailing_top, "top_time": pd.Timestamp(top_time), "top_label": top_label,
        "bottom": trailing_bottom, "bottom_time": pd.Timestamp(bottom_time), "bottom_label": bottom_label,
        "bias": bias,
        "fib_long": float(fib_long), "fib_short": float(fib_short),
        "with_trend_dir": with_trend_dir, "with_trend_fib": with_trend_fib,
        "counter_dir": counter_dir, "counter_fib": counter_fib,
    }


def classify_zone(price, top, bottom, band=PD_BAND):
    """
    Same thresholds as the merged bot's PremiumDiscountZones/classify_zone:
    premium = top `band` fraction of the range, discount = bottom `band`
    fraction, everything else (including the whole 50% midline) is
    equilibrium — deliberately narrow bands, not a simple above/below-50% split.
    """
    rng = top - bottom
    if rng <= 0:
        return "Equilibrium", 50.0
    pct = (price - bottom) / rng * 100
    premium_bottom = (1 - band) * top + band * bottom
    discount_top = (1 - band) * bottom + band * top
    if price >= premium_bottom:
        zone = "Premium"
    elif price <= discount_top:
        zone = "Discount"
    else:
        zone = "Equilibrium"
    return zone, round(pct, 1)


# ---------------------------------------------------------------------------
# SIMPLIFIED ORDER BLOCK DETECTION
# ---------------------------------------------------------------------------
def find_fractals(df, left, right):
    """Simple symmetric fractal highs/lows — only used for the OB proxy below,
    not for the premium/discount range (see compute_trailing_range for that)."""
    highs, lows = [], []
    n = len(df)
    for i in range(left, n - right):
        window_high = df["high"].iloc[i - left:i + right + 1]
        window_low = df["low"].iloc[i - left:i + right + 1]
        if df["high"].iloc[i] == window_high.max():
            highs.append((i, df["high"].iloc[i]))
        if df["low"].iloc[i] == window_low.min():
            lows.append((i, df["low"].iloc[i]))
    return highs, lows


def find_order_blocks(df, lookback=150, left=5, right=5):
    """
    Lightweight self-contained OB finder: last opposite-colored candle before
    a break of a recent fractal swing point becomes the order block. Returns
    the most recent unmitigated bullish and bearish OB (or None each). This
    is a proxy for a full OB engine, not a faithful order-flow reconstruction
    — good enough to flag a level to watch, not precision-tuned.
    """
    sub = df.tail(lookback).reset_index(drop=True)
    highs, lows = find_fractals(sub, left, right)

    bearish_ob = None
    for idx, level in reversed(lows):
        broken_at = next((j for j in range(idx + 1, len(sub)) if sub["close"].iloc[j] < level), None)
        if broken_at is None:
            continue
        for k in range(broken_at - 1, idx - 1, -1):
            if sub["close"].iloc[k] > sub["open"].iloc[k]:
                ob_high, ob_low = float(sub["high"].iloc[k]), float(sub["low"].iloc[k])
                tail = sub["high"].iloc[broken_at + 1:]
                mitigated = (tail > ob_low).any() if len(tail) else False
                if not mitigated:
                    bearish_ob = {"high": ob_high, "low": ob_low, "time": sub["ts"].iloc[k]}
                break
        if bearish_ob:
            break

    bullish_ob = None
    for idx, level in reversed(highs):
        broken_at = next((j for j in range(idx + 1, len(sub)) if sub["close"].iloc[j] > level), None)
        if broken_at is None:
            continue
        for k in range(broken_at - 1, idx - 1, -1):
            if sub["close"].iloc[k] < sub["open"].iloc[k]:
                ob_high, ob_low = float(sub["high"].iloc[k]), float(sub["low"].iloc[k])
                tail = sub["low"].iloc[broken_at + 1:]
                mitigated = (tail < ob_high).any() if len(tail) else False
                if not mitigated:
                    bullish_ob = {"high": ob_high, "low": ob_low, "time": sub["ts"].iloc[k]}
                break
        if bullish_ob:
            break

    return bullish_ob, bearish_ob


# ---------------------------------------------------------------------------
# BREAKER BLOCKS (port of AlgoAlpha's "Breaker Blocks Signals", 4H only)
#
# Cumulative same-direction move distance (updist/downdist) gets z-scored
# against a rolling window; a z-score crossing above the threshold flags an
# impulsive move. That creates an order-block candidate box from the last
# opposite-colored candle before the impulse. If price later closes back
# through that candidate for two consecutive bars, the OB failed and flips
# polarity into a "breaker" block (a failed bullish OB becomes bearish
# resistance; a failed bearish OB becomes bullish support). Boxes expire
# after BREAKER_MAX_AGE bars if never mitigated.
#
# Only 4H breakers get computed at all — per your call, no daily, no LTF.
# On top of that, a breaker only counts if it's on the correct SIDE of the
# live 4H dealing range: bullish breaker in the discount half, bearish
# breaker in the premium half. Anything on the wrong side gets thrown out
# entirely — it's never returned, drawn, or mentioned.
#
# Note this deliberately does NOT reuse PD_BAND (0.05), which only tags the
# outer 5% edges of the range as Premium/Discount and calls the other 90%
# Equilibrium. A breaker that forms right around the midpoint — technically
# in "equilibrium" by that narrow definition — still leans to one side of
# the 50% line, and that lean is what matters for a breaker: it still tags
# discount or premium, just close to the border. So breaker filtering uses
# its own BREAKER_ZONE_BAND (default 0.5 = a straight half-range split, no
# equilibrium gap at all) instead of PD_BAND. The main Premium/Discount zone
# label shown for daily/4H is untouched — it still uses the narrow PD_BAND.
# ---------------------------------------------------------------------------
def compute_breakers(df, z_len=100, z_threshold=4.0, max_age=500):
    """
    Returns (bull_breakers, bear_breakers): lists of currently active
    (unmitigated, unexpired) breaker boxes as of the last bar, each
    {'top', 'bottom', 'start', 'formed_time'}. Unfiltered by zone — the
    caller applies the discount/premium filter.
    """
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    ts = df["ts"].values
    n = len(df)

    updist = np.zeros(n)
    downdist = np.zeros(n)
    for i in range(n):
        prev_up = updist[i - 1] if i > 0 else 0.0
        prev_dn = downdist[i - 1] if i > 0 else 0.0
        updist[i] = prev_up + (c[i] - o[i]) if c[i] > o[i] else 0.0
        downdist[i] = prev_dn + (o[i] - c[i]) if c[i] < o[i] else 0.0

    up_mean = pd.Series(updist).rolling(z_len).mean().values
    up_std = pd.Series(updist).rolling(z_len).std(ddof=0).values
    dn_mean = pd.Series(downdist).rolling(z_len).mean().values
    dn_std = pd.Series(downdist).rolling(z_len).std(ddof=0).values

    z_up = np.where((up_std > 0) & ~np.isnan(up_std), (updist - up_mean) / np.where(up_std == 0, np.nan, up_std), np.nan)
    z_dn = np.where((dn_std > 0) & ~np.isnan(dn_std), (downdist - dn_mean) / np.where(dn_std == 0, np.nan, dn_std), np.nan)

    def overlaps(top_new, bottom_new, *box_lists):
        for boxes in box_lists:
            for bx in boxes:
                if top_new > bx["bottom"] and bottom_new < bx["top"]:
                    return True
        return False

    last_down_idx, last_up_idx = None, None
    bull_cand, bear_cand = [], []   # OB candidates awaiting mitigation
    breaker_bull, breaker_bear = [], []  # confirmed, still-active breakers

    for i in range(n):
        if c[i] < o[i]:
            last_down_idx = i
        if c[i] > o[i]:
            last_up_idx = i

        prev_z_up = z_up[i - 1] if i > 0 else np.nan
        prev_z_dn = z_dn[i - 1] if i > 0 else np.nan
        bullish_signal = (not np.isnan(z_up[i]) and not np.isnan(prev_z_up)
                          and prev_z_up <= z_threshold and z_up[i] > z_threshold and prev_z_up != 0)
        bearish_signal = (not np.isnan(z_dn[i]) and not np.isnan(prev_z_dn)
                          and prev_z_dn <= z_threshold and z_dn[i] > z_threshold and prev_z_dn != 0)

        if bullish_signal and last_down_idx is not None:
            t, b = float(h[last_down_idx]), float(l[last_down_idx])
            if not overlaps(t, b, bull_cand, bear_cand, breaker_bull, breaker_bear):
                bull_cand.append({"top": t, "bottom": b, "start": last_down_idx})

        if bearish_signal and last_up_idx is not None:
            t, b = float(h[last_up_idx]), float(l[last_up_idx])
            if not overlaps(t, b, bull_cand, bear_cand, breaker_bull, breaker_bear):
                bear_cand.append({"top": t, "bottom": b, "start": last_up_idx})

        still = []
        for bx in bull_cand:
            mitigated = i >= 1 and c[i] < bx["bottom"] and c[i - 1] < bx["bottom"]
            expired = (i - bx["start"]) >= max_age
            if mitigated and not overlaps(bx["top"], bx["bottom"], breaker_bull, breaker_bear):
                breaker_bear.append({"top": bx["top"], "bottom": bx["bottom"], "start": i, "formed_time": ts[i]})
            if not (mitigated or expired):
                still.append(bx)
        bull_cand = still

        still = []
        for bx in bear_cand:
            mitigated = i >= 1 and c[i] > bx["top"] and c[i - 1] > bx["top"]
            expired = (i - bx["start"]) >= max_age
            if mitigated and not overlaps(bx["top"], bx["bottom"], breaker_bull, breaker_bear):
                breaker_bull.append({"top": bx["top"], "bottom": bx["bottom"], "start": i, "formed_time": ts[i]})
            if not (mitigated or expired):
                still.append(bx)
        bear_cand = still

        breaker_bull = [bx for bx in breaker_bull
                         if not (i >= 1 and c[i] < bx["bottom"] and c[i - 1] < bx["bottom"])
                         and (i - bx["start"]) < max_age]
        breaker_bear = [bx for bx in breaker_bear
                         if not (i >= 1 and c[i] > bx["top"] and c[i - 1] > bx["top"])
                         and (i - bx["start"]) < max_age]

    for bx in breaker_bull + breaker_bear:
        bx["formed_time"] = pd.Timestamp(bx["formed_time"])

    return breaker_bull, breaker_bear


def find_valid_breakers(df, top, bottom, z_len=100, z_threshold=4.0, max_age=500, band=BREAKER_ZONE_BAND):
    """
    Runs compute_breakers, then drops anything on the wrong side of the
    range: bull breaker must be in the discount half, bear breaker must be
    in the premium half. This uses a plain which-half split (band=0.5 by
    default), not the narrow PD_BAND edge-zone definition — see the comment
    block above compute_breakers for why. Everything on the wrong side is
    discarded — never returned. Returns the single most recently formed
    valid bull breaker and bear breaker (or None each).
    """
    bull_breakers, bear_breakers = compute_breakers(df, z_len, z_threshold, max_age)

    valid_bull = [bx for bx in bull_breakers
                  if classify_zone((bx["top"] + bx["bottom"]) / 2, top, bottom, band)[0] == "Discount"]
    valid_bear = [bx for bx in bear_breakers
                  if classify_zone((bx["top"] + bx["bottom"]) / 2, top, bottom, band)[0] == "Premium"]

    best_bull = max(valid_bull, key=lambda bx: bx["start"]) if valid_bull else None
    best_bear = max(valid_bear, key=lambda bx: bx["start"]) if valid_bear else None
    return best_bull, best_bear


# ---------------------------------------------------------------------------
# CHART SNAPSHOTS
# ---------------------------------------------------------------------------
def make_chart(df, zone_range, bull_ob, bear_ob, title, path, bull_breaker=None, bear_breaker=None):
    """
    Candles for the last CHART_BARS bars with the premium/discount range
    shaded (red = premium, green = discount, gray = equilibrium), the
    strong/weak swing labels on the top/bottom boundary, both 0.71 fib
    entry levels, any unmitigated order blocks (blue/orange), and — 4H
    only — a zone-filtered breaker block (teal = bullish support breaker
    in discount, crimson = bearish resistance breaker in premium). Saved
    to `path`.
    """
    plot_df = df.tail(CHART_BARS).set_index("ts")[["open", "high", "low", "close", "volume"]]

    top, bottom = zone_range["top"], zone_range["bottom"]
    eq_hi = 0.525 * top + 0.475 * bottom
    eq_lo = 0.525 * bottom + 0.475 * top

    fig, axes = mpf.plot(
        plot_df, type="candle", style="charles", volume=True,
        title=title, returnfig=True, figsize=(10, 6),
    )
    ax = axes[0]
    ax.axhspan(eq_hi, top, color="red", alpha=0.08)
    ax.axhspan(bottom, eq_lo, color="green", alpha=0.08)
    ax.axhspan(eq_lo, eq_hi, color="gray", alpha=0.06)
    ax.axhline(top, color="red", linestyle="--", linewidth=0.8)
    ax.axhline(bottom, color="green", linestyle="--", linewidth=0.8)
    ax.axhline(eq_hi, color="gray", linestyle=":", linewidth=0.6)
    ax.axhline(eq_lo, color="gray", linestyle=":", linewidth=0.6)

    x_right = len(plot_df) - 1
    ax.text(x_right, top, f" {zone_range['top_label']} {top:.2f}", va="bottom", ha="right",
            fontsize=8, color="darkred")
    ax.text(x_right, bottom, f" {zone_range['bottom_label']} {bottom:.2f}", va="top", ha="right",
            fontsize=8, color="darkgreen")

    fib_long, fib_short = zone_range.get("fib_long"), zone_range.get("fib_short")
    if fib_long is not None:
        emphasize = zone_range.get("with_trend_dir") == "LONG"
        ax.axhline(fib_long, color="lime", linestyle="-" if emphasize else "--",
                   linewidth=1.8 if emphasize else 1.1, zorder=5)
        tag = " (with-trend)" if emphasize else " (counter-trend)" if zone_range.get("counter_dir") == "LONG" else ""
        ax.text(0, fib_long, f" Fib .71 Long{tag} — {fib_long:.2f} ", va="center", ha="left",
                fontsize=8, fontweight="bold" if emphasize else "normal", color="darkgreen",
                backgroundcolor="white")
    if fib_short is not None:
        emphasize = zone_range.get("with_trend_dir") == "SHORT"
        ax.axhline(fib_short, color="magenta", linestyle="-" if emphasize else "--",
                   linewidth=1.8 if emphasize else 1.1, zorder=5)
        tag = " (with-trend)" if emphasize else " (counter-trend)" if zone_range.get("counter_dir") == "SHORT" else ""
        ax.text(0, fib_short, f" Fib .71 Short{tag} — {fib_short:.2f} ", va="center", ha="left",
                fontsize=8, fontweight="bold" if emphasize else "normal", color="purple",
                backgroundcolor="white")

    if bull_ob:
        ax.axhspan(bull_ob["low"], bull_ob["high"], color="blue", alpha=0.15)
    if bear_ob:
        ax.axhspan(bear_ob["low"], bear_ob["high"], color="orange", alpha=0.15)

    if bull_breaker:
        ax.axhspan(bull_breaker["bottom"], bull_breaker["top"], facecolor="teal", alpha=0.28,
                   edgecolor="teal", linewidth=1.2, zorder=4)
        mid = (bull_breaker["top"] + bull_breaker["bottom"]) / 2
        ax.text(x_right, mid, f" Bull Breaker (discount) {bull_breaker['bottom']:.2f}-{bull_breaker['top']:.2f} ",
                va="center", ha="right", fontsize=8, fontweight="bold", color="teal", backgroundcolor="white")
    if bear_breaker:
        ax.axhspan(bear_breaker["bottom"], bear_breaker["top"], facecolor="crimson", alpha=0.22,
                   edgecolor="crimson", linewidth=1.2, zorder=4)
        mid = (bear_breaker["top"] + bear_breaker["bottom"]) / 2
        ax.text(x_right, mid, f" Bear Breaker (premium) {bear_breaker['bottom']:.2f}-{bear_breaker['top']:.2f} ",
                va="center", ha="right", fontsize=8, fontweight="bold", color="crimson", backgroundcolor="white")

    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# ANALYSIS
# ---------------------------------------------------------------------------
def analyze():
    daily = fetch_ohlcv(SYMBOL, "1d", 500)
    h4 = fetch_ohlcv(SYMBOL, "4h", 500)

    price = float(h4["close"].iloc[-1])

    d_range = compute_structure(daily, SWING_LENGTH_DAILY)
    h_range = compute_structure(h4, SWING_LENGTH_4H)

    d_zone, d_pct = classify_zone(price, d_range["top"], d_range["bottom"])
    h_zone, h_pct = classify_zone(price, h_range["top"], h_range["bottom"])

    d_bull_ob, d_bear_ob = find_order_blocks(daily, OB_LOOKBACK)
    h_bull_ob, h_bear_ob = find_order_blocks(h4, OB_LOOKBACK)

    # 4H only, per your call — daily/LTF breakers aren't computed at all.
    h_bull_breaker, h_bear_breaker = find_valid_breakers(
        h4, h_range["top"], h_range["bottom"],
        BREAKER_Z_LEN, BREAKER_Z_THRESHOLD, BREAKER_MAX_AGE,
    )

    aligned = d_zone in ("Premium", "Discount") and d_zone == h_zone
    if aligned:
        bias = "LONG" if d_zone == "Discount" else "SHORT"
        counter_bias = "SHORT" if bias == "LONG" else "LONG"
        watch_fib = h_range["fib_long"] if bias == "LONG" else h_range["fib_short"]
        counter_fib = h_range["fib_short"] if bias == "LONG" else h_range["fib_long"]
        watch_ob = h_bull_ob if bias == "LONG" else h_bear_ob
        counter_ob = h_bear_ob if bias == "LONG" else h_bull_ob
    else:
        bias, counter_bias = "NO ALIGNMENT", None
        watch_fib = counter_fib = watch_ob = counter_ob = None

    data = {
        "symbol": SYMBOL, "price": price,
        "daily": {"zone": d_zone, "pct": d_pct, **d_range, "bull_ob": d_bull_ob, "bear_ob": d_bear_ob},
        "h4": {"zone": h_zone, "pct": h_pct, **h_range, "bull_ob": h_bull_ob, "bear_ob": h_bear_ob,
               "bull_breaker": h_bull_breaker, "bear_breaker": h_bear_breaker},
        "aligned": aligned,
        "bias": bias, "watch_fib": watch_fib, "watch_ob": watch_ob,
        "counter_bias": counter_bias, "counter_fib": counter_fib, "counter_ob": counter_ob,
    }
    return data, daily, h4


# ---------------------------------------------------------------------------
# SUMMARY (LLM, with a templated fallback if no API key)
# ---------------------------------------------------------------------------
def fmt_ob(ob):
    return f"{ob['low']:.2f}-{ob['high']:.2f}" if ob else "none unmitigated"


def fmt_breaker(bx):
    return f"{bx['bottom']:.2f}-{bx['top']:.2f}" if bx else "none active in zone"


def fallback_summary(d):
    lines = [
        f"{d['symbol']} @ {d['price']:.2f}",
        f"Daily: {d['daily']['zone']} ({d['daily']['pct']}% of range) — "
        f"{d['daily']['top_label']} {d['daily']['top']:.2f} / {d['daily']['bottom_label']} {d['daily']['bottom']:.2f}",
        f"4H: {d['h4']['zone']} ({d['h4']['pct']}% of range) — "
        f"{d['h4']['top_label']} {d['h4']['top']:.2f} / {d['h4']['bottom_label']} {d['h4']['bottom']:.2f}",
        "",
        f"LONG setup — 4H fib .71 buy zone: {d['h4']['fib_long']:.2f} | 4H bullish OB: {fmt_ob(d['h4']['bull_ob'])} "
        f"| 4H bull breaker (discount only): {fmt_breaker(d['h4']['bull_breaker'])}",
        f"SHORT setup — 4H fib .71 sell zone: {d['h4']['fib_short']:.2f} | 4H bearish OB: {fmt_ob(d['h4']['bear_ob'])} "
        f"| 4H bear breaker (premium only): {fmt_breaker(d['h4']['bear_breaker'])}",
    ]
    if d["aligned"]:
        lines.append(f"\nAligned -> with-trend bias is {d['bias']} at {d['watch_fib']:.2f}. "
                     f"If you want to fade it instead, the counter-trend {d['counter_bias']} zone is {d['counter_fib']:.2f}.")
    else:
        lines.append("\nDaily and 4H aren't in the same zone right now, so there's no with-trend "
                     "alignment — either setup above would be counter to one of the two timeframes.")
    return "\n".join(lines)


def llm_summary(d):
    if not (ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY):
        return fallback_summary(d)

    prompt = f"""Break down this SMC premium/discount state for {d['symbol']} as a Telegram trade brief covering BOTH possible directions — the person wants to see the long setup and the short setup every time, not just whichever one is aligned, so they can also choose to fade the aligned move if they want.

Price: {d['price']:.2f}

Daily zone: {d['daily']['zone']} ({d['daily']['pct']}% of range), range {d['daily']['bottom']:.2f}-{d['daily']['top']:.2f}
Daily structure: {d['daily']['top_label']} at {d['daily']['top']:.2f}, {d['daily']['bottom_label']} at {d['daily']['bottom']:.2f} (bias: {d['daily']['bias'] or 'unconfirmed'})

4H zone: {d['h4']['zone']} ({d['h4']['pct']}% of range), range {d['h4']['bottom']:.2f}-{d['h4']['top']:.2f}
4H structure: {d['h4']['top_label']} at {d['h4']['top']:.2f}, {d['h4']['bottom_label']} at {d['h4']['bottom']:.2f} (bias: {d['h4']['bias'] or 'unconfirmed'})

4H LONG setup: fib .71 buy zone at {d['h4']['fib_long']:.2f}, bullish OB {fmt_ob(d['h4']['bull_ob'])}, bull breaker (only valid if sitting in discount) {fmt_breaker(d['h4']['bull_breaker'])}
4H SHORT setup: fib .71 sell zone at {d['h4']['fib_short']:.2f}, bearish OB {fmt_ob(d['h4']['bear_ob'])}, bear breaker (only valid if sitting in premium) {fmt_breaker(d['h4']['bear_breaker'])}

Daily/4H alignment: {'YES — with-trend bias is ' + d['bias'] + ', counter-trend fade would be ' + str(d['counter_bias']) if d['aligned'] else 'NO — daily and 4H are in different zones right now'}

Write a clear, structured plain-text brief (no markdown headers, no disclaimers) with these parts, each 1-3 lines:
1. Daily and 4H state — zone, and which side is the Strong vs Weak swing on each
2. LONG setup — the fib .71 buy price, OB confluence, whether a valid bull breaker (discount-zone only) backs it up, and whether it's the with-trend or counter-trend trade right now
3. SHORT setup — the fib .71 sell price, OB confluence, whether a valid bear breaker (premium-zone only) backs it up, and whether it's the with-trend or counter-trend trade right now
4. Bottom line — which direction has HTF/LTF alignment behind it, and what the honest case for fading it would look like instead"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if b.type == "text").strip()
    except Exception as e:
        log.warning(f"LLM summary failed, falling back to template: {e}")
        return fallback_summary(d)


# ---------------------------------------------------------------------------
# TELEGRAM
# ---------------------------------------------------------------------------
def send_telegram(text):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log.warning("Telegram not configured — printing instead:\n%s", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text}, timeout=10)
        r.raise_for_status()
    except Exception as e:
        log.error(f"Telegram send failed: {e}")


def send_telegram_photo(path, caption=""):
    if not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID):
        log.warning("Telegram not configured — skipping photo send for %s", path)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            r = requests.post(
                url, data={"chat_id": TELEGRAM_CHAT_ID, "caption": caption[:1024]},
                files={"photo": f}, timeout=20,
            )
        r.raise_for_status()
    except Exception as e:
        log.error(f"Telegram photo send failed: {e}")


# ---------------------------------------------------------------------------
# STATE PERSISTENCE (avoid duplicate alerts across restarts)
# ---------------------------------------------------------------------------
def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"daily_zone": None, "h4_zone": None, "last_post_ts": 0}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def run_once(state, force=False):
    data, daily_df, h4_df = analyze()
    changed = (data["daily"]["zone"] != state["daily_zone"]) or (data["h4"]["zone"] != state["h4_zone"])
    stale = (time.time() - state["last_post_ts"]) > HEARTBEAT_HOURS * 3600

    if force or changed or stale:
        text = llm_summary(data)

        safe_symbol = SYMBOL.replace("/", "_").replace(":", "_")
        daily_chart = os.path.join(CHART_DIR, f"{safe_symbol}_daily.png")
        h4_chart = os.path.join(CHART_DIR, f"{safe_symbol}_4h.png")
        make_chart(daily_df, data["daily"], data["daily"]["bull_ob"], data["daily"]["bear_ob"],
                   f"{SYMBOL} Daily — {data['daily']['zone']}", daily_chart)
        make_chart(h4_df, data["h4"], data["h4"]["bull_ob"], data["h4"]["bear_ob"],
                   f"{SYMBOL} 4H — {data['h4']['zone']}", h4_chart,
                   bull_breaker=data["h4"]["bull_breaker"], bear_breaker=data["h4"]["bear_breaker"])

        send_telegram(text)
        send_telegram_photo(daily_chart, caption=f"{SYMBOL} Daily — {data['daily']['zone']}")
        send_telegram_photo(h4_chart, caption=f"{SYMBOL} 4H — {data['h4']['zone']}")

        log.info("Posted update (%s)", "startup" if force else "change" if changed else "heartbeat")
        state["last_post_ts"] = time.time()
    else:
        log.info("No change: daily=%s 4h=%s", data["daily"]["zone"], data["h4"]["zone"])

    state["daily_zone"] = data["daily"]["zone"]
    state["h4_zone"] = data["h4"]["zone"]
    save_state(state)
    return data


def main():
    state = load_state()
    log.info("Startup — running initial analysis for %s", SYMBOL)
    run_once(state, force=True)

    while True:
        time.sleep(POLL_SECONDS)
        try:
            run_once(state)
        except Exception as e:
            log.error(f"Analysis loop error: {e}")


if __name__ == "__main__":
    main()
