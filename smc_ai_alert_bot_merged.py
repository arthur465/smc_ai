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


def compute_trailing_range(df, length):
    """
    Walks the dataframe bar-by-bar maintaining trailing top/bottom exactly like
    the indicator: re-anchor the side whose leg just confirmed, then let both
    sides expand to the current bar's high/low. Returns the range as of the
    most recent bar.
    """
    n = len(df)
    high, low = df["high"].values, df["low"].values
    ts = df["ts"].values
    legs = compute_legs(df, length)

    trailing_top = float(high[0])
    trailing_bottom = float(low[0])
    top_time = ts[0]
    bottom_time = ts[0]

    for i in range(n):
        if i >= length and legs[i] != legs[i - 1]:
            src_i = i - length
            if legs[i] == 1:  # new bullish leg -> swing low just confirmed
                trailing_bottom = float(low[src_i])
                bottom_time = ts[src_i]
            else:  # new bearish leg -> swing high just confirmed
                trailing_top = float(high[src_i])
                top_time = ts[src_i]

        if high[i] >= trailing_top:
            trailing_top = float(high[i])
            top_time = ts[i]
        if low[i] <= trailing_bottom:
            trailing_bottom = float(low[i])
            bottom_time = ts[i]

    return {
        "top": trailing_top, "top_time": pd.Timestamp(top_time),
        "bottom": trailing_bottom, "bottom_time": pd.Timestamp(bottom_time),
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
# CHART SNAPSHOTS
# ---------------------------------------------------------------------------
def make_chart(df, zone_range, bull_ob, bear_ob, title, path):
    """
    Candles for the last CHART_BARS bars with the premium/discount range
    shaded (red = premium, green = discount, gray = equilibrium) and any
    unmitigated order blocks overlaid as blue (bullish) / orange (bearish)
    bands. Saved to `path` as a PNG.
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

    if bull_ob:
        ax.axhspan(bull_ob["low"], bull_ob["high"], color="blue", alpha=0.15)
    if bear_ob:
        ax.axhspan(bear_ob["low"], bear_ob["high"], color="orange", alpha=0.15)

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

    d_range = compute_trailing_range(daily, SWING_LENGTH_DAILY)
    h_range = compute_trailing_range(h4, SWING_LENGTH_4H)

    d_zone, d_pct = classify_zone(price, d_range["top"], d_range["bottom"])
    h_zone, h_pct = classify_zone(price, h_range["top"], h_range["bottom"])

    d_bull_ob, d_bear_ob = find_order_blocks(daily, OB_LOOKBACK)
    h_bull_ob, h_bear_ob = find_order_blocks(h4, OB_LOOKBACK)

    aligned = d_zone in ("Premium", "Discount") and d_zone == h_zone
    if aligned:
        bias = "LONG" if d_zone == "Discount" else "SHORT"
        watch_ob = h_bull_ob if bias == "LONG" else h_bear_ob
    else:
        bias, watch_ob = "NO ALIGNMENT", None

    data = {
        "symbol": SYMBOL, "price": price,
        "daily": {"zone": d_zone, "pct": d_pct, **d_range, "bull_ob": d_bull_ob, "bear_ob": d_bear_ob},
        "h4": {"zone": h_zone, "pct": h_pct, **h_range, "bull_ob": h_bull_ob, "bear_ob": h_bear_ob},
        "aligned": aligned, "bias": bias, "watch_ob": watch_ob,
    }
    return data, daily, h4


# ---------------------------------------------------------------------------
# SUMMARY (LLM, with a templated fallback if no API key)
# ---------------------------------------------------------------------------
def fmt_ob(ob):
    return f"{ob['low']:.2f}-{ob['high']:.2f}" if ob else "none unmitigated"


def fallback_summary(d):
    lines = [
        f"{d['symbol']} @ {d['price']:.2f}",
        f"Daily: {d['daily']['zone']} ({d['daily']['pct']}% of range {d['daily']['bottom']:.2f}-{d['daily']['top']:.2f})",
        f"4H: {d['h4']['zone']} ({d['h4']['pct']}% of range {d['h4']['bottom']:.2f}-{d['h4']['top']:.2f})",
    ]
    if d["aligned"]:
        lines.append(f"Aligned -> bias {d['bias']}. Watch 4H OB: {fmt_ob(d['watch_ob'])}")
    else:
        lines.append("Not aligned — wait for daily and 4H to agree before sizing up.")
    return "\n".join(lines)


def llm_summary(d):
    if not (ANTHROPIC_AVAILABLE and ANTHROPIC_API_KEY):
        return fallback_summary(d)

    prompt = f"""Summarize this SMC premium/discount state for {d['symbol']} as a short Telegram trade brief.

Price: {d['price']:.2f}
Daily zone: {d['daily']['zone']} ({d['daily']['pct']}% of range), range {d['daily']['bottom']:.2f}-{d['daily']['top']:.2f}
4H zone: {d['h4']['zone']} ({d['h4']['pct']}% of range), range {d['h4']['bottom']:.2f}-{d['h4']['top']:.2f}
Alignment: {'YES, bias ' + d['bias'] if d['aligned'] else 'NO'}
Daily bullish OB: {fmt_ob(d['daily']['bull_ob'])}
Daily bearish OB: {fmt_ob(d['daily']['bear_ob'])}
4H bullish OB: {fmt_ob(d['h4']['bull_ob'])}
4H bearish OB: {fmt_ob(d['h4']['bear_ob'])}

Write 5-8 lines, plain text, no markdown headers, no disclaimers:
- current daily and 4H zone state
- whether they're aligned and what bias that implies
- if aligned, the specific order block price levels to watch for entries
- if not aligned, what needs to happen for alignment"""

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=400,
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
                   f"{SYMBOL} 4H — {data['h4']['zone']}", h4_chart)

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
