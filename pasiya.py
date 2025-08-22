import os
import time
import json
import math
import queue
import pytz
import uuid
import hmac
import hashlib
import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler

# ----------------------------
# Config & Constants
# ----------------------------
TZ = pytz.timezone("Asia/Colombo")
OANDA_ENV = os.getenv("OANDA_ENV", "practice").lower()
OANDA_DOMAIN = "https://api-fxtrade.oanda.com" if OANDA_ENV == "live" else "https://api-fxpractice.oanda.com"
OANDA_API_KEY = os.getenv("#IDbc5rf6D3", "")
OANDA_ACCOUNT_ID = os.getenv("1600009265", "")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
ADMIN_TELEGRAM_ID = os.getenv("ADMIN_TELEGRAM_ID")

DEFAULT_SYMBOLS = ["EUR_USD", "GBP_USD", "USD_JPY"]
CANDLE_GRANULARITY = "M5"  # M1/M5/M15/M30/H1/H4/D
LOOKBACK = 500  # candles to pull for features
RISK_PER_TRADE = 0.01  # 1%
RR_TP = 2.0  # take profit RR
RR_SL = 1.0  # stop loss RR
AUTO_TRADING = True
DB_PATH = "forex_bot.db"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger("forex-bot")

# ----------------------------
# Utilities
# ----------------------------

def now_colombo():
    return datetime.now(TZ)


def today_bounds_colombo():
    today = now_colombo().date()
    start = TZ.localize(datetime.combine(today, datetime.min.time()))
    end = TZ.localize(datetime.combine(today, datetime.max.time()))
    return start, end


def iso(dt: datetime) -> str:
    return dt.astimezone(pytz.UTC).isoformat()


# ----------------------------
# Storage
# ----------------------------

class Store:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self._init()

    def _init(self):
        c = self.conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS trades (
              id TEXT PRIMARY KEY,
              time TEXT,
              symbol TEXT,
              side TEXT,
              units REAL,
              entry REAL,
              sl REAL,
              tp REAL,
              exit_time TEXT,
              exit_price REAL,
              pnl REAL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS equity (
              time TEXT,
              balance REAL,
              nav REAL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS settings (
              key TEXT PRIMARY KEY,
              value TEXT
            )
            """
        )
        self.conn.commit()

    def set(self, key: str, value: str):
        c = self.conn.cursor()
        c.execute("REPLACE INTO settings(key, value) VALUES(?, ?)", (key, value))
        self.conn.commit()

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        c = self.conn.cursor()
        c.execute("SELECT value FROM settings WHERE key=?", (key,))
        row = c.fetchone()
        return row[0] if row else default

    def log_trade_open(self, tid: str, t: datetime, symbol: str, side: str, units: float, entry: float, sl: float, tp: float):
        c = self.conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO trades(id,time,symbol,side,units,entry,sl,tp,exit_time,exit_price,pnl) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (tid, iso(t), symbol, side, units, entry, sl, tp, None, None, None),
        )
        self.conn.commit()

    def log_trade_close(self, tid: str, exit_time: datetime, exit_price: float, pnl: float):
        c = self.conn.cursor()
        c.execute(
            "UPDATE trades SET exit_time=?, exit_price=?, pnl=? WHERE id=?",
            (iso(exit_time), exit_price, pnl, tid),
        )
        self.conn.commit()

    def log_equity(self, t: datetime, balance: float, nav: float):
        c = self.conn.cursor()
        c.execute("INSERT INTO equity(time, balance, nav) VALUES(?,?,?)", (iso(t), balance, nav))
        self.conn.commit()

    def pnl_today(self) -> float:
        start, end = today_bounds_colombo()
        c = self.conn.cursor()
        c.execute(
            "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE exit_time IS NOT NULL AND time BETWEEN ? AND ?",
            (iso(start), iso(end)),
        )
        v = c.fetchone()[0]
        return float(v or 0.0)

store = Store(DB_PATH)

# ----------------------------
# OANDA Client (minimal)
# ----------------------------

class Oanda:
    def __init__(self, api_key: str, account_id: str, domain: str):
        self.api_key = api_key
        self.account_id = account_id
        self.domain = domain
        self.h = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def candles(self, instrument: str, count: int, granularity: str) -> pd.DataFrame:
        url = f"{self.domain}/v3/instruments/{instrument}/candles"
        params = {"count": count, "granularity": granularity, "price": "M"}
        r = requests.get(url, headers=self.h, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()["candles"]
        rows = []
        for c in data:
            rows.append({
                "time": pd.to_datetime(c["time"]),
                "open": float(c["mid"]["o"]),
                "high": float(c["mid"]["h"]),
                "low": float(c["mid"]["l"]),
                "close": float(c["mid"]["c"]),
                "complete": c["complete"],
            })
        df = pd.DataFrame(rows).set_index("time")
        return df

    def account_summary(self) -> Dict:
        url = f"{self.domain}/v3/accounts/{self.account_id}/summary"
        r = requests.get(url, headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json()["account"]

    def positions(self) -> List[Dict]:
        url = f"{self.domain}/v3/accounts/{self.account_id}/openPositions"
        r = requests.get(url, headers=self.h, timeout=30)
        r.raise_for_status()
        return r.json().get("positions", [])

    def close_all(self) -> int:
        positions = self.positions()
        closed = 0
        for p in positions:
            instrument = p["instrument"]
            # market close both sides
            for side in ["long", "short"]:
                if float(p.get(side, {}).get("units", "0")) != 0:
                    url = f"{self.domain}/v3/accounts/{self.account_id}/positions/{instrument}/close"
                    payload = {"longUnits": "ALL"} if side == "long" else {"shortUnits": "ALL"}
                    r = requests.put(url, headers=self.h, data=json.dumps(payload), timeout=30)
                    if r.status_code in (200, 201):
                        closed += 1
        return closed

    def market_order(self, instrument: str, units: int, sl_price: Optional[float], tp_price: Optional[float]) -> Tuple[str, float]:
        url = f"{self.domain}/v3/accounts/{self.account_id}/orders"
        payload = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "timeInForce": "FOK",
                "positionFill": "DEFAULT",
            }
        }
        if sl_price:
            payload["order"]["stopLossOnFill"] = {"price": f"{sl_price:.5f}"}
        if tp_price:
            payload["order"]["takeProfitOnFill"] = {"price": f"{tp_price:.5f}"}
        r = requests.post(url, headers=self.h, data=json.dumps(payload), timeout=30)
        r.raise_for_status()
        resp = r.json()
        fill = resp.get("orderFillTransaction") or resp.get("orderCreateTransaction")
        trade_id = fill.get("tradeOpened", {}).get("tradeID") or fill.get("id", str(uuid.uuid4()))
        price = float(fill.get("price", 0.0))
        return trade_id, price

    def prices(self, instrument: str) -> float:
        url = f"{self.domain}/v3/accounts/{self.account_id}/pricing"
        r = requests.get(url, headers=self.h, params={"instruments": instrument}, timeout=30)
        r.raise_for_status()
        bids = r.json()["prices"][0]["bids"][0]["price"]
        asks = r.json()["prices"][0]["asks"][0]["price"]
        return (float(bids) + float(asks)) / 2.0


# ----------------------------
# Indicators & ML
# ----------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / (ma_down + 1e-12)
    return 100 - (100 / (1 + rs))

class AISignal:
    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200))
        ])
        self.fitted = False

    def features(self, df: pd.DataFrame) -> pd.DataFrame:
        x = pd.DataFrame(index=df.index)
        x["ret"] = df["close"].pct_change()
        x["ema_fast"] = ema(df["close"], 9)
        x["ema_slow"] = ema(df["close"], 21)
        x["rsi"] = rsi(df["close"], 14)
        x["ema_diff"] = x["ema_fast"] - x["ema_slow"]
        x["ema_cross"] = (x["ema_fast"] > x["ema_slow"]).astype(int)
        x = x.dropna()
        return x

    def make_labels(self, df: pd.DataFrame, horizon: int = 3, thresh: float = 0.0003) -> pd.Series:
        fwd = df["close"].pct_change(horizon).shift(-horizon)
        y = (fwd > thresh).astype(int)
        return y.loc[self.features(df).index]

    def fit(self, df: pd.DataFrame):
        x = self.features(df)
        y = self.make_labels(df)
        y = y.loc[x.index]
        if len(x) < 200:
            return
        self.model.fit(x, y)
        self.fitted = True

    def predict_signal(self, df: pd.DataFrame) -> int:
        x = self.features(df).iloc[-1:]
        try:
            proba = self.model.predict_proba(x)[0, 1]
        except NotFittedError:
            return 0
        # 1 = buy, -1 = sell, 0 = flat
        if proba > 0.55:
            return 1
        elif proba < 0.45:
            return -1
        return 0

ai = AISignal()

# ----------------------------
# Trading Engine
# ----------------------------

@dataclass
class EngineState:
    symbols: List[str] = field(default_factory=lambda: DEFAULT_SYMBOLS.copy())
    risk: float = RISK_PER_TRADE
    rr_tp: float = RR_TP
    rr_sl: float = RR_SL
    auto: bool = AUTO_TRADING

state = EngineState()

oanda = Oanda(OANDA_API_KEY, OANDA_ACCOUNT_ID, OANDA_DOMAIN)


def position_size(price: float, balance: float, risk: float, sl_pips: float, pip_value_per_unit: float = 0.0001) -> int:
    # very simplified sizing for majors quoted to 5 decimals (EUR_USD etc.)
    risk_amount = balance * risk
    per_unit_loss = sl_pips * pip_value_per_unit
    if per_unit_loss <= 0:
        return 0
    units = int(max(0, risk_amount / per_unit_loss))
    return units


def compute_sl_tp(entry: float, side: int, rr_sl: float, rr_tp: float, atr_pips: float = 10) -> Tuple[float, float]:
    # simple SL/TP using ATR-like fixed pips
    sl_pips = atr_pips * rr_sl
    tp_pips = atr_pips * rr_tp
    pip = 0.0001
    if side > 0:
        sl = entry - sl_pips * pip
        tp = entry + tp_pips * pip
    else:
        sl = entry + sl_pips * pip
        tp = entry - tp_pips * pip
    return round(sl, 5), round(tp, 5)


def backfill_and_train(symbol: str):
    df = oanda.candles(symbol, LOOKBACK, CANDLE_GRANULARITY)
    ai.fit(df)


def evaluate_signal(symbol: str) -> int:
    df = oanda.candles(symbol, LOOKBACK, CANDLE_GRANULARITY)
    # Fallback rule
    ema_fast = ema(df["close"], 9)
    ema_slow = ema(df["close"], 21)
    rule_sig = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
    # AI overlay
    ai_sig = ai.predict_signal(df)
    if ai.fitted and ai_sig != 0:
        return ai_sig
    return rule_sig


def trade_symbol(symbol: str) -> Optional[str]:
    try:
        sig = evaluate_signal(symbol)
        price = oanda.prices(symbol)
        acc = oanda.account_summary()
        balance = float(acc["balance"])
        sl, tp = compute_sl_tp(price, sig, state.rr_sl, state.rr_tp)
        units = position_size(price, balance, state.risk, sl_pips=10)
        if units == 0:
            return "Sizing resulted in 0 units."
        if sig < 0:
            units = -units
        tid, fill_price = oanda.market_order(symbol, units, sl, tp)
        store.log_trade_open(tid, datetime.utcnow(), symbol, "BUY" if sig>0 else "SELL", units, fill_price, sl, tp)
        return f"Opened {symbol} {'LONG' if sig>0 else 'SHORT'} {abs(units)} @ {fill_price:.5f} (SL {sl:.5f} / TP {tp:.5f})"
    except Exception as e:
        logger.exception("trade_symbol error")
        return f"Error: {e}"


# ----------------------------
# Telegram Admin Bot
# ----------------------------

ADMIN_ONLY_TEXT = "Admin only command."


def is_admin(update: Update) -> bool:
    if ADMIN_TELEGRAM_ID is None:
        return True  # open until set
    try:
        return str(update.effective_user.id) == str(ADMIN_TELEGRAM_ID)
    except Exception:
        return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    await update.message.reply_text(
        """🤖 *AI Forex Bot* Online

/Status – bot state
/Toggle – start/stop auto trading
/Symbol – set/list symbols (e.g. `/symbol EUR_USD GBP_USD`)
/Risk – set risk per trade (e.g. `/risk 0.01`)
/TP – set take-profit RR (e.g. `/tp 2.0`)
/SL – set stop-loss RR (e.g. `/sl 1.0`)
/Open – force market order (`/open EUR_USD buy` or `sell`)
/CloseAll – close all positions
/PNL_Today – realized PnL since midnight 🇱🇰
/SetAdmin – lock bot to your Telegram ID
""",
        parse_mode="Markdown",
    )

async def setadmin(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.args and is_admin(update):
        admin_id = context.args[0]
        os.environ["ADMIN_TELEGRAM_ID"] = admin_id
        global ADMIN_TELEGRAM_ID
        ADMIN_TELEGRAM_ID = admin_id
        await update.message.reply_text(f"✅ Admin set to {admin_id}")
    else:
        await update.message.reply_text("Usage: /setadmin <telegram_id>")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    try:
        acc = oanda.account_summary()
        balance = float(acc.get("balance", 0))
        nav = float(acc.get("NAV", balance))
        store.log_equity(datetime.utcnow(), balance, nav)
    except Exception:
        balance = nav = float('nan')
    open_positions = oanda.positions()
    text = (
        f"⚙️ Auto: {'ON' if state.auto else 'OFF'}\n"
        f"💼 Balance: {balance:.2f}\n"
        f"📊 NAV: {nav:.2f}\n"
        f"🎯 Risk: {state.risk:.2%}\n"
        f"🎯 RR (TP/SL): {state.rr_tp}/{state.rr_sl}\n"
        f"📈 Symbols: {', '.join(state.symbols)}\n"
        f"📂 Open positions: {len(open_positions)}"
    )
    await update.message.reply_text(text)

async def toggle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    state.auto = not state.auto
    await update.message.reply_text(f"Auto trading: {'ON' if state.auto else 'OFF'}")

async def symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    if context.args:
        state.symbols = [s.replace("/", "_").upper() for s in context.args]
        await update.message.reply_text(f"✅ Symbols set: {', '.join(state.symbols)}")
    else:
        await update.message.reply_text(f"Symbols: {', '.join(state.symbols)}")

async def risk(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    try:
        r = float(context.args[0])
        state.risk = max(0.001, min(0.05, r))
        await update.message.reply_text(f"✅ Risk set to {state.risk:.2%}")
    except Exception:
        await update.message.reply_text("Usage: /risk 0.01")

async def tp(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    try:
        v = float(context.args[0])
        state.rr_tp = max(0.2, min(5.0, v))
        await update.message.reply_text(f"✅ TP RR set to {state.rr_tp}")
    except Exception:
        await update.message.reply_text("Usage: /tp 2.0")

async def sl(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    try:
        v = float(context.args[0])
        state.rr_sl = max(0.2, min(5.0, v))
        await update.message.reply_text(f"✅ SL RR set to {state.rr_sl}")
    except Exception:
        await update.message.reply_text("Usage: /sl 1.0")

async def open_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    if len(context.args) < 2:
        await update.message.reply_text("Usage: /open EUR_USD buy|sell")
        return
    symbol = context.args[0].replace("/", "_").upper()
    side = context.args[1].lower()
    try:
        sig = 1 if side == "buy" else -1
        price = oanda.prices(symbol)
        acc = oanda.account_summary()
        balance = float(acc["balance"])
        sl, tpv = compute_sl_tp(price, sig, state.rr_sl, state.rr_tp)
        units = position_size(price, balance, state.risk, sl_pips=10)
        if sig < 0:
            units = -units
        tid, fill_price = oanda.market_order(symbol, units, sl, tpv)
        store.log_trade_open(tid, datetime.utcnow(), symbol, "BUY" if sig>0 else "SELL", units, fill_price, sl, tpv)
        await update.message.reply_text(f"✅ Opened {symbol} {side.upper()} {abs(units)} @ {fill_price:.5f}")
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def closeall(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    n = oanda.close_all()
    await update.message.reply_text(f"Closed {n} sides.")

async def pnl_today_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_admin(update):
        await update.message.reply_text(ADMIN_ONLY_TEXT)
        return
    pnl = store.pnl_today()
    await update.message.reply_text(f"📅 PnL Today (🇱🇰): {pnl:.2f}")

# ----------------------------
# Scheduler Loop
# ----------------------------

class Worker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.stop_flag = threading.Event()

    def run(self):
        # initial training
        try:
            backfill_and_train(DEFAULT_SYMBOLS[0])
        except Exception:
            logger.warning("Initial training failed; continuing with rules.")
        while not self.stop_flag.is_set():
            try:
                if state.auto:
                    for sym in state.symbols:
                        msg = trade_symbol(sym)
                        logger.info(msg)
                # log equity snapshot hourly
                if now_colombo().minute == 0:
                    try:
                        acc = oanda.account_summary()
                        store.log_equity(datetime.utcnow(), float(acc["balance"]), float(acc.get("NAV", acc["balance"])) )
                    except Exception:
                        pass
                # daily PnL at 23:59 LKT
                local = now_colombo()
                if local.hour == 23 and local.minute == 59:
                    pnl = store.pnl_today()
                    try:
                        text = f"📊 Daily PnL (🇱🇰 {local.date()}): {pnl:.2f}"
                        # cannot send directly from worker; print to logs; Telegram app pulls on /status
                        logger.info(text)
                    except Exception:
                        pass
            except Exception:
                logger.exception("Worker loop error")
            time.sleep(60)  # run each minute per symbol

worker = Worker()

# ----------------------------
# Main
# ----------------------------

def main():
    if not OANDA_API_KEY or not OANDA_ACCOUNT_ID or not TELEGRAM_TOKEN:
        logger.error("Missing env vars. Set OANDA_API_KEY, OANDA_ACCOUNT_ID, TELEGRAM_TOKEN.")
        return

    worker.start()

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("toggle", toggle))
    app.add_handler(CommandHandler("symbol", symbol))
    app.add_handler(CommandHandler("risk", risk))
    app.add_handler(CommandHandler("tp", tp))
    app.add_handler(CommandHandler("sl", sl))
    app.add_handler(CommandHandler("open", open_cmd))
    app.add_handler(CommandHandler("closeall", closeall))
    app.add_handler(CommandHandler("pnl_today", pnl_today_cmd))
    app.add_handler(CommandHandler("setadmin", setadmin))

    logger.info("Telegram bot running…")
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
