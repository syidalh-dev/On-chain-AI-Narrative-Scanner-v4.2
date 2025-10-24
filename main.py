# -*- coding: utf-8 -*-
"""
Smart On-chain + AI Narrative Scanner v4.2
Sources: CoinGecko (primary), CoinMarketCap (fallback), Binance, DexScreener, DeFiLlama, optional X/Twitter
Sends: Telegram alerts (Top K rare opportunities)
Notes:
 - Put secrets in environment/GitHub Secrets:
   TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, CMC_API_KEY (optional), X_BEARER_TOKEN (optional)
"""

import os
import time
import json
import requests
import pandas as pd
from datetime import datetime, timedelta

# Try import HuggingFace pipeline if available
try:
    from transformers import pipeline
    HF_AVAILABLE = True
except Exception:
    HF_AVAILABLE = False

# -------- CONFIG --------
MAX_MARKET_CAP = 50_000_000  # $50M max market cap for low-cap focus
MIN_CG_VOLUME = 10_000       # filter tiny-volume coins early (USD)
MIN_DEX_LIQUIDITY_USD = 100_000
MIN_DEX_VOLUME_24H_USD = 50_000
TOP_K = 3                    # number of signals to send
HISTORY_FILE = "sent_signals_history.json"  # local file to avoid duplicates in time window
NO_REPEAT_HOURS = 24         # don't resend same symbol within this many hours

COINGECKO_API = "https://api.coingecko.com/api/v3"

# -------- SECRETS (from env / GitHub Secrets) --------
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
CMC_API_KEY = os.getenv("CMC_API_KEY")        # optional
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN")  # optional

# -------- AI setup --------
sentiment_model = None
if HF_AVAILABLE:
    try:
        sentiment_model = pipeline("sentiment-analysis")
        print("✅ HuggingFace sentiment pipeline loaded")
    except Exception as e:
        print("⚠️ Failed to load HF pipeline:", e)
        sentiment_model = None
else:
    print("⚠️ transformers not installed — HF sentiment disabled (fallback will be used)")

# -------- helpers --------
def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("⚠️ failed to save history:", e)

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("⚠️ failed to load history:", e)
        return {}

def safe_get(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            d = d[k]
        else:
            return default
    return d

def ai_sentiment_score(texts):
    """Use HF pipeline if available, otherwise simple keyword fallback"""
    if not texts:
        return 0.0
    joined = []
    for t in texts:
        if not t:
            continue
        joined.append(str(t))
    if not joined:
        return 0.0
    if sentiment_model:
        try:
            res = sentiment_model(joined)
            scores = []
            for r in res:
                lbl = r.get("label","").upper()
                sc = float(r.get("score", 0.0) or 0.0)
                scores.append(sc if lbl.startswith("POS") else -sc)
            return sum(scores) / max(1, len(scores))
        except Exception as e:
            print("⚠️ HF sentiment error:", e)
    # fallback simple heuristic
    text = " ".join(joined).lower()
    pos = sum(text.count(w) for w in ["pump","bull","moon","gain","partnership","upgrade","listing","announce","launch"])
    neg = sum(text.count(w) for w in ["rug","scam","hack","dump","sell","takedown","fraud"])
    score = (pos - neg) / max(1, (pos + neg + 1))
    return max(-1.0, min(1.0, score))

# -------- Data source functions --------
def fetch_coingecko_markets(per_page=250, page=1):
    url = f"{COINGECKO_API}/coins/markets"
    params = {"vs_currency":"usd","order":"market_cap_asc","per_page":per_page,"page":page,"sparkline":"false"}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        df = pd.DataFrame(data)
        print(f"✅ CoinGecko markets page {page}: {len(df)} coins")
        return df
    except Exception as e:
        print("❌ CoinGecko fetch failed:", e)
        return pd.DataFrame()

def fetch_coingecko_coin_detail(cg_id):
    url = f"{COINGECKO_API}/coins/{cg_id}"
    params = {"localization":"false","tickers":"false","market_data":"false","community_data":"true","developer_data":"true","sparkline":"false"}
    try:
        r = requests.get(url, params=params, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # non-fatal
        return {}

def fetch_dexscreener_pairs():
    urls_try = [
        "https://api.dexscreener.com/latest/dex/pairs",
        "https://api.dexscreener.com/latest/dex/tokens",
        "https://api.dexscreener.com/latest"
    ]
    for url in urls_try:
        try:
            r = requests.get(url, timeout=20)
            if r.status_code != 200:
                continue
            js = r.json()
            pairs = js.get("pairs") if isinstance(js, dict) and "pairs" in js else (js if isinstance(js, list) else [])
            df = pd.DataFrame(pairs)
            print(f"✅ DexScreener fetched {len(df)} pairs from {url}")
            return df
        except Exception as e:
            print("⚠️ DexScreener attempt failed:", url, e)
            continue
    print("❌ DexScreener all endpoints failed")
    return pd.DataFrame()

def fetch_defillama_protocols():
    try:
        url = "https://api.llama.fi/protocols"
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        df = pd.DataFrame(r.json())
        print(f"✅ DeFiLlama fetched {len(df)} protocols")
        return df
    except Exception as e:
        print("⚠️ DeFiLlama failed:", e)
        return pd.DataFrame()

def fetch_binance_ticker_for_symbol(sym):
    # try SYMBOLUSDT -> SYMBOLBTC -> SYMBOLETH
    tries = [f"{sym}USDT", f"{sym}BTC", f"{sym}ETH"]
    for t in tries:
        url = f"https://api.binance.com/api/v3/ticker/24hr?symbol={t}"
        try:
            r = requests.get(url, timeout=8)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None

def fetch_cmc_listings(limit=50):
    if not CMC_API_KEY:
        return pd.DataFrame()
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {"X-CMC_PRO_API_KEY": CMC_API_KEY}
    params = {"limit": limit, "sort":"market_cap","convert":"USD"}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
        return pd.DataFrame(data)
    except Exception as e:
        print("⚠️ CMC fetch failed:", e)
        return pd.DataFrame()

def get_x_mentions(symbol, max_results=10):
    if not X_BEARER_TOKEN:
        return 0
    search_url = "https://api.twitter.com/2/tweets/search/recent"
    headers = {"Authorization": f"Bearer {X_BEARER_TOKEN}"}
    params = {"query": f"{symbol} -is:retweet lang:en", "max_results": max_results}
    try:
        r = requests.get(search_url, headers=headers, params=params, timeout=10)
        if r.status_code != 200:
            return 0
        data = r.json()
        return len(data.get("data", []))
    except Exception as e:
        print("⚠️ X fetch error:", e)
        return 0

# -------- Core scoring & selection --------
def analyze_and_select(coingecko_df, dex_df, llama_df, top_k=TOP_K):
    if coingecko_df is None or coingecko_df.empty:
        return pd.DataFrame()

    results = []
    # iterate over coin list but first filter by market cap and minimum volume
    for _, row in coingecko_df.iterrows():
        try:
            cg_id = row.get("id")
            symbol = str(row.get("symbol","")).upper()
            name = row.get("name","")
            price = float(row.get("current_price") or 0)
            market_cap = float(row.get("market_cap") or 0)
            total_volume = float(row.get("total_volume") or 0)
            community_score = float(row.get("community_score") or 0) if "community_score" in row else 0.0
            developer_score = float(row.get("developer_score") or 0) if "developer_score" in row else 0.0

            if market_cap <= 0 or market_cap > MAX_MARKET_CAP:
                continue
            if total_volume < MIN_CG_VOLUME:
                # allow some exceptions if community score high
                if community_score < 20:
                    continue

            # get detailed coin info once for contract addresses & description
            detail = fetch_coingecko_coin_detail(cg_id) if cg_id else {}
            # get contract addresses (platforms)
            platforms = detail.get("platforms", {}) if isinstance(detail, dict) else {}
            contract_addrs = {k:v for k,v in platforms.items() if v}  # chain->address
            # short description
            desc = detail.get("description", {}).get("en","") if isinstance(detail, dict) else ""

            # DEX matching: try to find liquidity/volume in dex_df by contract address or symbol/pairName
            dex_liq = 0.0
            dex_vol24 = 0.0
            if not dex_df.empty:
                try:
                    # search by contract addresses if available
                    found = pd.DataFrame()
                    for addr in contract_addrs.values():
                        if not addr:
                            continue
                        # DexScreener pair structure may include baseToken or token0/1; try contains
                        mask = dex_df.apply(lambda x: (isinstance(x.get("baseToken"), dict) and x.get("baseToken", {}).get("contractAddress","").lower()==addr.lower()) 
                                            or (isinstance(x.get("quoteToken"), dict) and x.get("quoteToken", {}).get("contractAddress","").lower()==addr.lower()), axis=1)
                        tmp = dex_df[mask]
                        if not tmp.empty:
                            found = pd.concat([found, tmp], ignore_index=True)
                    # fallback: match by symbol in pairName or baseToken.symbol
                    if found.empty:
                        mask2 = dex_df.apply(lambda x: (isinstance(x.get("baseToken"), dict) and x.get("baseToken", {}).get("symbol","").upper()==symbol) 
                                             or (isinstance(x.get("pairName"), str) and symbol in x.get("pairName","").upper()), axis=1)
                        found = dex_df[mask2]
                    if not found.empty:
                        # compute max liquidity across matches
                        liqs = found.apply(lambda x: safe_get(x.to_dict(), "liquidity", "usd", default=0) or 0, axis=1).tolist()
                        vols = found.apply(lambda x: safe_get(x.to_dict(), "volume", "h24", default=0) or 0, axis=1).tolist()
                        dex_liq = max([float(v or 0) for v in liqs]) if liqs else 0.0
                        dex_vol24 = max([float(v or 0) for v in vols]) if vols else 0.0
                except Exception as e:
                    dex_liq = 0.0
                    dex_vol24 = 0.0

            # DeFiLlama presence
            onchain_presence = False
            if not llama_df.empty:
                try:
                    # match by symbol in llama's symbol column if exists
                    if "symbol" in llama_df.columns:
                        onchain_presence = any(llama_df["symbol"].fillna("").str.upper() == symbol)
                except Exception:
                    onchain_presence = False

            # Binance verification
            bin_info = fetch_binance_ticker_for_symbol(symbol)
            binance_vol = 0.0
            try:
                if bin_info:
                    # Binance returns quoteVolume sometimes depending on pair
                    binance_vol = float(bin_info.get("quoteVolume", 0) or bin_info.get("volume", 0) or 0)
            except Exception:
                binance_vol = 0.0

            # Narrative: X mentions + AI sentiment (name + short desc)
            x_mentions = get_x_mentions(symbol) if X_BEARER_TOKEN else 0
            ai_score = ai_sentiment_score([name, desc[:300]])

            # compute component scores
            tech_score = 1.0 if (dex_liq >= MIN_DEX_LIQUIDITY_USD and (dex_vol24 >= MIN_DEX_VOLUME_24H_USD or binance_vol >= MIN_DEX_VOLUME_24H_USD)) else 0.35
            social_score = min(1.0, (community_score + developer_score) / 200.0)  # normalize
            narrative_score = max(-1.0, min(1.0, (ai_score + (x_mentions/10.0)) / 2.0))  # normalize roughly
            onchain_score = 1.0 if onchain_presence else 0.0

            total_score = (0.35 * tech_score) + (0.30 * social_score) + (0.25 * max(0.0, narrative_score)) + (0.10 * onchain_score)

            results.append({
                "symbol": symbol,
                "id": cg_id,
                "name": name,
                "price": price,
                "market_cap": market_cap,
                "total_volume": total_volume,
                "dex_liquidity": dex_liq,
                "dex_volume_24h": dex_vol24,
                "binance_volume": binance_vol,
                "x_mentions": x_mentions,
                "narrative_score": narrative_score,
                "tech_score": tech_score,
                "social_score": social_score,
                "onchain_score": onchain_score,
                "total_score": total_score
            })
        except Exception as e:
            print("⚠️ coin loop error:", e)
            continue

    if not results:
        return pd.DataFrame()
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by="total_score", ascending=False).head(top_k)
    print(f"✅ selected top {len(df_sorted)} signals")
    return df_sorted

# -------- Telegram sending --------
def send_telegram(message, max_retries=2):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram secrets missing. Add TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    for attempt in range(max_retries+1):
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200:
                print("📤 Telegram message sent")
                return True
            else:
                print("⚠️ Telegram API returned", r.status_code, r.text)
        except Exception as e:
            print("⚠️ Telegram send exception:", e)
        time.sleep(1 + attempt*2)
    return False

# -------- Main flow --------
def main():
    print("=== Smart AI Scanner v4.2 running at", datetime.utcnow().isoformat(), "UTC ===")
    history = load_history()

    # 1. get CoinGecko markets (page 1)
    cg_page1 = fetch_coingecko_markets(per_page=250, page=1)
    if cg_page1.empty:
        print("⚠️ CoinGecko returned empty page1, trying fallback: CMC (if available)")
        if CMC_API_KEY:
            cmc_df = fetch_cmc_listings(limit=50)
            # map CMC to minimal DataFrame mimicking CG fields (basic)
            if not cmc_df.empty:
                # convert CMC structure to simpler
                try:
                    mapped = []
                    for _, r in cmc_df.iterrows():
                        mapped.append({
                            "id": r.get("slug") or "",
                            "symbol": r.get("symbol","").lower(),
                            "name": r.get("name",""),
                            "current_price": r.get("quote",{}).get("USD",{}).get("price",0),
                            "market_cap": r.get("quote",{}).get("USD",{}).get("market_cap",0),
                            "total_volume": r.get("quote",{}).get("USD",{}).get("volume_24h",0)
                        })
                    cg_df = pd.DataFrame(mapped)
                except Exception:
                    cg_df = pd.DataFrame()
            else:
                cg_df = pd.DataFrame()
        else:
            cg_df = pd.DataFrame()
    else:
        cg_df = cg_page1

    if cg_df.empty:
        print("❌ No market data available, aborting.")
        return

    # 2. dex & llama data
    dex_df = fetch_dexscreener_pairs()
    llama_df = fetch_defillama_protocols()

    # 3. compute technical placeholders (optional)
    cg_df = cg_df.copy()
    cg_df["RSI"] = 60
    cg_df["EMA20_gt_EMA50"] = True

    # 4. analyze and score
    picks = analyze_and_select(cg_df, dex_df, llama_df, top_k=TOP_K)
    if picks.empty:
        send_telegram("📉 Smart AI v4.2 — لا توجد فرص نادرة الآن.")
        return

    # 5. filter by history to avoid duplicate signals
    now = datetime.utcnow()
    to_send = []
    for _, p in picks.iterrows():
        sym = p["symbol"]
        last_ts = history.get(sym)
        if last_ts:
            try:
                last_dt = datetime.fromisoformat(last_ts)
                if now - last_dt < timedelta(hours=NO_REPEAT_HOURS):
                    print(f"ℹ️ skipping {sym} — sent {now - last_dt} ago")
                    continue
            except Exception:
                pass
        to_send.append(p)

    if not to_send:
        print("ℹ️ no new signals after history filter")
        return

    # 6. prepare message
    msg = f"🚀 Smart AI Rare Opportunities v4.2\n🕒 {now.strftime('%Y-%m-%d %H:%M UTC')}\n\n"
    for p in to_send:
        msg += (f"• {p['symbol']} — ${p['price']:.6f} | Score: {p['total_score']*100:.0f}\n"
                f"  MC: ${int(p['market_cap']):,} | Vol24: ${int(p['total_volume']):,}\n"
                f"  DexLiq: ${int(p['dex_liquidity']):,} | DexVol24: ${int(p['dex_volume_24h']):,}\n"
                f"  Narrative:{p['narrative_score']:.2f} | Social:{p['social_score']:.2f} | Tech:{p['tech_score']:.2f}\n\n")

    sent_ok = send_telegram(msg)
    if sent_ok:
        for p in to_send:
            history[p["symbol"]] = now.isoformat()
        save_history(history)

if __name__ == "__main__":
    main()
