import os
from flask import Flask, render_template, jsonify, request
import numpy as np
from scipy.stats import norm
import yfinance as yf
import datetime

app = Flask(__name__)

def prix_bs(s0, k, t, r, sigma, option_type="call"):
    if t <= 0:
        return max(s0 - k, 0) if option_type == "call" else max(k - s0, 0)
    if sigma <= 0:
        return max(s0 - k * np.exp(-r * t), 0)
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == "call":
        return float(s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2))
    else:
        return float(k * np.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1))

def historical_vol(ticker_symbol, window=252):
    data = yf.Ticker(ticker_symbol).history(period="1y")
    prices = data["Close"]
    returns = np.log(prices / prices.shift(1))
    vol = float(returns.std() * np.sqrt(window))
    return vol


def brownian_motion(T, N):
    dt = T / N
    dW = np.sqrt(dt) * np.random.randn(N)
    W  = np.cumsum(dW)
    return dW, W


def euler(s0, sigma, t, dW, r=0.05):
    N  = len(dW)
    dt = t / N
    s  = s0
    for i in range(N):
        s = s + r * s * dt + sigma * s * dW[i]
    return float(s)


def milstein(s0, sigma, t, dW, r=0.05):
    N  = len(dW)
    dt = t / N
    s  = s0
    for i in range(N):
        s = s + r*s*dt + sigma*s*dW[i] + 0.5*sigma**2*s*(dW[i]**2 - dt)
    return float(s)


def monte_carlo(s0, k, sigma, t, r=0.05, N=10000, option_type="call"):
    z   = np.random.normal(0, 1, N)
    sT  = s0 * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)
    if option_type == "call":
        payoffs = np.maximum(sT - k, 0)
    else:
        payoffs = np.maximum(k - sT, 0)
    return float(np.exp(-r * t) * np.mean(payoffs)), sT.tolist()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/ticker", methods=["GET"])
def get_ticker():
    ticker = request.args.get("ticker", "AAPL").upper()
    try:
        t = yf.Ticker(ticker)
        data = t.history(period="1y")
        if data.empty or len(data) < 10:
            return jsonify({"error": f"Données insuffisantes pour {ticker}"}), 400

        prices  = data["Close"]
        returns = np.log(prices / prices.shift(1)).dropna()
        vol     = float(returns.std() * np.sqrt(252))

        info    = t.fast_info
        price   = float(info.last_price) if hasattr(info, "last_price") else float(prices.iloc[-1])

        return jsonify({
            "ticker":    ticker,
            "price":     round(price, 4),
            "vol":       round(vol, 6),
            "vol_pct":   round(vol * 100, 2),
            "n_days":    len(returns),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/blackscholes", methods=["POST"])
def blackscholes():
    d     = request.json
    s0    = float(d["s0"])
    k     = float(d["k"])
    t     = float(d["t"])
    r     = float(d["r"])
    sigma = float(d["sigma"])

    call = prix_bs(s0, k, t, r, sigma, "call")
    put  = prix_bs(s0, k, t, r, sigma, "put")

    lo     = s0 * 0.4
    hi     = s0 * 1.8
    prices = [round(lo + i * (hi - lo) / 80, 2) for i in range(81)]
    call_payoff = [round(max(p - k, 0), 4) for p in prices]
    put_payoff  = [round(max(k - p, 0), 4) for p in prices]

    return jsonify({
        "call":         round(call, 4),
        "put":          round(put, 4),
        "payoff_x":     prices,
        "call_payoff":  call_payoff,
        "put_payoff":   put_payoff,
    })


@app.route("/api/montecarlo", methods=["POST"])
def montecarlo():
    d     = request.json
    s0    = float(d["s0"])
    k     = float(d["k"])
    t     = float(d["t"])
    r     = float(d["r"])
    sigma = float(d["sigma"])
    n     = int(d.get("n", 10000))

    mc_call, sT_list = monte_carlo(s0, k, sigma, t, r, n, "call")
    mc_put,  _       = monte_carlo(s0, k, sigma, t, r, n, "put")
    bs_call          = prix_bs(s0, k, t, r, sigma, "call")
    bs_put           = prix_bs(s0, k, t, r, sigma, "put")

    sT   = np.array(sT_list)
    bins = 45
    lo, hi = sT.min(), sT.max()
    bsize  = (hi - lo) / bins
    hist   = [0] * bins
    labels = [round(lo + i * bsize, 1) for i in range(bins)]
    for v in sT:
        b = min(int((v - lo) / bsize), bins - 1)
        hist[b] += 1

    return jsonify({
        "mc_call":   round(mc_call, 4),
        "mc_put":    round(mc_put, 4),
        "bs_call":   round(bs_call, 4),
        "bs_put":    round(bs_put, 4),
        "err_call":  round(abs(mc_call - bs_call), 5),
        "err_put":   round(abs(mc_put  - bs_put),  5),
        "hist_x":    labels,
        "hist_y":    hist,
    })


@app.route("/api/convergence", methods=["POST"])
def convergence():
    d     = request.json
    s0    = float(d["s0"])
    k     = float(d["k"])
    t     = float(d["t"])
    sigma = float(d["sigma"])
    r     = 0.05
    Ns = [10, 100, 500, 1000, 5000]
    M     = 1000   # schémas SDE
    Mmc   = 3000  # Monte Carlo GBM pur

    bs_call = prix_bs(s0, k, t, r, sigma, "call")
    bs_put  = prix_bs(s0, k, t, r, sigma, "put")

    rows = []
    for N in Ns:
        dt = t / N

        sum_ec = sum_ep = sum_mc_s = sum_mp = 0.0
        for _ in range(M):
            dW = np.sqrt(dt) * np.random.randn(N)
            se = euler(s0, sigma, t, dW, r)
            sm = milstein(s0, sigma, t, dW, r)
            sum_ec   += max(se - k, 0)
            sum_ep   += max(k - se, 0)
            sum_mc_s += max(sm - k, 0)
            sum_mp   += max(k - sm, 0)

        euler_call  = np.exp(-r*t) * sum_ec   / M
        euler_put   = np.exp(-r*t) * sum_ep   / M
        milst_call  = np.exp(-r*t) * sum_mc_s / M
        milst_put   = np.exp(-r*t) * sum_mp   / M

        z   = np.random.normal(0, 1, Mmc)
        sT  = s0 * np.exp((r - 0.5*sigma**2)*t + sigma*np.sqrt(t)*z)
        mc_call_v = float(np.exp(-r*t) * np.mean(np.maximum(sT - k, 0)))
        mc_put_v  = float(np.exp(-r*t) * np.mean(np.maximum(k - sT, 0)))

        rows.append({
            "N":           N,
            "bs_call":     round(bs_call, 4),
            "bs_put":      round(bs_put, 4),
            "mc_call":     round(mc_call_v, 4),
            "mc_put":      round(mc_put_v, 4),
            "euler_call":  round(euler_call, 4),
            "euler_put":   round(euler_put, 4),
            "milst_call":  round(milst_call, 4),
            "milst_put":   round(milst_put, 4),
            "err_mc":      round(abs(mc_call_v  - bs_call), 5),
            "err_euler":   round(abs(euler_call - bs_call), 5),
            "err_milst":   round(abs(milst_call - bs_call), 5),
            "err_mc_put":  round(abs(mc_put_v   - bs_put),  5),
            "err_euler_put": round(abs(euler_put - bs_put), 5),
            "err_milst_put": round(abs(milst_put - bs_put), 5),
        })

    return jsonify({"rows": rows, "Ns": Ns})

@app.route("/api/convergence_forte", methods=["POST"])
def convergence_forte():
    d     = request.json
    s0    = float(d["s0"])
    t     = float(d["t"])
    sigma = float(d["sigma"])
    r     = 0.05
    Ns = [10, 100, 500, 1000, 5000]
    M     = 10000  # trajectoires pour moyenner l'erreur forte

    rows = []
    for N in Ns:
        dt = t / N
        err_e = 0.0
        err_m = 0.0

        for _ in range(M):
            dW = np.sqrt(dt) * np.random.randn(N)
            W_T = np.sum(dW)

            S_exact = s0 * np.exp((r - 0.5*sigma**2)*t + sigma*W_T)

            # Euler
            se = s0
            for dw in dW:
                se = se + r*se*dt + sigma*se*dw
            err_e += abs(se - S_exact)

            # Milstein
            sm = s0
            for dw in dW:
                sm = sm + r*sm*dt + sigma*sm*dw + 0.5*sigma**2*sm*(dw**2 - dt)
            err_m += abs(sm - S_exact)

        rows.append({
            "N":          N,
            "err_euler":  round(err_e / M, 6),
            "err_milst":  round(err_m / M, 6),
        })

    return jsonify({"rows": rows, "Ns": Ns})

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import threading

_ml_model      = None
_ml_rmse       = None
_ml_train_done = False
_ml_lock       = threading.Lock()


def generate_dataset(n_samples=3000):
    data = []
    for _ in range(n_samples):
        S0    = np.random.uniform(50, 500)   
        K     = S0 * np.random.uniform(0.7, 1.3)  
        T     = np.random.uniform(0.1, 2)
        r     = 0.05
        sigma = np.random.uniform(0.1, 0.5)
        z     = np.random.normal(0, 1, 5000)
        sT    = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
        price_call = float(np.exp(-r*T) * np.mean(np.maximum(sT - K, 0)))
        price_put  = float(np.exp(-r*T) * np.mean(np.maximum(K - sT, 0)))
        data.append([K, T, S0, r, sigma, price_call, price_put])
    return pd.DataFrame(data, columns=["K","T","S0","r","sigma","price_call","price_put"])


@app.route("/api/ml/train", methods=["POST"])
def ml_train():
    global _ml_model, _ml_rmse, _ml_train_done

    d         = request.json or {}
    n_samples = int(d.get("n_samples", 3000))

    try:
        df = generate_dataset(n_samples)
        X  = df[["K","T","S0","r","sigma"]]

        X_train, X_test, y_call_train, y_call_test, y_put_train, y_put_test = train_test_split(
            X, df["price_call"], df["price_put"], test_size=0.2, random_state=42
        )

        model_call = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
        )
        model_call.fit(X_train, y_call_train)
        rmse_call = float(np.sqrt(mean_squared_error(y_call_test, model_call.predict(X_test))))

        model_put = GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
        )
        model_put.fit(X_train, y_put_train)
        rmse_put = float(np.sqrt(mean_squared_error(y_put_test, model_put.predict(X_test))))

        with _ml_lock:
            _ml_model      = {"call": model_call, "put": model_put}
            _ml_rmse       = {"call": rmse_call,  "put": rmse_put}
            _ml_train_done = True

        return jsonify({
            "rmse_call":   round(rmse_call, 6),
            "rmse_put":    round(rmse_put,  6),
            "n_samples":   n_samples,
            "features":    ["K","T","S0","r","sigma"],
            "importances": [round(float(v), 6) for v in model_call.feature_importances_],
            "n_estimators": model_call.n_estimators,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ml/predict", methods=["POST"])
def ml_predict():
    global _ml_model, _ml_train_done

    if not _ml_train_done or _ml_model is None:
        return jsonify({"error": "Modèle non entraîné. Lancez d'abord l'entraînement."}), 400

    d     = request.json
    s0    = float(d["s0"])
    k     = float(d["k"])
    t     = float(d["t"])
    r     = float(d.get("r", 0.05))
    sigma = float(d["sigma"])

    sample        = pd.DataFrame([[k, t, s0, r, sigma]], columns=["K","T","S0","r","sigma"])
    ml_call       = float(_ml_model["call"].predict(sample)[0])
    ml_put        = float(_ml_model["put"].predict(sample)[0])
    bs_call       = prix_bs(s0, k, t, r, sigma, "call")
    bs_put        = prix_bs(s0, k, t, r, sigma, "put")

    return jsonify({
        "ml_call":       round(ml_call, 4),
        "ml_put":        round(ml_put,  4),
        "bs_call":       round(bs_call, 4),
        "bs_put":        round(bs_put,  4),
        "err_abs_call":  round(abs(ml_call - bs_call), 5),
        "err_pct_call":  round(abs(ml_call - bs_call) / max(bs_call, 0.0001) * 100, 3),
        "err_abs_put":   round(abs(ml_put  - bs_put),  5),
        "err_pct_put":   round(abs(ml_put  - bs_put)  / max(bs_put,  0.0001) * 100, 3),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port)