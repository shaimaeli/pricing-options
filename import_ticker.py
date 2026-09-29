import yfinance as yf
import pandas as pd
import datetime
import numpy as np


# =========================
# VOLATILITÉ HISTORIQUE
# =========================
def historical_vol(ticker_symbol, window=252):

    data = yf.Ticker(ticker_symbol).history(period="1y")

    prices = data["Close"]

    returns = np.log(prices / prices.shift(1))

    vol = returns.std() * np.sqrt(window)

    return vol


# =========================
# OPTIONS + VOL HISTORIQUE
# =========================
def recuperer_options_smile(ticker_symbol, duree_jours=365):

    ticker = yf.Ticker(ticker_symbol)

    today = datetime.date.today()
    one_year_later = today + datetime.timedelta(days=duree_jours)

    dates_exp = ticker.options

    dates_exp = [
        d for d in dates_exp
        if datetime.datetime.strptime(d, "%Y-%m-%d").date() <= one_year_later
    ]

    all_options = []

    # =========================
    # VOLATILITÉ HISTORIQUE (UNE SEULE VALEUR)
    # =========================
    vol_hist = historical_vol(ticker_symbol)

    for expiry in dates_exp:

        opt = ticker.option_chain(expiry)
        expiry_date = datetime.datetime.strptime(expiry, "%Y-%m-%d").date()

        # ================= CALLS =================
        calls = opt.calls.copy()
        calls['type'] = 'call'
        calls['T'] = (expiry_date - today).days / 365
        calls['expiration'] = expiry_date

        # ajouter vol historique
        calls['vol_hist'] = vol_hist

        all_options.append(
            calls[['type', 'strike', 'lastPrice', 'bid', 'ask', 'T', 'expiration', 'vol_hist']]
        )

        # ================= PUTS =================
        puts = opt.puts.copy()
        puts['type'] = 'put'
        puts['T'] = (expiry_date - today).days / 365
        puts['expiration'] = expiry_date

        puts['vol_hist'] = vol_hist

        all_options.append(
            puts[['type', 'strike', 'lastPrice', 'bid', 'ask', 'T', 'expiration', 'vol_hist']]
        )

    df_options = pd.concat(all_options, ignore_index=True)

    df_options = df_options.sort_values(['expiration', 'strike']).reset_index(drop=True)

    return df_options