from flask import Flask, render_template, jsonify, request
import numpy as np
import yfinance as yf
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

app = Flask(__name__)

NB_PAS = [10, 50, 100, 500, 1000]   # valeurs de N testées pour la convergence
modeles = {}                        # contiendra les modèles ML "call" et "put"


# =========================================================
# PRICING
# =========================================================

def black_scholes(s0, k, t, r, sigma):
    """Prix exact d'un call et d'un put européens."""
    d1 = (np.log(s0 / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)

    call = s0 * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    put = k * np.exp(-r * t) * norm.cdf(-d2) - s0 * norm.cdf(-d1)
    return call, put


def monte_carlo(s0, k, t, r, sigma, n_sim=10000):
    """Prix par Monte Carlo : moyenne actualisée des payoffs simulés."""
    z = np.random.randn(n_sim)
    st = s0 * np.exp((r - 0.5 * sigma**2) * t + sigma * np.sqrt(t) * z)

    actualisation = np.exp(-r * t)
    call = actualisation * np.mean(np.maximum(st - k, 0))
    put = actualisation * np.mean(np.maximum(k - st, 0))
    return call, put, st


def simuler_schemas(s0, t, r, sigma, n_steps, n_paths):
    """Simule Euler, Milstein et la solution exacte avec le même mouvement brownien."""
    dt = t / n_steps
    dw = np.sqrt(dt) * np.random.randn(n_paths, n_steps)

    s_euler = np.full(n_paths, s0)
    s_milstein = np.full(n_paths, s0)

    for i in range(n_steps):
        s_euler = s_euler + r * s_euler * dt + sigma * s_euler * dw[:, i]
        s_milstein = (s_milstein + r * s_milstein * dt
                      + sigma * s_milstein * dw[:, i]
                      + 0.5 * sigma**2 * s_milstein * (dw[:, i]**2 - dt))

    s_exact = s0 * np.exp((r - 0.5 * sigma**2) * t + sigma * dw.sum(axis=1))
    return s_euler, s_milstein, s_exact


def calculer_convergence(s0, k, t, r, sigma, n_paths=2000):
    """Erreur faible (sur le prix du call) et erreur forte (sur les trajectoires)."""
    prix_exact, _ = black_scholes(s0, k, t, r, sigma)
    actualisation = np.exp(-r * t)
    resultats = []

    for n_steps in NB_PAS:
        s_euler, s_milstein, s_exact = simuler_schemas(s0, t, r, sigma, n_steps, n_paths)

        prix_euler = actualisation * np.mean(np.maximum(s_euler - k, 0))
        prix_milstein = actualisation * np.mean(np.maximum(s_milstein - k, 0))

        resultats.append({
            "N": n_steps,
            "faible_euler": float(abs(prix_euler - prix_exact)),
            "faible_milstein": float(abs(prix_milstein - prix_exact)),
            "forte_euler": float(np.mean(np.abs(s_euler - s_exact))),
            "forte_milstein": float(np.mean(np.abs(s_milstein - s_exact))),
        })
    return resultats


# =========================================================
# MACHINE LEARNING
# =========================================================

def generer_donnees(n=3000):
    """Tire des paramètres au hasard et calcule les prix Black-Scholes correspondants."""
    s0 = np.random.uniform(50, 500, n)
    k = s0 * np.random.uniform(0.7, 1.3, n)
    t = np.random.uniform(0.1, 2, n)
    r = np.random.uniform(0, 0.1, n)
    sigma = np.random.uniform(0.1, 0.5, n)   # le modèle ne connaît que cette plage de volatilité

    call, put = black_scholes(s0, k, t, r, sigma)
    X = np.column_stack([s0, k, t, r, sigma])
    return X, call, put


def entrainer_modeles(n=3000):
    """Entraîne un modèle pour le call et un pour le put. Renvoie les RMSE sur le jeu de test."""
    X, call, put = generer_donnees(n)
    X_train, X_test, call_train, call_test, put_train, put_test = train_test_split(
        X, call, put, test_size=0.2, random_state=42)

    rmse = {}
    for nom, y_train, y_test in [("call", call_train, call_test), ("put", put_train, put_test)]:
        modele = GradientBoostingRegressor(n_estimators=200, max_depth=3)
        modele.fit(X_train, y_train)
        modeles[nom] = modele
        rmse[nom] = float(np.sqrt(np.mean((modele.predict(X_test) - y_test) ** 2)))
    return rmse


# =========================================================
# ROUTES FLASK
# =========================================================

def lire_parametres():
    """Récupère S0, K, T, r et sigma envoyés par le navigateur."""
    d = request.json
    return d["s0"], d["k"], d["t"], d["r"], d["sigma"]


@app.route("/")
def accueil():
    return render_template("index.html")


@app.route("/api/ticker")
def api_ticker():
    """Dernier prix et volatilité historique annualisée d'un ticker."""
    symbole = request.args.get("ticker", "AAPL").upper()
    donnees = yf.Ticker(symbole).history(period="1y")

    if donnees.empty or len(donnees) < 10:
        return jsonify({"error": "Ticker introuvable ou données insuffisantes"}), 400

    cours = donnees["Close"]
    rendements = np.log(cours / cours.shift(1)).dropna()
    volatilite = rendements.std() * np.sqrt(252)

    return jsonify({"prix": float(cours.iloc[-1]), "volatilite": float(volatilite)})


@app.route("/api/prix", methods=["POST"])
def api_prix():
    """Prix Black-Scholes et Monte Carlo, plus l'histogramme des S_T simulés."""
    s0, k, t, r, sigma = lire_parametres()

    bs_call, bs_put = black_scholes(s0, k, t, r, sigma)
    mc_call, mc_put, st = monte_carlo(s0, k, t, r, sigma)
    effectifs, bornes = np.histogram(st, bins=40)

    return jsonify({
        "bs_call": float(bs_call), "bs_put": float(bs_put),
        "mc_call": float(mc_call), "mc_put": float(mc_put),
        "hist_x": [round(float(b), 1) for b in bornes[:-1]],
        "hist_y": effectifs.tolist(),
    })


@app.route("/api/convergence", methods=["POST"])
def api_convergence():
    s0, k, t, r, sigma = lire_parametres()
    return jsonify(calculer_convergence(s0, k, t, r, sigma))


@app.route("/api/ml/entrainer", methods=["POST"])
def api_ml_entrainer():
    return jsonify(entrainer_modeles())


@app.route("/api/ml/predire", methods=["POST"])
def api_ml_predire():
    if not modeles:
        return jsonify({"error": "Entraînez d'abord le modèle"}), 400

    s0, k, t, r, sigma = lire_parametres()
    x = [[s0, k, t, r, sigma]]
    bs_call, bs_put = black_scholes(s0, k, t, r, sigma)

    return jsonify({
        "ml_call": float(modeles["call"].predict(x)[0]),
        "ml_put": float(modeles["put"].predict(x)[0]),
        "bs_call": float(bs_call), "bs_put": float(bs_put),
    })


if __name__ == "__main__":
    app.run(debug=True)