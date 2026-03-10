"""
Options Analytics Dashboard
============================
Production-ready options pricing and risk analysis tool.
Built with Black-Scholes (1973) and Cox-Ross-Rubinstein (1979).

Run: streamlit run options_dashboard.py
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Options Analytics Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Sora:wght@300;400;600;700&display=swap');

  /* Root vars */
  :root {
    --bg-primary: #0E1117;
    --bg-sidebar: #1B2838;
    --bg-card: #161B27;
    --bg-card-hover: #1E2535;
    --border: #2A3347;
    --accent-blue: #3B82F6;
    --accent-green: #10B981;
    --accent-red: #EF4444;
    --accent-gold: #F59E0B;
    --text-primary: #E8EDF5;
    --text-secondary: #8899AA;
    --text-muted: #4A5568;
    --font-mono: 'JetBrains Mono', monospace;
    --font-sans: 'Sora', sans-serif;
  }

  /* Base */
  html, body, [class*="css"] {
    font-family: var(--font-sans);
    background-color: var(--bg-primary);
    color: var(--text-primary);
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stTextInput label,
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] .stSelectbox label {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-family: var(--font-mono) !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  section[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
    padding-top: 4px;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
    transition: border-color 0.2s ease;
  }
  [data-testid="metric-container"]:hover {
    border-color: var(--accent-blue) !important;
  }
  [data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: var(--bg-card) !important;
    border-radius: 10px !important;
    padding: 4px !important;
    border: 1px solid var(--border) !important;
    gap: 2px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: var(--text-secondary) !important;
    border-radius: 7px !important;
    padding: 8px 16px !important;
    transition: all 0.2s ease;
  }
  .stTabs [aria-selected="true"] {
    background: var(--accent-blue) !important;
    color: white !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
  }

  /* Dataframe */
  .stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
  }

  /* Buttons */
  .stButton > button {
    background: var(--accent-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 7px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    letter-spacing: 0.04em;
    transition: opacity 0.2s ease, transform 0.1s ease;
  }
  .stButton > button:hover {
    opacity: 0.88 !important;
    transform: translateY(-1px);
  }

  /* Info / warning / success boxes */
  .stAlert {
    border-radius: 8px !important;
    font-family: var(--font-mono) !important;
    font-size: 0.78rem !important;
  }

  /* Section headers */
  h1, h2, h3 {
    font-family: var(--font-sans) !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
  }

  /* Live badge */
  .live-badge {
    display: inline-block;
    background: rgba(16,185,129,0.15);
    border: 1px solid rgba(16,185,129,0.4);
    color: #10B981;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-top: 4px;
    letter-spacing: 0.04em;
  }

  /* Moneyness badge */
  .badge-itm { background:rgba(16,185,129,0.15); border:1px solid rgba(16,185,129,0.4); color:#10B981; }
  .badge-otm { background:rgba(239,68,68,0.15);  border:1px solid rgba(239,68,68,0.4);  color:#EF4444; }
  .badge-atm { background:rgba(59,130,246,0.15); border:1px solid rgba(59,130,246,0.4); color:#3B82F6; }
  .moneyness-badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    padding: 12px 28px;
    border-radius: 10px;
    letter-spacing: 0.1em;
    margin-top: 8px;
  }

  /* Greek row highlight */
  .greek-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
  }
  .greek-card:hover { border-color: var(--accent-blue); }

  /* Sidebar footer */
  .sidebar-footer {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #4A5568;
    border-top: 1px solid #2A3347;
    padding-top: 12px;
    line-height: 1.7;
    margin-top: 20px;
  }

  /* Divider */
  hr { border-color: var(--border) !important; margin: 20px 0 !important; }

  /* Number inputs */
  input[type="number"] {
    font-family: 'JetBrains Mono', monospace !important;
    background: #0E1117 !important;
  }

  /* Plotly chart bg */
  .js-plotly-plot .plotly .main-svg {
    border-radius: 10px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PRICING ENGINE
# ─────────────────────────────────────────────

def black_scholes(S, K, T, r, sigma, q=0.0, option_type="Call") -> dict:
    """
    Black-Scholes option pricing formula.

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiry in years
        r: Risk-free rate (decimal)
        sigma: Volatility (decimal)
        q: Continuous dividend yield (decimal)
        option_type: "Call" or "Put"

    Returns:
        dict with keys: price, d1, d2, intrinsic, time_value

    Formula:
        d1 = [ln(S/K) + (r - q + sigma²/2) * T] / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        Call = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)
        Put  = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)
    """
    if T <= 0:
        intrinsic = max(S - K, 0) if option_type == "Call" else max(K - S, 0)
        return {"price": intrinsic, "d1": 0.0, "d2": 0.0,
                "intrinsic": intrinsic, "time_value": 0.0}
    if sigma <= 0:
        intrinsic = max(S - K, 0) if option_type == "Call" else max(K - S, 0)
        return {"price": intrinsic, "d1": 0.0, "d2": 0.0,
                "intrinsic": intrinsic, "time_value": 0.0}

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "Call":
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        intrinsic = max(S - K, 0.0)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        intrinsic = max(K - S, 0.0)

    time_value = max(price - intrinsic, 0.0)
    return {"price": float(price), "d1": float(d1), "d2": float(d2),
            "intrinsic": float(intrinsic), "time_value": float(time_value)}


def greeks(S, K, T, r, sigma, q=0.0, option_type="Call") -> dict:
    """
    Compute all option Greeks analytically from Black-Scholes.

    Returns dict with keys:
        delta:  dP/dS — hedge ratio, [0,1] for calls, [-1,0] for puts
        gamma:  d²P/dS² — rate of delta change, always positive
        vega:   dP/dσ — sensitivity to volatility (per 1% change in vol)
        theta:  dP/dT — time decay per calendar day (negative for long options)
        rho:    dP/dr — sensitivity to interest rate (per 1% change in r)

    Key formulas:
        delta_call = e^(-qT) * N(d1)
        delta_put  = e^(-qT) * (N(d1) - 1)
        gamma      = e^(-qT) * N'(d1) / (S * sigma * sqrt(T))
        vega       = S * e^(-qT) * N'(d1) * sqrt(T) / 100  [per 1%]
        theta_call = [-S*N'(d1)*sigma*e^(-qT) / (2*sqrt(T))
                      - r*K*e^(-rT)*N(d2)
                      + q*S*e^(-qT)*N(d1)] / 365  [per day]
        rho_call   = K*T*e^(-rT)*N(d2) / 100  [per 1%]
    """
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    bs = black_scholes(S, K, T, r, sigma, q, option_type)
    d1, d2 = bs["d1"], bs["d2"]
    nd1_pdf = norm.pdf(d1)

    # Delta
    if option_type == "Call":
        delta = np.exp(-q * T) * norm.cdf(d1)
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)

    # Gamma (same for call and put)
    gamma = np.exp(-q * T) * nd1_pdf / (S * sigma * np.sqrt(T))

    # Vega per 1% vol change
    vega = S * np.exp(-q * T) * nd1_pdf * np.sqrt(T) / 100

    # Theta per calendar day
    if option_type == "Call":
        theta = (
            -S * nd1_pdf * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            - r * K * np.exp(-r * T) * norm.cdf(d2)
            + q * S * np.exp(-q * T) * norm.cdf(d1)
        ) / 365
    else:
        theta = (
            -S * nd1_pdf * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
            + r * K * np.exp(-r * T) * norm.cdf(-d2)
            - q * S * np.exp(-q * T) * norm.cdf(-d1)
        ) / 365

    # Rho per 1% rate change
    if option_type == "Call":
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "vega":  float(vega),
        "theta": float(theta),
        "rho":   float(rho),
    }


def binomial_tree(S, K, T, r, sigma, q=0.0, steps=100, option_type="Call",
                  american=False) -> dict:
    """
    Cox-Ross-Rubinstein binomial tree pricing.

    Args:
        american: If True, check early exercise at each node (for puts)

    Returns:
        dict with keys: price, tree_prices (first 10 steps for visualisation)

    Algorithm:
        dt = T / steps
        u  = exp(sigma * sqrt(dt))       # up factor
        d  = 1 / u                       # down factor (ensures recombining tree)
        p  = (exp((r-q)*dt) - d) / (u-d) # risk-neutral probability

        Terminal payoffs: max(S * u^j * d^(n-j) - K, 0) for j=0..n
        Backward induction: V[i] = e^(-r*dt) * (p*V_up + (1-p)*V_down)
        American: at each node take max(V, intrinsic)
    """
    steps = min(steps, 1000)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    discount = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)
    p = np.clip(p, 0.0, 1.0)

    # Terminal asset prices
    j_arr = np.arange(steps + 1)
    S_T = S * (u ** j_arr) * (d ** (steps - j_arr))

    if option_type == "Call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # Backward induction
    for i in range(steps - 1, -1, -1):
        V = discount * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
        if american:
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            intrinsic = np.maximum(S_i - K, 0.0) if option_type == "Call" else np.maximum(K - S_i, 0.0)
            V = np.maximum(V, intrinsic)

    price = float(V[0])

    # Build small tree for visualisation (first vis_steps steps)
    vis_steps = min(10, steps)
    tree_S = {}
    tree_V_nodes = {}

    for step in range(vis_steps + 1):
        for node in range(step + 1):
            s_val = S * (u ** node) * (d ** (step - node))
            tree_S[(step, node)] = s_val

    # Rerun small tree for option values
    small_steps = vis_steps
    small_dt = T / steps  # keep same dt for accuracy
    j_small = np.arange(small_steps + 1)
    S_T_small = S * (u ** j_small) * (d ** (small_steps - j_small))
    if option_type == "Call":
        V_small = np.maximum(S_T_small - K, 0.0)
    else:
        V_small = np.maximum(K - S_T_small, 0.0)

    node_values = {small_steps: V_small.copy()}
    V_tmp = V_small.copy()
    for i in range(small_steps - 1, -1, -1):
        V_tmp = discount * (p * V_tmp[1:i + 2] + (1 - p) * V_tmp[0:i + 1])
        if american:
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            intr = np.maximum(S_i - K, 0.0) if option_type == "Call" else np.maximum(K - S_i, 0.0)
            V_tmp = np.maximum(V_tmp, intr)
        node_values[i] = V_tmp.copy()

    for step in range(vis_steps + 1):
        for node in range(step + 1):
            tree_V_nodes[(step, node)] = float(node_values[step][node])

    return {
        "price": price,
        "tree_S": tree_S,
        "tree_V": tree_V_nodes,
        "vis_steps": vis_steps,
        "u": u, "d": d, "p": p,
    }


def implied_volatility(market_price, S, K, T, r, q=0.0, option_type="Call",
                       tol=1e-6, max_iter=100) -> dict:
    """
    Newton-Raphson solver for implied volatility.

    Finds σ* such that BS(σ*) = market_price.

    Algorithm:
        Initial guess: sigma_0 = sqrt(2 * abs(ln(S/K) + r*T) / T)
        Iteration: sigma_{n+1} = sigma_n - (BS(sigma_n) - market_price) / vega(sigma_n)

    Returns:
        float: implied vol as decimal, or np.nan if no convergence

    Note:
        Vega is used as the Jacobian (derivative of price w.r.t. sigma).
        If vega < 1e-10 at any step (deep ITM/OTM), abort and return nan.
    """
    if T <= 0 or market_price <= 0:
        return {"iv": np.nan, "iterations": 0, "converged": False}

    # Initial guess
    moneyness = np.log(S / K) + r * T
    sigma = np.sqrt(2 * abs(moneyness) / T) if abs(moneyness) > 1e-10 else 0.3
    sigma = max(sigma, 0.01)

    for i in range(max_iter):
        bs_result = black_scholes(S, K, T, r, sigma, q, option_type)
        g_result  = greeks(S, K, T, r, sigma, q, option_type)
        price_diff = bs_result["price"] - market_price
        vega_val   = g_result["vega"] * 100  # convert back from per-1% to per-unit

        if abs(vega_val) < 1e-10:
            return {"iv": np.nan, "iterations": i + 1, "converged": False}

        sigma_new = sigma - price_diff / vega_val
        sigma_new = max(sigma_new, 1e-6)

        if abs(sigma_new - sigma) < tol:
            return {"iv": float(sigma_new), "iterations": i + 1, "converged": True}
        sigma = sigma_new

    # Fallback: try Brent's method
    try:
        def objective(s):
            return black_scholes(S, K, T, r, s, q, option_type)["price"] - market_price
        iv_brent = brentq(objective, 1e-6, 10.0, xtol=tol)
        return {"iv": float(iv_brent), "iterations": max_iter, "converged": True}
    except Exception:
        return {"iv": np.nan, "iterations": max_iter, "converged": False}


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="#0E1117",
    plot_bgcolor="#0E1117",
    font=dict(family="JetBrains Mono, monospace", color="#8899AA", size=11),
    margin=dict(l=50, r=30, t=50, b=50),
    xaxis=dict(gridcolor="#2A3347", zerolinecolor="#2A3347", linecolor="#2A3347"),
    yaxis=dict(gridcolor="#2A3347", zerolinecolor="#2A3347", linecolor="#2A3347"),
    legend=dict(
        bgcolor="rgba(22,27,39,0.85)",
        bordercolor="#2A3347",
        borderwidth=1,
        font=dict(size=10),
    ),
)


def apply_layout(fig, title="", height=420):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(
        size=13, color="#E8EDF5", family="Sora, sans-serif"), x=0.03), height=height)
    return fig


def fetch_spot_price(ticker_str):
    try:
        import yfinance as yf
        t = yf.Ticker(ticker_str.strip().upper())
        price = t.fast_info["lastPrice"]
        if price and price > 0:
            return float(price), True
    except Exception:
        pass
    return None, False


def moneyness_label(S, K, option_type):
    pct = (S / K - 1) * 100
    if option_type == "Call":
        if abs(pct) < 1.0:
            return "ATM", "badge-atm", f"{pct:+.2f}%"
        elif pct > 0:
            return "ITM", "badge-itm", f"{pct:+.2f}%"
        else:
            return "OTM", "badge-otm", f"{pct:+.2f}%"
    else:
        if abs(pct) < 1.0:
            return "ATM", "badge-atm", f"{pct:+.2f}%"
        elif pct < 0:
            return "ITM", "badge-itm", f"{pct:+.2f}%"
        else:
            return "OTM", "badge-otm", f"{pct:+.2f}%"


# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
if "spot_price" not in st.session_state:
    st.session_state.spot_price = 195.0
if "live_price" not in st.session_state:
    st.session_state.live_price = None
if "live_ok" not in st.session_state:
    st.session_state.live_ok = False
if "iv_result" not in st.session_state:
    st.session_state.iv_result = None
if "apply_iv" not in st.session_state:
    st.session_state.apply_iv = False


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 16px 0;'>
      <span style='font-family: Sora, sans-serif; font-size: 1.1rem; font-weight: 700;
                   color: #E8EDF5; letter-spacing:0.04em;'>OPTIONS ANALYTICS</span><br>
      <span style='font-family: JetBrains Mono, monospace; font-size: 0.68rem;
                   color: #4A5568; letter-spacing: 0.08em;'>BLACK-SCHOLES  ·  CRR BINOMIAL</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Ticker ──
    ticker = st.text_input("Ticker Symbol", value="AAPL", key="ticker_input")

    if st.button("🔄  Fetch Live Price", key="fetch_btn"):
        price, ok = fetch_spot_price(ticker)
        if ok:
            st.session_state.live_price = price
            st.session_state.live_ok = True
            st.session_state.spot_price = price
        else:
            st.session_state.live_ok = False
            st.session_state.live_price = None
            st.warning("⚠ Could not fetch — using manual input")

    if st.session_state.live_ok and st.session_state.live_price:
        st.markdown(
            f'<span class="live-badge">● LIVE  ${st.session_state.live_price:,.2f}</span>',
            unsafe