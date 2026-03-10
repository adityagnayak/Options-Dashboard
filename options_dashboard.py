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
            unsafe_allow_html=True,
        )
    st.markdown("")

    # ── Spot Price ──
    S = st.number_input(
        "Spot Price (S)",
        min_value=0.01,
        step=1.0,
        value=float(st.session_state.spot_price),
        format="%.2f",
        key="spot_input",
    )

    # ── Strike ──
    K = st.number_input(
        "Strike Price (K)",
        min_value=0.01,
        step=1.0,
        value=float(S),
        format="%.2f",
        key="strike_input",
    )

    # ── Days to Expiry ──
    days = st.slider("Days to Expiry", min_value=1, max_value=730, value=90,
                     key="days_slider")
    T_years = days / 365.0

    # ── Volatility ──
    # Apply IV to slider: must write to session_state key directly —
    # Streamlit ignores value= on keyed widgets after first render.
    if st.session_state.apply_iv and st.session_state.iv_result:
        iv_val = st.session_state.iv_result.get("iv", np.nan)
        if not np.isnan(iv_val):
            st.session_state["vol_slider"] = max(1, min(200, int(iv_val * 100)))
        st.session_state.apply_iv = False

    vol_pct = st.slider("Volatility (%)", min_value=1, max_value=200,
                        value=25, key="vol_slider")
    sigma = vol_pct / 100.0

    # ── Risk-Free Rate ──
    r_pct = st.number_input("Risk-Free Rate (%)", min_value=0.0, max_value=20.0,
                            step=0.1, value=5.0, format="%.1f", key="rate_input")
    r = r_pct / 100.0

    # ── Dividend Yield ──
    q_pct = st.number_input("Dividend Yield (%)", min_value=0.0, max_value=20.0,
                            step=0.1, value=0.0, format="%.1f", key="div_input")
    q = q_pct / 100.0

    st.markdown("---")

    # ── Option type / model ──
    option_type = st.radio("Option Type", options=["Call", "Put"],
                           horizontal=True, key="opt_type")
    model = st.selectbox("Pricing Model",
                         options=["Black-Scholes", "Binomial Tree", "Both"],
                         key="model_select")

    show_binomial_steps = model in ["Binomial Tree", "Both"]
    if show_binomial_steps:
        binom_steps = st.slider("Binomial Steps", min_value=10, max_value=500,
                                value=100, key="binom_steps")
    else:
        binom_steps = 100

    american = st.checkbox("American Style (Early Exercise)", value=False,
                           key="american_check")

    # ── Sidebar footer ──
    st.markdown("""
    <div class="sidebar-footer">
      Built with Black-Scholes (1973)<br>
      and Cox-Ross-Rubinstein (1979)<br><br>
      Model assumes log-normal returns,<br>
      continuous trading, no dividends<br>
      (unless specified), constant vol.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# VALIDATION GUARDS
# ─────────────────────────────────────────────
error_state = None

if S <= 0:
    error_state = "❌ Spot price cannot be zero or negative."
elif T_years <= 0:
    error_state = f"ℹ Option has expired. Intrinsic value: ${max(S-K,0) if option_type=='Call' else max(K-S,0):.4f}"
elif sigma <= 0:
    error_state = f"ℹ Zero volatility — option worth intrinsic value only: ${max(S-K,0) if option_type=='Call' else max(K-S,0):.4f}"

if error_state:
    st.warning(error_state)
    st.stop()


# ─────────────────────────────────────────────
# COMPUTE PRICES & GREEKS
# ─────────────────────────────────────────────
bs_result  = black_scholes(S, K, T_years, r, sigma, q, option_type)
grk_result = greeks(S, K, T_years, r, sigma, q, option_type)

bt_result = None
if model in ["Binomial Tree", "Both"]:
    bt_result = binomial_tree(S, K, T_years, r, sigma, q, binom_steps, option_type, american)

bs_price = bs_result["price"]
bt_price = bt_result["price"] if bt_result else None


# ─────────────────────────────────────────────
# MAIN PANEL HEADER
# ─────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"""
    <div style='margin-bottom:6px;'>
      <span style='font-family:Sora,sans-serif; font-size:1.6rem; font-weight:700;
                   color:#E8EDF5;'>{ticker.upper()}</span>
      <span style='font-family:JetBrains Mono,monospace; font-size:0.82rem;
                   color:#4A5568; margin-left:12px;'>
        S={S:.2f}  ·  K={K:.2f}  ·  T={days}d  ·  σ={vol_pct}%  ·  r={r_pct:.1f}%
      </span>
    </div>
    """, unsafe_allow_html=True)
with col_h2:
    mnm_label, mnm_cls, mnm_pct = moneyness_label(S, K, option_type)
    st.markdown(
        f'<div style="text-align:right; margin-top:4px;">'
        f'<span class="moneyness-badge {mnm_cls}">{mnm_label} {mnm_pct}</span></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Pricing Summary",
    "🔢 Greeks",
    "📈 Payoff Diagram",
    "🌡️ Heatmap",
    "🌳 Binomial Tree",
])


# ═══════════════════════════════════════════════
# TAB 1 — PRICING SUMMARY
# ═══════════════════════════════════════════════
with tab1:
    col1, col2 = st.columns(2)

    with col1:
        color_delta = "normal" if option_type == "Call" else "inverse"
        if model == "Black-Scholes":
            st.metric(
                label=f"📐 Black-Scholes {option_type} Price",
                value=f"${bs_price:.4f}",
                delta=f"Δ={grk_result['delta']:.4f}",
                delta_color=color_delta,
            )
        elif model == "Binomial Tree":
            st.metric(
                label=f"🌳 Binomial {option_type} Price",
                value=f"${bt_price:.4f}",
                delta=f"{binom_steps} steps",
                delta_color="off",
            )
        else:
            st.metric(
                label=f"📐 Black-Scholes Price",
                value=f"${bs_price:.4f}",
                delta=f"Δ={grk_result['delta']:.4f}",
                delta_color=color_delta,
            )

    with col2:
        if model == "Both" and bt_price is not None:
            diff = abs(bs_price - bt_price)
            st.metric(
                label=f"🌳 Binomial Price ({binom_steps} steps)",
                value=f"${bt_price:.4f}",
                delta=f"Diff from BS: ${diff:.4f}",
                delta_color="inverse" if diff > 0.01 else "off",
            )
        elif model == "Black-Scholes":
            st.metric(
                label="💡 Intrinsic Value",
                value=f"${bs_result['intrinsic']:.4f}",
                delta=f"Time Value: ${bs_result['time_value']:.4f}",
                delta_color="off",
            )
        else:
            st.metric(
                label="📐 Black-Scholes (Reference)",
                value=f"${bs_price:.4f}",
                delta=f"BS vs BT diff: ${abs(bs_price - bt_price):.4f}" if bt_price else "",
                delta_color="off",
            )

    st.markdown("<hr/>", unsafe_allow_html=True)

    col3, col4 = st.columns(2)

    with col3:
        # Intrinsic vs Time Value horizontal stacked bar
        intrinsic_val = bs_result["intrinsic"]
        time_val      = bs_result["time_value"]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Intrinsic Value",
            x=[intrinsic_val],
            y=["Option Price"],
            orientation="h",
            marker_color="#F59E0B",
            text=f"${intrinsic_val:.4f}",
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig_bar.add_trace(go.Bar(
            name="Time Value",
            x=[time_val],
            y=["Option Price"],
            orientation="h",
            marker_color="#3B82F6",
            text=f"${time_val:.4f}",
            textposition="inside",
            insidetextanchor="middle",
        ))
        fig_bar.update_layout(
            barmode="stack",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#0E1117",
            font=dict(family="JetBrains Mono, monospace", color="#8899AA", size=11),
            height=180,
            title=dict(text="Intrinsic vs Time Value", font=dict(size=12, color="#E8EDF5"), x=0.0),
            showlegend=True,
            margin=dict(l=10, r=10, t=40, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=10)),
            xaxis=dict(gridcolor="#2A3347", zerolinecolor="#2A3347", linecolor="#2A3347"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)", showticklabels=False),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col4:
        mnm_lbl, mnm_cls2, _ = moneyness_label(S, K, option_type)
        pct_dist = (S / K - 1) * 100
        st.markdown(f"""
        <div style="padding:20px 10px 10px 10px;">
          <div style="font-family:JetBrains Mono,monospace; font-size:0.72rem;
                      color:#4A5568; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:8px;">
            Moneyness
          </div>
          <div class="moneyness-badge {mnm_cls2}" style="font-size:1.5rem; margin-bottom:14px;">
            {mnm_lbl}
          </div>
          <div style="font-family:JetBrains Mono,monospace; font-size:0.85rem; color:#8899AA; margin-top:10px;">
            Spot:  <span style="color:#E8EDF5; font-weight:600;">${S:,.2f}</span><br>
            Strike: <span style="color:#E8EDF5; font-weight:600;">${K:,.2f}</span><br>
            Distance: <span style="color:#E8EDF5; font-weight:600;">{pct_dist:+.2f}%</span>
          </div>
          <div style="font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#4A5568; margin-top:12px;">
            d1 = {bs_result['d1']:.4f} &nbsp;|&nbsp; d2 = {bs_result['d2']:.4f}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
# TAB 2 — GREEKS
# ═══════════════════════════════════════════════
with tab2:
    g = grk_result

    def greek_interpret(name, val, option_type, S):
        if name == "Delta":
            return f"Option moves ${abs(val):.2f} for every $1 move in {ticker.upper()}"
        elif name == "Gamma":
            return f"Delta changes by {val:.4f} for each $1 move in the stock"
        elif name == "Vega":
            return f"Option gains/loses ${abs(val):.4f} per 1% change in volatility"
        elif name == "Theta":
            return f"Option loses ${abs(val):.4f} per calendar day (time decay)"
        elif name == "Rho":
            return f"Option {'gains' if val > 0 else 'loses'} ${abs(val):.4f} per 1% rise in rates"
        return ""

    greeks_data = [
        ("Delta", g["delta"],  "Δ"),
        ("Gamma", g["gamma"],  "Γ"),
        ("Vega",  g["vega"],   "ν"),
        ("Theta", g["theta"],  "Θ"),
        ("Rho",   g["rho"],    "ρ"),
    ]

    st.markdown("### Option Greeks")
    col_g1, col_g2 = st.columns([1, 1])

    with col_g1:
        for name, val, sym in greeks_data:
            interp = greek_interpret(name, val, option_type, S)
            val_color = "#10B981" if val >= 0 else "#EF4444"
            st.markdown(f"""
            <div class="greek-card">
              <div style="display:flex; justify-content:space-between; align-items:baseline;">
                <span style="font-family:Sora,sans-serif; font-weight:700; font-size:0.9rem;
                             color:#E8EDF5;">{sym} {name}</span>
                <span style="font-family:JetBrains Mono,monospace; font-size:1.15rem;
                             font-weight:700; color:{val_color};">{val:.4f}</span>
              </div>
              <div style="font-family:JetBrains Mono,monospace; font-size:0.72rem;
                          color:#8899AA; margin-top:4px;">{interp}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_g2:
        # Delta gauge
        delta_val = g["delta"]
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=delta_val,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Δ Delta", "font": {"size": 14, "color": "#E8EDF5", "family": "Sora"}},
            number={"font": {"color": "#3B82F6", "size": 36, "family": "JetBrains Mono"}, "valueformat": ".4f"},
            gauge={
                "axis": {"range": [-1, 1], "tickcolor": "#4A5568",
                         "tickfont": {"size": 9, "family": "JetBrains Mono"}, "tickwidth": 1},
                "bar": {"color": "#3B82F6", "thickness": 0.28},
                "bgcolor": "#161B27",
                "borderwidth": 1,
                "bordercolor": "#2A3347",
                "steps": [
                    {"range": [-1, 0], "color": "rgba(239,68,68,0.12)"},
                    {"range": [0, 1],  "color": "rgba(16,185,129,0.12)"},
                ],
                "threshold": {
                    "line": {"color": "#F59E0B", "width": 2},
                    "thickness": 0.8,
                    "value": delta_val,
                },
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor="#0E1117",
            font={"color": "#8899AA", "family": "JetBrains Mono"},
            height=260,
            margin=dict(l=20, r=20, t=40, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.markdown("---")
    st.markdown("### Greeks vs Spot Price")

    greek_choice = st.selectbox(
        "Select Greek to plot",
        options=["Delta", "Gamma", "Vega", "Theta", "Rho"],
        key="greek_plot_choice",
    )
    greek_key_map = {"Delta": "delta", "Gamma": "gamma", "Vega": "vega",
                     "Theta": "theta", "Rho": "rho"}

    S_range = np.linspace(0.5 * K, 1.5 * K, 200)
    greek_vals = [
        greeks(s, K, T_years, r, sigma, q, option_type)[greek_key_map[greek_choice]]
        for s in S_range
    ]

    fig_greek = go.Figure()
    fig_greek.add_trace(go.Scatter(
        x=S_range, y=greek_vals, mode="lines",
        name=greek_choice,
        line=dict(color="#3B82F6", width=2.5),
    ))
    fig_greek.add_vline(x=S, line_dash="dash", line_color="#F59E0B", line_width=1.5,
                        annotation_text=f"S={S:.2f}", annotation_font_color="#F59E0B",
                        annotation_font_size=10)
    apply_layout(fig_greek, title=f"{greek_choice} vs Spot Price", height=380)
    fig_greek.update_xaxes(title_text="Spot Price ($)", title_font_size=11)
    fig_greek.update_yaxes(title_text=greek_choice, title_font_size=11)
    st.plotly_chart(fig_greek, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 3 — PAYOFF DIAGRAM
# ═══════════════════════════════════════════════
with tab3:
    S_range_p = np.linspace(0.5 * K, 1.5 * K, 200)
    premium = bs_price

    # Payoff at expiry
    if option_type == "Call":
        payoff_expiry = np.maximum(S_range_p - K, 0) - premium
    else:
        payoff_expiry = np.maximum(K - S_range_p, 0) - premium

    # Current theoretical value (BS at current T)
    current_values = np.array([
        black_scholes(s, K, T_years, r, sigma, q, option_type)["price"] - premium
        for s in S_range_p
    ])

    # Break-even
    if option_type == "Call":
        breakeven = K + premium
    else:
        breakeven = K - premium

    fig_payoff = go.Figure()

    # Expiry payoff
    fig_payoff.add_trace(go.Scatter(
        x=S_range_p, y=payoff_expiry,
        mode="lines", name="P&L at Expiry",
        line=dict(color="#3B82F6", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(59,130,246,0.07)",
    ))

    # Current theoretical value
    fig_payoff.add_trace(go.Scatter(
        x=S_range_p, y=current_values,
        mode="lines", name=f"Current Value (T={days}d)",
        line=dict(color="#F59E0B", width=2, dash="dash"),
    ))

    # Zero line
    fig_payoff.add_hline(y=0, line_dash="dot", line_color="#4A5568", line_width=1)

    # Current spot
    fig_payoff.add_vline(x=S, line_dash="dash", line_color="#8899AA", line_width=1,
                         annotation_text=f"S={S:.2f}", annotation_font_color="#8899AA",
                         annotation_font_size=10)

    # Break-even
    if 0.5 * K <= breakeven <= 1.5 * K:
        fig_payoff.add_vline(x=breakeven, line_dash="dot", line_color="#10B981", line_width=1.5,
                             annotation_text=f"B/E={breakeven:.2f}",
                             annotation_font_color="#10B981", annotation_font_size=10)

    apply_layout(fig_payoff, title=f"P&L Diagram — Long {option_type} (Premium=${premium:.4f})", height=420)
    fig_payoff.update_xaxes(title_text="Spot Price at Expiry ($)", title_font_size=11)
    fig_payoff.update_yaxes(title_text="Profit / Loss ($)", title_font_size=11)

    # ── Strategy Overlays ──
    strategy = st.selectbox(
        "Add Strategy Overlay",
        options=["None", "Long Straddle", "Bull Call Spread", "Protective Put"],
        key="strategy_select",
    )

    if strategy != "None":
        spread_k2 = K * 1.05

        if strategy == "Long Straddle":
            prem_call = black_scholes(S, K, T_years, r, sigma, q, "Call")["price"]
            prem_put  = black_scholes(S, K, T_years, r, sigma, q, "Put")["price"]
            total_prem = prem_call + prem_put
            strat_pnl = (np.maximum(S_range_p - K, 0) + np.maximum(K - S_range_p, 0)) - total_prem
            strat_label = f"Long Straddle (prem=${total_prem:.2f})"

        elif strategy == "Bull Call Spread":
            prem_long  = black_scholes(S, K, T_years, r, sigma, q, "Call")["price"]
            prem_short = black_scholes(S, spread_k2, T_years, r, sigma, q, "Call")["price"]
            net_prem = prem_long - prem_short
            strat_pnl = (np.maximum(S_range_p - K, 0) - np.maximum(S_range_p - spread_k2, 0)) - net_prem
            strat_label = f"Bull Call Spread K={K:.0f}/{spread_k2:.0f} (net=${net_prem:.2f})"

        elif strategy == "Protective Put":
            prem_put = black_scholes(S, K, T_years, r, sigma, q, "Put")["price"]
            stock_pnl = S_range_p - S
            put_pnl   = np.maximum(K - S_range_p, 0) - prem_put
            strat_pnl = stock_pnl + put_pnl
            strat_label = f"Protective Put (K={K:.0f})"

        fig_payoff.add_trace(go.Scatter(
            x=S_range_p, y=strat_pnl,
            mode="lines", name=strat_label,
            line=dict(color="#A855F7", width=2.5, dash="longdash"),
        ))

    st.plotly_chart(fig_payoff, use_container_width=True)

    # Summary stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Premium Paid", f"${premium:.4f}")
    c2.metric("Break-Even", f"${breakeven:.2f}")
    c3.metric("Max Profit", "Unlimited" if option_type == "Call" else f"${K - premium:.4f}")
    c4.metric("Max Loss", f"${premium:.4f}")


# ═══════════════════════════════════════════════
# TAB 4 — HEATMAP
# ═══════════════════════════════════════════════
with tab4:
    vol_range  = np.linspace(0.05, 1.00, 20)  # 5% to 100%
    spot_range = np.linspace(0.5 * K, 1.5 * K, 20)

    # Vectorised computation: price heatmap
    SIGMA_GRID, S_GRID = np.meshgrid(vol_range, spot_range)  # shape (20, 20)

    def bs_price_vec(S_g, K, T, r, sig_g, q, otype):
        """Vectorised Black-Scholes for 2D grids."""
        eps = 1e-10
        safe_T   = np.maximum(T, eps)
        safe_sig = np.maximum(sig_g, eps)
        d1 = (np.log(S_g / K) + (r - q + 0.5 * safe_sig ** 2) * safe_T) / (safe_sig * np.sqrt(safe_T))
        d2 = d1 - safe_sig * np.sqrt(safe_T)
        if otype == "Call":
            price = S_g * np.exp(-q * safe_T) * norm.cdf(d1) - K * np.exp(-r * safe_T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * safe_T) * norm.cdf(-d2) - S_g * np.exp(-q * safe_T) * norm.cdf(-d1)
        return price

    def bs_delta_vec(S_g, K, T, r, sig_g, q, otype):
        """Vectorised delta for 2D grids."""
        eps = 1e-10
        safe_T   = np.maximum(T, eps)
        safe_sig = np.maximum(sig_g, eps)
        d1 = (np.log(S_g / K) + (r - q + 0.5 * safe_sig ** 2) * safe_T) / (safe_sig * np.sqrt(safe_T))
        if otype == "Call":
            return np.exp(-q * safe_T) * norm.cdf(d1)
        else:
            return np.exp(-q * safe_T) * (norm.cdf(d1) - 1)

    price_grid = bs_price_vec(S_GRID, K, T_years, r, SIGMA_GRID, q, option_type)
    delta_grid = bs_delta_vec(S_GRID, K, T_years, r, SIGMA_GRID, q, option_type)

    vol_ticks  = [f"{int(v*100)}%" for v in vol_range]
    spot_ticks = [f"${s:.0f}" for s in spot_range]

    # Price Heatmap
    fig_hm1 = go.Figure(go.Heatmap(
        z=price_grid,
        x=vol_range * 100,
        y=spot_range,
        colorscale="RdYlGn",
        colorbar=dict(title="Price ($)", tickfont=dict(family="JetBrains Mono", size=9)),
        hoverongaps=False,
        hovertemplate="Vol: %{x:.0f}%<br>Spot: $%{y:.2f}<br>Price: $%{z:.4f}<extra></extra>",
    ))

    # Mark current (S, sigma) with white star
    fig_hm1.add_trace(go.Scatter(
        x=[sigma * 100], y=[S],
        mode="markers",
        marker=dict(symbol="star", size=16, color="white",
                    line=dict(color="#0E1117", width=1.5)),
        name="Current",
        hovertemplate=f"Current: S=${S:.2f}, σ={vol_pct}%<extra></extra>",
    ))

    apply_layout(fig_hm1, title=f"{option_type} Price Heatmap (Spot × Volatility)", height=420)
    fig_hm1.update_xaxes(title_text="Volatility (%)", title_font_size=11)
    fig_hm1.update_yaxes(title_text="Spot Price ($)", title_font_size=11)
    st.plotly_chart(fig_hm1, use_container_width=True)

    # Delta Heatmap
    fig_hm2 = go.Figure(go.Heatmap(
        z=delta_grid,
        x=vol_range * 100,
        y=spot_range,
        colorscale="RdYlGn",
        colorbar=dict(title="Delta", tickfont=dict(family="JetBrains Mono", size=9)),
        zmin=-1, zmax=1,
        hoverongaps=False,
        hovertemplate="Vol: %{x:.0f}%<br>Spot: $%{y:.2f}<br>Delta: %{z:.4f}<extra></extra>",
    ))
    fig_hm2.add_trace(go.Scatter(
        x=[sigma * 100], y=[S],
        mode="markers",
        marker=dict(symbol="star", size=16, color="white",
                    line=dict(color="#0E1117", width=1.5)),
        name="Current",
        hovertemplate=f"Current: S=${S:.2f}, σ={vol_pct}%<extra></extra>",
    ))

    apply_layout(fig_hm2, title=f"{option_type} Delta Heatmap (Spot × Volatility)", height=420)
    fig_hm2.update_xaxes(title_text="Volatility (%)", title_font_size=11)
    fig_hm2.update_yaxes(title_text="Spot Price ($)", title_font_size=11)
    st.plotly_chart(fig_hm2, use_container_width=True)


# ═══════════════════════════════════════════════
# TAB 5 — BINOMIAL TREE
# ═══════════════════════════════════════════════
with tab5:
    if bt_result is None:
        # Need to compute even if model != binomial
        bt_result_vis = binomial_tree(S, K, T_years, r, sigma, q, binom_steps, option_type, american)
    else:
        bt_result_vis = bt_result

    vis_steps = bt_result_vis["vis_steps"]
    tree_S    = bt_result_vis["tree_S"]
    tree_V    = bt_result_vis["tree_V"]

    n_vis = st.slider("Tree Depth to Display", min_value=2, max_value=min(10, vis_steps),
                      value=min(6, vis_steps), key="tree_depth")

    # Build node positions
    node_x, node_y, node_text, node_color, node_hover = [], [], [], [], []
    edge_x, edge_y = [], []

    # Collect min/max option values for color scaling
    all_v = [tree_V.get((step, node), 0.0) for step in range(n_vis + 1)
             for node in range(step + 1)]
    v_min, v_max = min(all_v), max(all_v)
    v_range = v_max - v_min if v_max != v_min else 1.0

    def val_to_color(v):
        t = (v - v_min) / v_range
        # red → yellow → green
        r_c = int(239 * (1 - t) + 16 * t)
        g_c = int(68  * (1 - t) + 185 * t)
        b_c = int(68  * (1 - t) + 129 * t)
        return f"rgb({r_c},{g_c},{b_c})"

    for step in range(n_vis + 1):
        for node in range(step + 1):
            x_pos = step
            # Centre nodes vertically: spread from -step to +step in steps of 2
            y_pos = -step + 2 * node  # gives symmetric layout
            s_val = tree_S.get((step, node), 0.0)
            v_val = tree_V.get((step, node), 0.0)

            node_x.append(x_pos)
            node_y.append(y_pos)
            node_text.append(f"${s_val:.2f}<br><b>${v_val:.3f}</b>")
            node_color.append(val_to_color(v_val))
            node_hover.append(f"Step {step}, Node {node}<br>S=${s_val:.4f}<br>V=${v_val:.4f}")

            # Edges to children
            if step < n_vis:
                cx = step + 1
                # up child
                uy = y_pos + 1
                # down child
                dy = y_pos - 1
                # up edge
                edge_x += [x_pos, cx, None]
                edge_y += [y_pos, uy, None]
                # down edge
                edge_x += [x_pos, cx, None]
                edge_y += [y_pos, dy, None]

    fig_tree = go.Figure()

    # Edges
    fig_tree.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(color="#2A3347", width=1.2),
        hoverinfo="none",
        showlegend=False,
    ))

    # Nodes
    fig_tree.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        marker=dict(size=30, color=node_color,
                    line=dict(color="#0E1117", width=1.5),
                    symbol="circle"),
        text=node_text,
        textfont=dict(size=7, family="JetBrains Mono", color="#E8EDF5"),
        textposition="middle center",
        hovertext=node_hover,
        hoverinfo="text",
        showlegend=False,
    ))

    apply_layout(fig_tree, title=f"CRR Binomial Tree — {n_vis} Steps (top=S, bold=V)", height=500)
    fig_tree.update_xaxes(title_text="Step", showgrid=False, dtick=1)
    fig_tree.update_yaxes(showgrid=False, showticklabels=False)
    st.plotly_chart(fig_tree, use_container_width=True)

    st.markdown(f"""
    <div style="font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#4A5568; margin-top:-10px; margin-bottom:12px;">
      u={bt_result_vis['u']:.4f} &nbsp;·&nbsp; d={bt_result_vis['d']:.4f} &nbsp;·&nbsp; p={bt_result_vis['p']:.4f}
      &nbsp;·&nbsp; Steps={binom_steps}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Convergence: Binomial → Black-Scholes")

    step_grid = [10, 20, 50, 100, 200, 500]
    binom_prices = [
        binomial_tree(S, K, T_years, r, sigma, q, n, option_type, american)["price"]
        for n in step_grid
    ]

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=step_grid, y=binom_prices,
        mode="lines+markers",
        name="Binomial Price",
        line=dict(color="#3B82F6", width=2.5),
        marker=dict(size=8, color="#3B82F6", symbol="circle"),
    ))
    fig_conv.add_hline(
        y=bs_price,
        line_dash="dash", line_color="#EF4444", line_width=2,
        annotation_text=f"BS Price = ${bs_price:.4f}",
        annotation_font_color="#EF4444",
        annotation_font_size=11,
    )
    apply_layout(fig_conv, title="Convergence of Binomial Tree to Black-Scholes", height=360)
    fig_conv.update_xaxes(title_text="Number of Steps", title_font_size=11, type="log")
    fig_conv.update_yaxes(title_text="Option Price ($)", title_font_size=11)
    st.plotly_chart(fig_conv, use_container_width=True)


# ═══════════════════════════════════════════════
# IV SOLVER EXPANDER
# ═══════════════════════════════════════════════
with st.expander("🔍 Implied Volatility Solver", expanded=False):
    st.markdown("""
    <div style="font-family:JetBrains Mono,monospace; font-size:0.72rem; color:#8899AA; margin-bottom:12px;">
      Enter a market-observed option price to back-solve for implied volatility using Newton-Raphson.
    </div>
    """, unsafe_allow_html=True)

    iv_col1, iv_col2, iv_col3 = st.columns([1, 1, 1])

    with iv_col1:
        market_price_input = st.number_input(
            "Market Price ($)",
            min_value=0.001, step=0.01,
            value=round(bs_price, 2),
            format="%.4f",
            key="market_price_iv",
        )

    with iv_col2:
        iv_opt_type = st.radio("Option Type", ["Call", "Put"],
                               horizontal=True, index=0 if option_type == "Call" else 1,
                               key="iv_opt_type")

    with iv_col3:
        solve_btn = st.button("⚡  Solve for Implied Vol", key="iv_solve_btn")

    if solve_btn:
        iv_res = implied_volatility(
            market_price_input, S, K, T_years, r, q, iv_opt_type
        )
        st.session_state.iv_result = iv_res

    if st.session_state.iv_result:
        iv_res = st.session_state.iv_result
        iv_val = iv_res["iv"]
        iv_iters = iv_res["iterations"]
        iv_conv = iv_res["converged"]

        iv_c1, iv_c2, iv_c3 = st.columns(3)

        if iv_conv and not np.isnan(iv_val):
            iv_c1.metric("Implied Volatility", f"{iv_val * 100:.4f}%",
                         delta=f"vs model σ={vol_pct:.1f}%")
            iv_c2.metric("Iterations", f"{iv_iters}")
            iv_c3.metric("Status", "✅ Converged")

            if st.button("📌  Apply IV to Model", key="apply_iv_btn"):
                st.session_state.apply_iv = True
                st.rerun()
        else:
            iv_c1.metric("Implied Volatility", "N/A")
            iv_c2.metric("Iterations", f"{iv_iters}")
            iv_c3.metric("Status", "❌ No Convergence")
            st.warning("⚠ Could not converge — check that market price is within theoretical bounds for this option.")
