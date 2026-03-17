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
    page_title="Options Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS (Institutional / Minimalist)
# ─────────────────────────────────────────────
st.markdown("""
<style>
  /* Root vars */
  :root {
    --bg-base: #121212;
    --bg-surface: #1E1E1E;
    --border-color: #2D2D2D;
    --text-primary: #EDEDED;
    --text-secondary: #A0A0A0;
    --accent-blue: #2962FF;
    --accent-green: #00C853;
    --accent-red: #D50000;
    --font-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  }

  /* Base */
  html, body, [class*="css"] {
    font-family: var(--font-sans);
    background-color: var(--bg-base);
    color: var(--text-primary);
  }

  /* Sidebar */
  section[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid var(--border-color);
  }
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stNumberInput label,
  section[data-testid="stSidebar"] .stTextInput label,
  section[data-testid="stSidebar"] .stRadio label,
  section[data-testid="stSidebar"] .stSelectbox label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    padding: 12px 16px !important;
    box-shadow: none !important;
  }
  [data-testid="metric-container"] label {
    color: var(--text-secondary) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: var(--font-mono) !important;
    font-size: 1.5rem !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-family: var(--font-mono) !important;
    font-size: 0.8rem !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border-color) !important;
    gap: 24px;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: var(--font-sans) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    padding: 8px 0px !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
  }
  .stTabs [aria-selected="true"] {
    color: var(--text-primary) !important;
    border-bottom: 2px solid var(--accent-blue) !important;
    background: transparent !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: var(--bg-surface) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
  }

  /* Buttons */
  .stButton > button {
    background: var(--bg-surface) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 4px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 4px 12px !important;
    transition: background 0.2s ease;
  }
  .stButton > button:hover {
    background: #2A2A2A !important;
    border-color: #404040 !important;
    color: #FFF !important;
  }

  /* Headers */
  h1, h2, h3 {
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
  }

  /* Custom tags */
  .mono-tag {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    padding: 2px 6px;
    border-radius: 4px;
    background: #2A2A2A;
    border: 1px solid #404040;
    color: #CCC;
  }
  .status-live { color: #00C853; border-color: rgba(0,200,83,0.3); background: rgba(0,200,83,0.05); }
  
  .tag-itm { color: #00C853; }
  .tag-otm { color: #FF5252; }
  .tag-atm { color: #448AFF; }

  /* Greek row highlight */
  .greek-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border-color);
  }
  .greek-row:last-child { border-bottom: none; }
  .greek-label { font-weight: 500; font-size: 0.85rem; color: var(--text-secondary); }
  .greek-val { font-family: var(--font-mono); font-size: 1rem; font-weight: 600; }

  hr { border-color: var(--border-color) !important; margin: 24px 0 !important; }
  input[type="number"], input[type="text"] { font-family: var(--font-mono) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PRICING ENGINE
# ─────────────────────────────────────────────

def black_scholes(S, K, T, r, sigma, q=0.0, option_type="Call") -> dict:
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
    if T <= 0 or sigma <= 0:
        if option_type == "Call":
            delta = 1.0 if S > K else (0.5 if S == K else 0.0)
        else:
            delta = -1.0 if S < K else (-0.5 if S == K else 0.0)
        return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}

    bs = black_scholes(S, K, T, r, sigma, q, option_type)
    d1, d2 = bs["d1"], bs["d2"]
    nd1_pdf = norm.pdf(d1)

    if option_type == "Call":
        delta = np.exp(-q * T) * norm.cdf(d1)
        theta = (-S * nd1_pdf * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        theta = (-S * nd1_pdf * sigma * np.exp(-q * T) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = np.exp(-q * T) * nd1_pdf / (S * sigma * np.sqrt(T))
    vega = S * np.exp(-q * T) * nd1_pdf * np.sqrt(T) / 100

    return {
        "delta": float(delta), "gamma": float(gamma),
        "vega":  float(vega), "theta": float(theta), "rho": float(rho)
    }

def binomial_tree(S, K, T, r, sigma, q=0.0, steps=100, option_type="Call", american=False) -> dict:
    steps = min(steps, 1000)
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    discount = np.exp(-r * dt)
    p = np.clip((np.exp((r - q) * dt) - d) / (u - d), 0.0, 1.0)

    j_arr = np.arange(steps + 1)
    S_T = S * (u ** j_arr) * (d ** (steps - j_arr))
    V = np.maximum(S_T - K, 0.0) if option_type == "Call" else np.maximum(K - S_T, 0.0)

    vis_steps = min(10, steps)
    node_values = {}
    if vis_steps == steps: node_values[steps] = V.copy()

    for i in range(steps - 1, -1, -1):
        V = discount * (p * V[1:i + 2] + (1 - p) * V[0:i + 1])
        if american:
            S_i = S * (u ** np.arange(i + 1)) * (d ** (i - np.arange(i + 1)))
            intrinsic = np.maximum(S_i - K, 0.0) if option_type == "Call" else np.maximum(K - S_i, 0.0)
            V = np.maximum(V, intrinsic)
        if i <= vis_steps: node_values[i] = V.copy()

    price = float(V[0])

    tree_S, tree_V_nodes = {}, {}
    for step in range(vis_steps + 1):
        for node in range(step + 1):
            tree_S[(step, node)] = S * (u ** node) * (d ** (step - node))
            tree_V_nodes[(step, node)] = float(node_values[step][node])

    return {"price": price, "tree_S": tree_S, "tree_V": tree_V_nodes, "vis_steps": vis_steps, "u": u, "d": d, "p": p}

def implied_volatility(market_price, S, K, T, r, q=0.0, option_type="Call", tol=1e-6, max_iter=100) -> dict:
    if T <= 0 or market_price <= 0: return {"iv": np.nan, "iterations": 0, "converged": False}
    moneyness = np.log(S / K) + r * T
    sigma = max(np.sqrt(2 * abs(moneyness) / T) if abs(moneyness) > 1e-10 else 0.3, 0.01)

    for i in range(max_iter):
        bs_result = black_scholes(S, K, T, r, sigma, q, option_type)
        vega_val = greeks(S, K, T, r, sigma, q, option_type)["vega"] * 100
        if abs(vega_val) < 1e-10: return {"iv": np.nan, "iterations": i + 1, "converged": False}
        sigma_new = max(sigma - (bs_result["price"] - market_price) / vega_val, 1e-6)
        if abs(sigma_new - sigma) < tol: return {"iv": float(sigma_new), "iterations": i + 1, "converged": True}
        sigma = sigma_new

    try:
        def obj(s): return black_scholes(S, K, T, r, s, q, option_type)["price"] - market_price
        return {"iv": float(brentq(obj, 1e-6, 10.0, xtol=tol)), "iterations": max_iter, "converged": True}
    except: return {"iv": np.nan, "iterations": max_iter, "converged": False}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="ui-monospace, SFMono-Regular, monospace", color="#A0A0A0", size=10),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor="#2D2D2D", zerolinecolor="#2D2D2D"),
    yaxis=dict(gridcolor="#2D2D2D", zerolinecolor="#2D2D2D"),
    legend=dict(bgcolor="rgba(18,18,18,0.8)", bordercolor="#2D2D2D", borderwidth=1),
)

def apply_layout(fig, title="", height=400):
    fig.update_layout(**PLOTLY_LAYOUT, title=dict(text=title, font=dict(size=13, color="#EDEDED", family="-apple-system, sans-serif")), height=height)
    return fig

def fetch_spot_price(ticker_str):
    try:
        import yfinance as yf
        price = yf.Ticker(ticker_str.strip().upper()).fast_info["lastPrice"]
        return float(price), True if price and price > 0 else (None, False)
    except: return None, False

def get_moneyness(S, K, option_type):
    pct = (S / K - 1) * 100
    if abs(pct) < 1.0: return "ATM", "tag-atm", pct
    if option_type == "Call": return "ITM" if pct > 0 else "OTM", "tag-itm" if pct > 0 else "tag-otm", pct
    return "ITM" if pct < 0 else "OTM", "tag-itm" if pct < 0 else "tag-otm", pct

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
for key, val in [("spot_price", 195.0), ("live_price", None), ("live_ok", False), ("iv_result", None), ("apply_iv", False)]:
    if key not in st.session_state: st.session_state[key] = val

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h3 style='font-size:1.1rem; margin-bottom:0;'>Options Analytics</h3>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.75rem; color:#A0A0A0; margin-bottom:20px;'>Pricing & Risk Engine</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    ticker = c1.text_input("Ticker", value="AAPL", key="ticker_input")
    if c2.button("Quote", use_container_width=True):
        price, ok = fetch_spot_price(ticker)
        st.session_state.update(live_price=price, live_ok=ok, spot_price=price if ok else st.session_state.spot_price)
        if not ok: st.error("Quote failed")

    if st.session_state.live_ok:
        st.markdown(f'<span class="mono-tag status-live">Live: ${st.session_state.live_price:.2f}</span>', unsafe_allow_html=True)
    
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

    S = st.number_input("Spot Price (S)", min_value=0.01, step=1.0, value=float(st.session_state.spot_price), format="%.2f")
    K = st.number_input("Strike Price (K)", min_value=0.01, step=1.0, value=float(S), format="%.2f")
    days = st.slider("Days to Expiry", 1, 730, 90)
    T_years = days / 365.0

    if st.session_state.apply_iv and st.session_state.iv_result:
        iv = st.session_state.iv_result.get("iv", np.nan)
        if not np.isnan(iv): st.session_state["vol_slider"] = max(1, min(200, int(iv * 100)))
        st.session_state.apply_iv = False

    vol_pct = st.slider("Volatility (%)", 1, 200, 25, key="vol_slider")
    sigma = vol_pct / 100.0

    r = st.number_input("Risk-Free Rate (%)", 0.0, 20.0, 5.0, step=0.1) / 100.0
    q = st.number_input("Dividend Yield (%)", 0.0, 20.0, 0.0, step=0.1) / 100.0

    st.markdown("---")
    option_type = st.radio("Option Type", ["Call", "Put"], horizontal=True)
    model = st.selectbox("Pricing Model", ["Black-Scholes", "Binomial Tree", "Both"])
    binom_steps = st.slider("Binomial Steps", 10, 500, 100) if model != "Black-Scholes" else 100
    american = st.checkbox("American Style")

# ─────────────────────────────────────────────
# VALIDATION GUARDS
# ─────────────────────────────────────────────
if S <= 0: st.error("Spot price must be > 0"); st.stop()
if T_years <= 0 or sigma <= 0:
    intr = max(S-K,0) if option_type=="Call" else max(K-S,0)
    st.info(f"Option expired or 0 vol. Intrinsic value: ${intr:.4f}")
    st.stop()

# ─────────────────────────────────────────────
# COMPUTE
# ─────────────────────────────────────────────
bs_res = black_scholes(S, K, T_years, r, sigma, q, option_type)
grk_res = greeks(S, K, T_years, r, sigma, q, option_type)
bt_res = binomial_tree(S, K, T_years, r, sigma, q, binom_steps, option_type, american) if model in ["Binomial Tree", "Both"] else None

bs_price = bs_res["price"]
bt_price = bt_res["price"] if bt_res else None

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
mnm, mnm_cls, pct_dist = get_moneyness(S, K, option_type)
st.markdown(f"""
<div style='display:flex; justify-content:space-between; align-items:flex-end; margin-bottom: 24px;'>
  <div>
    <h2 style='margin:0 0 4px 0;'>{ticker.upper()} {option_type}</h2>
    <span class='mono-tag'>S={S:.2f} | K={K:.2f} | T={days}d | σ={vol_pct}% | r={r*100:.1f}%</span>
  </div>
  <div style='text-align:right;'>
    <span class='mono-tag {mnm_cls}' style='font-size:0.85rem; border:none; background:transparent;'>{mnm} {pct_dist:+.2f}%</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tabs = st.tabs(["Pricing Summary", "Greeks", "Payoff Diagram", "Heatmap", "Binomial Tree"])

# ══ TAB 1: SUMMARY ══
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    if model in ["Black-Scholes", "Both"]:
        c1.metric("Black-Scholes Price", f"${bs_price:.4f}")
    if model in ["Binomial Tree", "Both"]:
        c2.metric("Binomial Price", f"${bt_price:.4f}")
    c3.metric("Intrinsic Value", f"${bs_res['intrinsic']:.4f}")
    c4.metric("Time Value", f"${bs_res['time_value']:.4f}")

    st.markdown("---")
    
    # Intrinsic vs Time Value Bar
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(name="Intrinsic", x=[bs_res["intrinsic"]], y=["Value"], orientation="h", marker_color="#2962FF"))
    fig_bar.add_trace(go.Bar(name="Time Value", x=[bs_res["time_value"]], y=["Value"], orientation="h", marker_color="#5C6BC0"))
    fig_bar.update_layout(
        barmode="stack", height=120, margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Premium Composition", font=dict(size=12, color="#EDEDED"), x=0),
        showlegend=True, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, showticklabels=False),
        legend=dict(orientation="h", y=1.2, font=dict(family="ui-monospace", size=10, color="#A0A0A0"))
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ══ TAB 2: GREEKS ══
with tabs[1]:
    col_g1, col_g2 = st.columns([1, 1.2])
    
    with col_g1:
        st.markdown("<div style='background: var(--bg-surface); border: 1px solid var(--border-color); border-radius: 6px; padding: 4px;'>", unsafe_allow_html=True)
        for name, val, sym in [("Delta", grk_res["delta"], "Δ"), ("Gamma", grk_res["gamma"], "Γ"), ("Vega", grk_res["vega"], "ν"), ("Theta", grk_res["theta"], "Θ"), ("Rho", grk_res["rho"], "ρ")]:
            color = "#00C853" if val >= 0 else "#FF5252"
            st.markdown(f"""
            <div class='greek-row'>
              <span class='greek-label'>{sym} {name}</span>
              <span class='greek-val' style='color:{color}'>{val:.4f}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_g2:
        S_rng = np.linspace(0.5 * K, 1.5 * K, 100)
        sel_grk = st.selectbox("Greek Profile", ["Delta", "Gamma", "Vega", "Theta", "Rho"], label_visibility="collapsed")
        g_map = {"Delta": "delta", "Gamma": "gamma", "Vega": "vega", "Theta": "theta", "Rho": "rho"}
        vals = [greeks(s, K, T_years, r, sigma, q, option_type)[g_map[sel_grk]] for s in S_rng]
        
        fig = go.Figure(go.Scatter(x=S_rng, y=vals, line=dict(color="#2962FF", width=2)))
        fig.add_vline(x=S, line_dash="dash", line_color="#A0A0A0", annotation_text=f"Spot={S:.2f}")
        apply_layout(fig, title=f"{sel_grk} vs Spot", height=280)
        st.plotly_chart(fig, use_container_width=True)

# ══ TAB 3: PAYOFF ══
with tabs[2]:
    S_rng = np.linspace(0.5 * K, 1.5 * K, 200)
    payoff = np.maximum(S_rng - K, 0) - bs_price if option_type == "Call" else np.maximum(K - S_rng, 0) - bs_price
    cur_val = np.array([black_scholes(s, K, T_years, r, sigma, q, option_type)["price"] - bs_price for s in S_rng])
    be = K + bs_price if option_type == "Call" else K - bs_price

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S_rng, y=payoff, name="Expiry", line=dict(color="#2962FF"), fill="tozeroy", fillcolor="rgba(41,98,255,0.1)"))
    fig.add_trace(go.Scatter(x=S_rng, y=cur_val, name="Current", line=dict(color="#FF8F00", dash="dash")))
    fig.add_hline(y=0, line_color="#404040", line_width=1)
    fig.add_vline(x=S, line_color="#A0A0A0", line_dash="dot", annotation_text=f"S={S:.2f}")
    if 0.5*K <= be <= 1.5*K: fig.add_vline(x=be, line_color="#00C853", line_dash="dot", annotation_text=f"B/E={be:.2f}")

    apply_layout(fig, title=f"P&L Profile (Premium: ${bs_price:.2f})")
    st.plotly_chart(fig, use_container_width=True)

# ══ TAB 4: HEATMAP ══
with tabs[3]:
    v_rng = np.linspace(0.05, 1.0, 20)
    s_rng = np.linspace(0.5 * K, 1.5 * K, 20)
    VG, SG = np.meshgrid(v_rng, s_rng)
    
    # Vectorized BS
    d1 = (np.log(SG / K) + (r - q + 0.5 * VG**2) * T_years) / (VG * np.sqrt(T_years))
    d2 = d1 - VG * np.sqrt(T_years)
    if option_type == "Call": p_grid = SG * np.exp(-q*T_years)*norm.cdf(d1) - K*np.exp(-r*T_years)*norm.cdf(d2)
    else: p_grid = K * np.exp(-r*T_years)*norm.cdf(-d2) - SG*np.exp(-q*T_years)*norm.cdf(-d1)

    fig = go.Figure(go.Heatmap(z=p_grid, x=v_rng*100, y=s_rng, colorscale="Blues"))
    fig.add_trace(go.Scatter(x=[sigma*100], y=[S], mode="markers", marker=dict(color="#FF8F00", size=10, symbol="x"), name="Current"))
    apply_layout(fig, title="Theoretical Price Matrix")
    fig.update_xaxes(title="Volatility (%)")
    fig.update_yaxes(title="Spot Price ($)")
    st.plotly_chart(fig, use_container_width=True)

# ══ TAB 5: TREE ══
with tabs[4]:
    if bt_res is None: bt_res = binomial_tree(S, K, T_years, r, sigma, q, binom_steps, option_type, american)
    n_vis = st.slider("Display Depth", 2, bt_res["vis_steps"], min(5, bt_res["vis_steps"]))
    
    nx, ny, nt, nc, ex, ey = [], [], [], [], [], []
    v_max = max(bt_res["tree_V"].values())
    
    for stp in range(n_vis + 1):
        for nd in range(stp + 1):
            y = -stp + 2*nd
            v = bt_res["tree_V"][(stp, nd)]
            nx.append(stp); ny.append(y)
            nt.append(f"${bt_res['tree_S'][(stp,nd)]:.2f}<br>{v:.2f}")
            nc.append(f"rgba(41,98,255,{max(0.1, v/v_max if v_max>0 else 0)})")
            
            if stp < n_vis:
                ex.extend([stp, stp+1, None, stp, stp+1, None])
                ey.extend([y, y+1, None, y, y-1, None])

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", line=dict(color="#2D2D2D", width=1), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=nx, y=ny, mode="markers+text", marker=dict(size=40, color=nc, line=dict(color="#2D2D2D")), text=nt, textfont=dict(size=9, color="#EDEDED")))
    apply_layout(fig, title=f"Binomial Lattice (Nodes: S / V)", height=450)
    fig.update_xaxes(showgrid=False, zeroline=False, dtick=1); fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────
# IV SOLVER 
# ─────────────────────────────────────────────
with st.expander("Implied Volatility Solver", expanded=False):
    st.markdown("<p style='font-size:0.8rem; color:#A0A0A0;'>Calculate implied volatility using Newton-Raphson.</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1, 1])
    m_price = c1.number_input("Observed Price ($)", min_value=0.001, value=max(0.001, round(bs_price, 2)), step=0.01)
    iv_opt = c2.radio("Option Type", ["Call", "Put"], horizontal=True, key="iv_opt")
    
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True) # alignment spacer
    if c3.button("Solve IV", use_container_width=True):
        st.session_state.iv_result = implied_volatility(m_price, S, K, T_years, r, q, iv_opt)

    if st.session_state.iv_result:
        res = st.session_state.iv_result
        if res["converged"] and not np.isnan(res["iv"]):
            st.success(f"Converged: **{res['iv']*100:.2f}%** in {res['iterations']} iterations.")
            if st.button("Apply to Model"):
                st.session_state.apply_iv = True
                st.rerun()
        else:
            st.error("Failed to converge. Price may violate theoretical bounds.")