from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from ai_copilot import (
    DEFAULT_MODEL,
    build_copilot_payload,
    generate_action_roadmap,
    generate_owner_brief,
)
from analyzer import build_insights, prepare_sales_data


st.set_page_config(
    page_title="AI Sales Co-Pilot",
    page_icon="📈",
    layout="wide",
)


def load_uploaded_data(uploaded_file) -> pd.DataFrame:
    file_name = uploaded_file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def format_currency(value: float) -> str:
    return f"Rs. {value:,.0f}"


def format_number(value: float) -> str:
    return f"{value:,.0f}"


def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, default)
    except StreamlitSecretNotFoundError:
        return default


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=DM+Sans:wght@400;500;700&display=swap');

        :root {
            --bg: #f6f1e8;
            --paper: rgba(255, 252, 246, 0.88);
            --panel: rgba(19, 48, 74, 0.08);
            --ink: #13243a;
            --muted: #5d6a78;
            --gold: #c9852d;
            --teal: #0e8f79;
            --brick: #c0543f;
            --line: rgba(19, 36, 58, 0.12);
            --shadow: 0 18px 45px rgba(24, 40, 59, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(201, 133, 45, 0.18), transparent 28%),
                radial-gradient(circle at top right, rgba(14, 143, 121, 0.16), transparent 24%),
                linear-gradient(180deg, #f6f1e8 0%, #f2ebe0 48%, #efe6d7 100%);
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            max-width: 1180px;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            letter-spacing: -0.03em;
        }

        [data-testid="stSidebar"] {
            background: rgba(19, 36, 58, 0.95);
        }

        [data-testid="stSidebar"] * {
            color: #f8f4ec;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] li,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] .stMarkdown {
            color: #f8f4ec !important;
        }

        [data-testid="stSidebar"] code {
            background: rgba(255, 216, 159, 0.16) !important;
            color: #ffe3b9 !important;
            border: 1px solid rgba(255, 216, 159, 0.22);
            border-radius: 8px;
            padding: 0.1rem 0.38rem;
        }

        [data-testid="stSidebar"] [data-baseweb="input"] input,
        [data-testid="stSidebar"] [data-baseweb="base-input"] input,
        [data-testid="stSidebar"] textarea {
            background: rgba(255, 248, 239, 0.96) !important;
            color: #13243a !important;
            border-radius: 14px !important;
        }

        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {
            color: #66768a !important;
            opacity: 1;
        }

        [data-testid="stSidebar"] [data-baseweb="input"] svg,
        [data-testid="stSidebar"] [data-baseweb="base-input"] svg {
            fill: #13243a !important;
            color: #13243a !important;
        }

        [data-testid="stFileUploader"] section,
        [data-testid="stExpander"],
        div[data-testid="stMetric"],
        div[data-testid="stDataFrame"] {
            border-radius: 22px !important;
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(19, 36, 58, 0.96), rgba(18, 65, 83, 0.94));
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 28px;
            padding: 2rem 2rem 1.6rem 2rem;
            box-shadow: var(--shadow);
            overflow: hidden;
            position: relative;
            margin-bottom: 1.2rem;
        }

        .hero-shell::after {
            content: "";
            position: absolute;
            inset: auto -60px -90px auto;
            width: 220px;
            height: 220px;
            border-radius: 999px;
            background: radial-gradient(circle, rgba(201, 133, 45, 0.30), transparent 65%);
        }

        .eyebrow {
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: #ffd89f;
            margin-bottom: 0.75rem;
        }

        .hero-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 3rem;
            line-height: 0.96;
            color: #fff8ef;
            margin: 0 0 0.8rem 0;
            max-width: 720px;
        }

        .hero-copy {
            color: rgba(248, 244, 236, 0.86);
            font-size: 1.02rem;
            max-width: 640px;
            margin-bottom: 1.1rem;
        }

        .hero-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.8rem;
            margin-top: 1rem;
        }

        .hero-pill {
            background: rgba(255, 248, 239, 0.08);
            border: 1px solid rgba(255, 248, 239, 0.10);
            border-radius: 18px;
            padding: 0.9rem 1rem;
            color: #fff8ef;
        }

        .hero-pill strong {
            display: block;
            font-size: 1rem;
            margin-bottom: 0.15rem;
        }

        .section-card {
            background: var(--paper);
            border: 1px solid var(--line);
            border-radius: 24px;
            padding: 1rem 1.15rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(10px);
            margin-bottom: 1rem;
        }

        .section-label {
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: var(--muted);
            font-weight: 700;
            margin-bottom: 0.3rem;
        }

        .kpi-card {
            background: linear-gradient(180deg, rgba(255, 252, 246, 0.94), rgba(252, 246, 235, 0.92));
            border: 1px solid rgba(19, 36, 58, 0.08);
            border-radius: 22px;
            padding: 1rem 1rem 0.9rem 1rem;
            box-shadow: var(--shadow);
        }

        .snapshot-card {
            background: linear-gradient(160deg, rgba(255, 252, 246, 0.96), rgba(242, 235, 224, 0.92));
            border: 1px solid rgba(19, 36, 58, 0.08);
            border-radius: 24px;
            padding: 1.1rem 1.1rem 1rem 1.1rem;
            box-shadow: var(--shadow);
            min-height: 180px;
        }

        .snapshot-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .snapshot-title {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            font-size: 1.6rem;
            line-height: 1.05;
            margin-bottom: 0.35rem;
        }

        .snapshot-copy {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 0.85rem;
        }

        .snapshot-stat {
            color: var(--ink);
            font-size: 1rem;
            font-weight: 700;
        }

        .briefing-shell {
            background: linear-gradient(140deg, rgba(19, 36, 58, 0.98), rgba(23, 88, 93, 0.93));
            border-radius: 28px;
            padding: 1.2rem;
            border: 1px solid rgba(255, 248, 239, 0.12);
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .briefing-shell h3 {
            color: #fff8ef;
            margin: 0.25rem 0 0.5rem 0;
        }

        .briefing-shell p {
            color: rgba(248, 244, 236, 0.84);
            margin-bottom: 0;
        }

        .kpi-label {
            color: var(--muted);
            font-size: 0.83rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .kpi-value {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 2rem;
            line-height: 1;
            color: var(--ink);
            margin-bottom: 0.25rem;
        }

        .kpi-foot {
            color: var(--muted);
            font-size: 0.92rem;
        }

        .insight-card {
            border-radius: 20px;
            padding: 1rem 1rem;
            margin-bottom: 0.75rem;
            background: linear-gradient(135deg, rgba(14, 143, 121, 0.10), rgba(201, 133, 45, 0.10));
            border: 1px solid rgba(14, 143, 121, 0.15);
        }

        .insight-card strong {
            display: block;
            font-family: 'Space Grotesk', sans-serif;
            margin-bottom: 0.2rem;
            color: var(--ink);
        }

        .alert-banner {
            border-radius: 22px;
            padding: 1rem 1.05rem;
            margin-bottom: 0.9rem;
            border: 1px solid rgba(19, 36, 58, 0.10);
            box-shadow: var(--shadow);
        }

        .alert-banner strong {
            display: block;
            margin-bottom: 0.2rem;
            font-family: 'Space Grotesk', sans-serif;
        }

        .alert-risk {
            background: linear-gradient(135deg, rgba(192, 84, 63, 0.12), rgba(201, 133, 45, 0.12));
        }

        .alert-opportunity {
            background: linear-gradient(135deg, rgba(14, 143, 121, 0.12), rgba(201, 133, 45, 0.10));
        }

        .mini-note {
            color: var(--muted);
            font-size: 0.94rem;
        }

        .upload-tip {
            background: rgba(201, 133, 45, 0.10);
            border: 1px solid rgba(201, 133, 45, 0.18);
            color: var(--ink);
            border-radius: 18px;
            padding: 0.85rem 1rem;
            margin-top: 0.7rem;
        }

        [data-testid="stSidebar"] .upload-tip {
            background: rgba(255, 248, 239, 0.08);
            border: 1px solid rgba(201, 133, 45, 0.25);
            color: #f8f4ec;
        }

        [data-testid="stSidebar"] .upload-tip strong {
            color: #ffe3b9;
        }

        .copilot-shell {
            background: linear-gradient(135deg, rgba(19, 36, 58, 0.98), rgba(38, 72, 50, 0.92));
            border: 1px solid rgba(255, 248, 239, 0.12);
            color: #fff8ef;
            border-radius: 26px;
            padding: 1.2rem 1.25rem;
            box-shadow: var(--shadow);
            margin-bottom: 1rem;
        }

        .copilot-shell h3 {
            color: #fff8ef;
            margin-bottom: 0.35rem;
        }

        .copilot-badge {
            display: inline-block;
            margin-bottom: 0.5rem;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            background: rgba(255, 216, 159, 0.15);
            color: #ffd89f;
            font-size: 0.76rem;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            font-weight: 700;
        }

        .copilot-copy {
            color: rgba(248, 244, 236, 0.88);
            margin-bottom: 0.2rem;
        }

        .status-chip {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            margin-bottom: 0.75rem;
        }

        .status-chip.connected {
            background: rgba(14, 143, 121, 0.18);
            border: 1px solid rgba(14, 143, 121, 0.35);
            color: #b9f3e8;
        }

        .status-chip.manual {
            background: rgba(201, 133, 45, 0.16);
            border: 1px solid rgba(201, 133, 45, 0.32);
            color: #ffe3b9;
        }

        .roadmap-card {
            border-radius: 24px;
            padding: 1.15rem;
            box-shadow: var(--shadow);
            border: 1px solid rgba(19, 36, 58, 0.08);
            min-height: 260px;
            margin-bottom: 1rem;
        }

        .roadmap-now {
            background: linear-gradient(150deg, rgba(192, 84, 63, 0.14), rgba(255, 252, 246, 0.95));
        }

        .roadmap-next {
            background: linear-gradient(150deg, rgba(201, 133, 45, 0.16), rgba(255, 252, 246, 0.95));
        }

        .roadmap-later {
            background: linear-gradient(150deg, rgba(14, 143, 121, 0.14), rgba(255, 252, 246, 0.95));
        }

        .roadmap-phase {
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-weight: 700;
            color: var(--muted);
            margin-bottom: 0.3rem;
        }

        .roadmap-title {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 1.35rem;
            color: var(--ink);
            margin-bottom: 0.4rem;
        }

        .roadmap-why {
            color: var(--muted);
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
        }

        .roadmap-impact {
            display: inline-block;
            margin-top: 0.6rem;
            padding: 0.35rem 0.6rem;
            border-radius: 999px;
            background: rgba(19, 36, 58, 0.07);
            color: var(--ink);
            font-size: 0.82rem;
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .hero-title {
                font-size: 2.2rem;
            }

            .hero-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi(label: str, value: str, footnote: str) -> None:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{footnote}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_block(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="hero-pill">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_snapshot_card(label: str, title: str, body: str, stat: str) -> None:
    st.markdown(
        f"""
        <div class="snapshot-card">
            <div class="snapshot-label">{label}</div>
            <div class="snapshot-title">{title}</div>
            <div class="snapshot-copy">{body}</div>
            <div class="snapshot-stat">{stat}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert(title: str, body: str, tone: str = "opportunity") -> None:
    st.markdown(
        f"""
        <div class="alert-banner alert-{tone}">
            <strong>{title}</strong>
            <span>{body}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_roadmap_card(phase: str, title: str, why: str, actions: list[str], impact: str) -> None:
    phase_class = {
        "Fix Now": "roadmap-now",
        "Grow Next": "roadmap-next",
        "Scale Later": "roadmap-later",
    }.get(phase, "roadmap-next")
    actions_html = "".join(f"<li>{action}</li>" for action in actions[:3])
    st.markdown(
        f"""
        <div class="roadmap-card {phase_class}">
            <div class="roadmap-phase">{phase}</div>
            <div class="roadmap-title">{title}</div>
            <div class="roadmap-why">{why}</div>
            <ul>{actions_html}</ul>
            <div class="roadmap-impact">{impact}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


inject_styles()

st.markdown(
    """
    <section class="hero-shell">
        <div class="eyebrow">Retail Intelligence for Indian SMEs</div>
        <div class="hero-title">See what is selling, what is slipping, and what to restock next.</div>
        <div class="hero-copy">
            Turn a basic Excel or CSV export into weekday trend detection, margin intelligence,
            and inventory guidance that a shop owner can act on right away.
        </div>
        <div class="hero-grid">
            <div class="hero-pill"><strong>Sales Drop Detection</strong><span>Spot weak days like Tuesday slowdowns before they become a pattern.</span></div>
            <div class="hero-pill"><strong>Margin Focus</strong><span>Find the products that actually create profit, not just volume.</span></div>
            <div class="hero-pill"><strong>Restock Radar</strong><span>Catch low-cover inventory before it causes missed revenue.</span></div>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Upload Guide")
    st.markdown(
        """
        Required columns:
        - `Order Date` or `date`
        - `Sub-Category` or `product`
        - `quantity`
        - `Amount` or `revenue`

        Helpful extras:
        - `Profit`
        - `stock`

        Your current CSV format with `Amount`, `Profit`, and `Sub-Category` is supported directly.
        """
    )
    st.markdown(
        '<div class="upload-tip"><strong>Best results:</strong> one row per sale or per product-date summary.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.header("Groq Copilot")
    saved_api_key = get_secret("GROQ_API_KEY", "")
    has_saved_key = bool(saved_api_key)

    if has_saved_key:
        st.markdown(
            '<div class="status-chip connected">Groq connected via Streamlit secrets</div>',
            unsafe_allow_html=True,
        )
        manual_key_override = st.toggle(
            "Use a different Groq key for this session",
            value=False,
            help="Keep this off for normal production use. Turn it on only if you want to test with another key.",
        )
    else:
        st.markdown(
            '<div class="status-chip manual">No saved Groq key detected</div>',
            unsafe_allow_html=True,
        )
        manual_key_override = True

    api_key_input = saved_api_key
    if manual_key_override:
        api_key_input = st.text_input(
            "Groq API key",
            value="",
            type="password",
            help="Paste a Groq key for this session, or save GROQ_API_KEY in Streamlit secrets for permanent use.",
        ) or saved_api_key

    model_name = st.text_input(
        "Model",
        value=DEFAULT_MODEL,
        help="Change this if you want to try a different Groq model.",
    )
    business_context = st.text_area(
        "Business context",
        placeholder="Example: Neighborhood cafe in Pune. Want to improve weekday evening sales.",
        help="Optional context helps the AI tailor its recommendations.",
    )
    focus_prompt = st.text_area(
        "Ask the copilot to focus on",
        placeholder="Example: Improve revenue without hurting margins too much.",
        help="Optional steering prompt for the owner briefing.",
    )


example_data = pd.DataFrame(
    [
        {"date": "2026-04-01", "product": "Tea Pack", "quantity": 15, "revenue": 3000, "cost": 1800, "stock": 40},
        {"date": "2026-04-02", "product": "Tea Pack", "quantity": 10, "revenue": 2000, "cost": 1200, "stock": 30},
        {"date": "2026-04-02", "product": "Cookies", "quantity": 8, "revenue": 1600, "cost": 700, "stock": 25},
        {"date": "2026-04-08", "product": "Cookies", "quantity": 18, "revenue": 3600, "cost": 1575, "stock": 9},
        {"date": "2026-04-09", "product": "Cold Coffee", "quantity": 20, "revenue": 5000, "cost": 3000, "stock": 11},
    ]
)

intro_col, upload_col = st.columns((1.1, 1))

with intro_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
    st.subheader("Upload, analyze, act")
    st.markdown(
        """
        This MVP reads your sales file, cleans common column names, and gives you:
        - weak weekdays and strong weekdays
        - top profit-driving sub-categories
        - restocking suggestions when stock data is available
        - a simple 7-day revenue forecast
        """
    )
    sample_path = Path(__file__).with_name("sample_sales.csv")
    if sample_path.exists():
        st.caption(f"Sample file available: {sample_path.name}")
    st.markdown('</div>', unsafe_allow_html=True)

with upload_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Data Upload</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose your sales export",
        type=["xlsx", "xls", "csv"],
        help="Use one row per sale or one row per product-date combination.",
        label_visibility="visible",
    )
    st.markdown('<div class="mini-note">CSV is the easiest format for a quick demo.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with st.expander("Preview the sample input format"):
    st.dataframe(example_data, use_container_width=True, hide_index=True)

if not uploaded_file:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Waiting for your sales file")
    st.caption("Upload `sample_sales.csv` to see the full dashboard in action.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

try:
    raw_data = load_uploaded_data(uploaded_file)
    all_sales_data = prepare_sales_data(raw_data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

filter_cols = st.columns(4)
with filter_cols[0]:
    category_options = sorted(all_sales_data["category"].dropna().unique().tolist()) if "category" in all_sales_data.columns else []
    selected_categories = st.multiselect("Category", category_options)
with filter_cols[1]:
    state_options = sorted(all_sales_data["state"].dropna().unique().tolist()) if "state" in all_sales_data.columns else []
    selected_states = st.multiselect("State", state_options)
with filter_cols[2]:
    payment_options = sorted(all_sales_data["payment_mode"].dropna().unique().tolist()) if "payment_mode" in all_sales_data.columns else []
    selected_payments = st.multiselect("Payment Mode", payment_options)
with filter_cols[3]:
    min_date = all_sales_data["date"].min().date()
    max_date = all_sales_data["date"].max().date()
    selected_dates = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

sales_data = all_sales_data.copy()
if selected_categories:
    sales_data = sales_data[sales_data["category"].isin(selected_categories)]
if selected_states:
    sales_data = sales_data[sales_data["state"].isin(selected_states)]
if selected_payments:
    sales_data = sales_data[sales_data["payment_mode"].isin(selected_payments)]
if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    sales_data = sales_data[
        (sales_data["date"].dt.date >= start_date) &
        (sales_data["date"].dt.date <= end_date)
    ]

if sales_data.empty:
    st.warning("No rows match the current filters. Broaden the selection to continue.")
    st.stop()

insights = build_insights(sales_data)

date_min = sales_data["date"].min()
date_max = sales_data["date"].max()
top_category = None
if "category" in sales_data.columns and sales_data["category"].notna().any():
    category_summary = (
        sales_data.groupby("category", as_index=False)
        .agg(revenue=("revenue", "sum"), margin=("gross_margin", "sum"), quantity=("quantity", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    top_category = category_summary.iloc[0]
else:
    category_summary = pd.DataFrame()

top_state = None
if "state" in sales_data.columns and sales_data["state"].notna().any():
    state_summary = (
        sales_data.groupby("state", as_index=False)
        .agg(revenue=("revenue", "sum"), margin=("gross_margin", "sum"), orders=("revenue", "size"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    top_state = state_summary.iloc[0]
else:
    state_summary = pd.DataFrame()

top_payment = None
if "payment_mode" in sales_data.columns and sales_data["payment_mode"].notna().any():
    payment_summary = (
        sales_data.groupby("payment_mode", as_index=False)
        .agg(revenue=("revenue", "sum"), margin=("gross_margin", "sum"), orders=("revenue", "size"))
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    top_payment = payment_summary.iloc[0]
else:
    payment_summary = pd.DataFrame()

if "year_month" in sales_data.columns and sales_data["year_month"].notna().any():
    monthly_summary = (
        sales_data.groupby("year_month", as_index=False)
        .agg(revenue=("revenue", "sum"), margin=("gross_margin", "sum"))
        .sort_values("year_month")
    )
else:
    monthly_summary = (
        sales_data.assign(year_month=sales_data["date"].dt.to_period("M").astype(str))
        .groupby("year_month", as_index=False)
        .agg(revenue=("revenue", "sum"), margin=("gross_margin", "sum"))
        .sort_values("year_month")
    )

st.markdown(
    f"""
    <div class="briefing-shell">
        <div class="copilot-badge">Executive Dashboard</div>
        <h3>Sales briefing for {date_min:%d %b %Y} to {date_max:%d %b %Y}</h3>
        <p>
            This view is now shaped around your actual sales dataset, so the interface highlights revenue,
            profit, geography, and payment behavior in a way that feels closer to a founder demo than a raw CSV reader.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Performance Snapshot</div>', unsafe_allow_html=True)
kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4)
with kpi_1:
    render_kpi("Total Revenue", format_currency(insights.summary["total_revenue"]), "Gross sales captured in the uploaded file")
with kpi_2:
    render_kpi("Total Margin", format_currency(insights.summary["total_margin"]), "Revenue minus product cost where available")
with kpi_3:
    render_kpi("Rows Processed", f"{insights.summary['total_orders']:,}", "Transactions or daily product rows analyzed")
with kpi_4:
    render_kpi("7-Day Forecast", format_currency(insights.summary["predicted_revenue_7d"]), "Simple linear projection from recent trend")

alert_left, alert_right = st.columns(2)
with alert_left:
    weakest_day = insights.weekday_performance.loc[insights.weekday_performance["avg_revenue"].idxmin()]
    render_alert(
        "Risk Signal",
        f"{weakest_day['weekday']} is your weakest sales day in the current filtered view. Build campaigns or bundles around that dip first.",
        tone="risk",
    )
with alert_right:
    top_product = insights.product_performance.iloc[0]
    render_alert(
        "Growth Opportunity",
        f"{top_product['product']} is leading profit contribution right now. Double down on visibility, bundles, or upsell strategy around it.",
        tone="opportunity",
    )

focus_1, focus_2, focus_3 = st.columns(3)
with focus_1:
    if top_category is not None:
        render_snapshot_card(
            "Top Category",
            str(top_category["category"]),
            "This category is currently driving the largest share of sales volume in the uploaded dataset.",
            f"Revenue {format_currency(float(top_category['revenue']))}",
        )
    else:
        render_snapshot_card(
            "Top Product",
            str(insights.product_performance.iloc[0]['product']),
            "This sub-category is contributing the strongest combined commercial performance right now.",
            f"Margin {format_currency(float(insights.product_performance.iloc[0]['total_margin']))}",
        )
with focus_2:
    if top_state is not None:
        render_snapshot_card(
            "Top State",
            str(top_state["state"]),
            "Geographic concentration matters for targeting campaigns and inventory planning.",
            f"Revenue {format_currency(float(top_state['revenue']))}",
        )
    else:
        render_snapshot_card(
            "High Margin Leader",
            str(insights.product_performance.iloc[0]["product"]),
            "This product currently contributes the most profit across the dataset.",
            f"Margin {format_currency(float(insights.product_performance.iloc[0]['total_margin']))}",
        )
with focus_3:
    if top_payment is not None:
        render_snapshot_card(
            "Top Payment Mode",
            str(top_payment["payment_mode"]),
            "Useful for checkout optimization and understanding which channels customers prefer.",
            f"Orders {format_number(float(top_payment['orders']))}",
        )
    else:
        render_snapshot_card(
            "Rows Analyzed",
            "Dataset Loaded",
            "The uploaded sales export has been normalized and is ready for owner-facing insights.",
            f"{format_number(insights.summary['total_orders'])} rows",
        )

st.markdown(
    """
    <div class="copilot-shell">
        <div class="copilot-badge">LLM Layer</div>
        <h3>Groq Owner Copilot</h3>
        <div class="copilot-copy">
            Generate a plain-English owner briefing from the sales analytics below. This gives you
            a sharper startup demo and a more useful summary for non-technical business users.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

copilot_col, config_col = st.columns((1.35, 1))
with copilot_col:
    trigger_brief = st.button("Generate Groq Owner Briefing", use_container_width=True)
with config_col:
    if api_key_input:
        if has_saved_key and not manual_key_override:
            st.caption(f"Groq connected from secrets using model `{model_name.strip() or DEFAULT_MODEL}`")
        else:
            st.caption(f"Groq ready with model `{model_name.strip() or DEFAULT_MODEL}`")
    else:
        st.caption("Add a Groq API key in the sidebar to enable the Groq briefing.")

roadmap_trigger = st.button("Build AI Action Roadmap", use_container_width=True)

if trigger_brief:
    if not api_key_input:
        st.warning("Add a Groq API key in the sidebar first, then generate the Groq briefing.")
    else:
        payload = build_copilot_payload(
            summary=insights.summary,
            insight_cards=insights.insight_cards,
            weekday_performance=insights.weekday_performance,
            product_performance=insights.product_performance,
            restock_table=insights.restock_table,
            business_context=business_context,
        )
        with st.spinner("Writing your owner briefing..."):
            try:
                st.session_state["owner_brief"] = generate_owner_brief(
                    api_key=api_key_input,
                    payload=payload,
                    model=model_name.strip() or DEFAULT_MODEL,
                    focus_prompt=focus_prompt,
                )
            except Exception as exc:
                st.error(f"AI briefing failed: {exc}")

if roadmap_trigger:
    if not api_key_input:
        st.warning("Add a Groq API key in the sidebar first, then build the roadmap.")
    else:
        payload = build_copilot_payload(
            summary=insights.summary,
            insight_cards=insights.insight_cards,
            weekday_performance=insights.weekday_performance,
            product_performance=insights.product_performance,
            restock_table=insights.restock_table,
            business_context=business_context,
        )
        with st.spinner("Building your action roadmap..."):
            try:
                st.session_state["action_roadmap"] = generate_action_roadmap(
                    api_key=api_key_input,
                    payload=payload,
                    model=model_name.strip() or DEFAULT_MODEL,
                    focus_prompt=focus_prompt,
                )
            except Exception as exc:
                st.error(f"AI roadmap failed: {exc}")

if st.session_state.get("owner_brief"):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Owner Briefing</div>', unsafe_allow_html=True)
    st.subheader("What the Groq sales copilot recommends")
    st.markdown(st.session_state["owner_brief"])
    st.markdown('</div>', unsafe_allow_html=True)

roadmap = st.session_state.get("action_roadmap")
if roadmap:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">AI Roadmap</div>', unsafe_allow_html=True)
    st.subheader(roadmap.get("headline", "Step-by-step growth roadmap"))
    st.caption(roadmap.get("priority", ""))
    roadmap_cols = st.columns(3)
    for col, step in zip(roadmap_cols, roadmap.get("steps", [])[:3]):
        with col:
            render_roadmap_card(
                phase=step.get("phase", "Grow Next"),
                title=step.get("title", "Action"),
                why=step.get("why", ""),
                actions=step.get("actions", []),
                impact=step.get("impact", ""),
            )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">AI Briefing</div>', unsafe_allow_html=True)
brief_left, brief_right = st.columns((1.25, 1))

with brief_left:
    st.subheader("What the owner should notice today")
    for index, card in enumerate(insights.insight_cards, start=1):
        st.markdown(
            f"""
            <div class="insight-card">
                <strong>Insight {index}</strong>
                <span>{card}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

with brief_right:
    render_info_block(
        "Best for demos",
        "This layout is designed to feel pitch-ready when you show it to a shop owner or small retail team.",
    )
    render_info_block(
        "Works with messy exports",
        "The app accepts common column aliases so real-world Excel files need less cleanup.",
    )
    render_info_block(
        "LLM enabled",
        "Add a Groq key in the sidebar and the app can turn the dashboard into an owner-friendly action brief.",
    )
st.markdown('</div>', unsafe_allow_html=True)

overview_tab, market_tab, data_tab = st.tabs(["Overview", "Market View", "Data Room"])

with overview_tab:
    left_col, right_col = st.columns((1.35, 1))

    with left_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Revenue Momentum</div>', unsafe_allow_html=True)
        st.subheader("Daily Revenue Trend")
        trend_chart = insights.daily_trend.set_index("date")["revenue"]
        st.line_chart(trend_chart, use_container_width=True, color="#0e8f79")
        st.caption("Use this to spot sharp dips, recovery weeks, and whether demand is flattening out.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Month by Month</div>', unsafe_allow_html=True)
        st.subheader("Monthly Revenue Arc")
        month_chart = monthly_summary.set_index("year_month")["revenue"]
        st.area_chart(month_chart, use_container_width=True, color="#c0543f")
        st.caption("This gives the founder-style view: where growth compounds, where seasonality bites, and where campaigns may have worked.")
        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Weekday Behavior</div>', unsafe_allow_html=True)
        st.subheader("Average Revenue by Day")
        weekday_chart = insights.weekday_performance.set_index("weekday")["avg_revenue"]
        st.bar_chart(weekday_chart, use_container_width=True, color="#c9852d")
        st.caption("This is the fastest way to verify claims like sales always falling on Tuesdays.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Profit Leaders</div>', unsafe_allow_html=True)
        st.subheader("Top Products by Margin")
        product_view = insights.product_performance.head(10).copy()
        product_view["total_revenue"] = product_view["total_revenue"].map(format_currency)
        product_view["total_margin"] = product_view["total_margin"].map(format_currency)
        product_view["avg_margin_pct"] = product_view["avg_margin_pct"].map(lambda value: f"{value:.1f}%")
        st.dataframe(product_view, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

with market_tab:
    market_left, market_right = st.columns((1, 1))

    with market_left:
        if not category_summary.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Category Mix</div>', unsafe_allow_html=True)
            st.subheader("Revenue by Category")
            st.bar_chart(category_summary.set_index("category")["revenue"], use_container_width=True, color="#0e8f79")
            category_view = category_summary.copy()
            category_view["revenue"] = category_view["revenue"].map(format_currency)
            category_view["margin"] = category_view["margin"].map(format_currency)
            category_view["quantity"] = category_view["quantity"].map(format_number)
            st.dataframe(category_view, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        if not payment_summary.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Checkout Behavior</div>', unsafe_allow_html=True)
            st.subheader("Payment Modes")
            st.bar_chart(payment_summary.set_index("payment_mode")["revenue"], use_container_width=True, color="#c9852d")
            payment_view = payment_summary.copy()
            payment_view["revenue"] = payment_view["revenue"].map(format_currency)
            payment_view["margin"] = payment_view["margin"].map(format_currency)
            payment_view["orders"] = payment_view["orders"].map(format_number)
            st.dataframe(payment_view, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with market_right:
        if not state_summary.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-label">Geography</div>', unsafe_allow_html=True)
            st.subheader("Top States by Revenue")
            st.bar_chart(state_summary.head(10).set_index("state")["revenue"], use_container_width=True, color="#c0543f")
            state_view = state_summary.head(10).copy()
            state_view["revenue"] = state_view["revenue"].map(format_currency)
            state_view["margin"] = state_view["margin"].map(format_currency)
            state_view["orders"] = state_view["orders"].map(format_number)
            st.dataframe(state_view, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Inventory Watch</div>', unsafe_allow_html=True)
        st.subheader("Restocking Recommendations")
        restock_view = insights.restock_table.copy()
        if "days_of_cover" in restock_view.columns:
            restock_view["days_of_cover"] = restock_view["days_of_cover"].map(
                lambda value: "-" if pd.isna(value) else f"{value:.1f}"
            )
        if "avg_daily_demand" in restock_view.columns:
            restock_view["avg_daily_demand"] = restock_view["avg_daily_demand"].map(lambda value: f"{value:.1f}")
        st.dataframe(restock_view, use_container_width=True, hide_index=True)
        st.caption("This section becomes much more actionable when your source file includes a real stock column.")
        st.markdown('</div>', unsafe_allow_html=True)

with data_tab:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Data Confidence</div>', unsafe_allow_html=True)
    st.subheader("Cleaned Data Preview")
    st.dataframe(sales_data.head(50), use_container_width=True, hide_index=True)
    st.caption("Review this preview if a result looks surprising. It shows the columns after cleanup and normalization.")
    st.markdown('</div>', unsafe_allow_html=True)
