from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ai_copilot import DEFAULT_MODEL, build_copilot_payload, generate_owner_brief
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
        - `date`
        - `product`
        - `quantity`
        - `revenue`

        Helpful extras:
        - `cost`
        - `stock`

        Aliases like `qty`, `sales`, `item`, and `inventory` are supported.
        """
    )
    st.markdown(
        '<div class="upload-tip"><strong>Best results:</strong> one row per sale or per product-date summary.</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.header("AI Copilot")
    saved_api_key = st.secrets.get("OPENAI_API_KEY", "")
    api_key_input = st.text_input(
        "OpenAI API key",
        value=saved_api_key,
        type="password",
        help="Optional. Add your key here or save it as OPENAI_API_KEY in Streamlit secrets.",
    )
    model_name = st.text_input(
        "Model",
        value=DEFAULT_MODEL,
        help="Change this if you want to try a different OpenAI text model.",
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
        - top margin products
        - recent-demand-based restocking suggestions
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
    sales_data = prepare_sales_data(raw_data)
    insights = build_insights(sales_data)
except Exception as exc:
    st.error(str(exc))
    st.stop()

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

st.markdown(
    """
    <div class="copilot-shell">
        <div class="copilot-badge">LLM Layer</div>
        <h3>AI Owner Copilot</h3>
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
    trigger_brief = st.button("Generate AI Owner Briefing", use_container_width=True)
with config_col:
    if api_key_input:
        st.caption(f"AI ready with model `{model_name.strip() or DEFAULT_MODEL}`")
    else:
        st.caption("Add an OpenAI API key in the sidebar to enable the AI briefing.")

if trigger_brief:
    if not api_key_input:
        st.warning("Add an OpenAI API key in the sidebar first, then generate the briefing.")
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

if st.session_state.get("owner_brief"):
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Owner Briefing</div>', unsafe_allow_html=True)
    st.subheader("What the AI sales copilot recommends")
    st.markdown(st.session_state["owner_brief"])
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
        "Add an OpenAI key in the sidebar and the app can turn the dashboard into an owner-friendly action brief.",
    )
st.markdown('</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-label">Weekday Behavior</div>', unsafe_allow_html=True)
    st.subheader("Average Revenue by Day")
    weekday_chart = insights.weekday_performance.set_index("weekday")["avg_revenue"]
    st.bar_chart(weekday_chart, use_container_width=True, color="#c9852d")
    st.caption("This is the fastest way to verify claims like sales always falling on Tuesdays.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Profit Leaders</div>', unsafe_allow_html=True)
    st.subheader("Top Products by Margin")
    product_view = insights.product_performance.head(10).copy()
    product_view["total_revenue"] = product_view["total_revenue"].map(format_currency)
    product_view["total_margin"] = product_view["total_margin"].map(format_currency)
    product_view["avg_margin_pct"] = product_view["avg_margin_pct"].map(lambda value: f"{value:.1f}%")
    st.dataframe(product_view, use_container_width=True, hide_index=True)
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
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-label">Data Confidence</div>', unsafe_allow_html=True)
st.subheader("Cleaned Data Preview")
st.dataframe(sales_data.head(50), use_container_width=True, hide_index=True)
st.caption("Review this preview if a result looks surprising. It shows the columns after cleanup and normalization.")
st.markdown('</div>', unsafe_allow_html=True)
