from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel


# ============================================================
# PAGE + PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "outputs" / "cleaned_data"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models" / "random_forest"

st.set_page_config(
    page_title="AutoMPG | Fuel Efficiency Analytics",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# VISUAL THEME
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --canvas: #0a1020;
        --surface: #111b2e;
        --surface-raised: #16243a;
        --line: #263750;
        --ink: #e8f0fb;
        --muted: #9aacc4;
        --blue: #60a5fa;
        --teal: #2dd4bf;
    }
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--canvas) !important;
        color: var(--ink) !important;
    }
    [data-testid="stHeader"] { background: rgba(10,16,32,.92) !important; }
    [data-testid="stToolbar"] { right: 1rem; }
    .block-container { max-width: 1480px; padding: 1.35rem 2.1rem 3rem; }
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stBottom"] { background: transparent !important; }
    p, li, label, span, div { text-rendering: optimizeLegibility; }
    .hero {
        padding: 1.8rem 2rem; border-radius: 22px; margin: .35rem 0 1.15rem;
        background: radial-gradient(circle at 86% 18%, rgba(45,212,191,.25), transparent 30%),
                    linear-gradient(118deg, #142746 0%, #173b66 54%, #075b68 100%);
        border: 1px solid rgba(125,180,230,.2);
        box-shadow: 0 18px 45px rgba(0,0,0,.24);
    }
    .hero-kicker { color:#9de9e2; font-size:.70rem; font-weight:850; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.55rem; }
    .hero h1 { color:#f8fbff !important; font-size:clamp(2rem,3.2vw,2.7rem); line-height:1.12; margin:0; font-weight:850; letter-spacing:-.035em; }
    .hero p { color:#d6e7f8 !important; margin:.7rem 0 0; font-size:.98rem; }
    .brand { color:#f0f6ff; font-size:1.08rem; font-weight:850; letter-spacing:-.02em; }
    .brand small { display:block; color:#91a7c2; font-size:.72rem; font-weight:650; letter-spacing:.07em; margin-top:.16rem; }
    .section-heading { font-size:1.35rem; font-weight:850; color:#edf5ff; margin:.35rem 0 .2rem; letter-spacing:-.02em; }
    .section-note { color:#9aacc4; font-size:.91rem; margin-bottom:1.05rem; }
    h1, h2, h3, h4, h5, h6, [data-testid="stMarkdownContainer"] strong { color:#edf5ff !important; }
    [data-testid="stMarkdownContainer"] p, [data-testid="stCaptionContainer"] { color:#a9bad0 !important; }
    [data-testid="stCaptionContainer"] { font-size:.82rem; }
    hr, [data-testid="stDivider"] { border-color:#263750 !important; }
    .info-panel {
        background:linear-gradient(145deg,#14233a,#111c2e); border:1px solid #2a405d;
        border-left:4px solid #38bdf8; border-radius:15px; padding:1.05rem 1.2rem;
        color:#cbd9ea !important; margin:.4rem 0 1rem; line-height:1.7;
        box-shadow:0 8px 22px rgba(0,0,0,.16);
    }
    .info-panel b { color:#f0f6ff; }
    .small-muted { color:#9aacc4 !important; font-size:.83rem; }
    .filter-panel-title { color:#edf5ff; font-size:1.02rem; font-weight:850; margin-bottom:.12rem; }
    .filter-panel-subtitle { color:#9aacc4; font-size:.82rem; margin-bottom:.65rem; }
    /* Dark, consistent filters and form controls */
    [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid #2a3d58 !important; border-radius:17px !important; background:#101a2c !important; }
    [data-testid="stSelectbox"] label, [data-testid="stMultiSelect"] label,
    [data-testid="stSlider"] label, [data-testid="stTextInput"] label,
    [data-testid="stNumberInput"] label, [data-testid="stRadio"] label {
        color:#cbd9ea !important; font-weight:700 !important; font-size:.82rem !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
        min-height:43px !important; background:#17263d !important; border:1px solid #344b69 !important;
        border-radius:11px !important; color:#edf5ff !important; box-shadow:none !important;
    }
    [data-testid="stSelectbox"] [data-baseweb="select"] *,
    [data-testid="stMultiSelect"] [data-baseweb="select"] *,
    [data-testid="stSelectbox"] input, [data-testid="stMultiSelect"] input { color:#edf5ff !important; }
    [data-baseweb="popover"] [role="listbox"], [data-baseweb="menu"] {
        background:#17263d !important; border:1px solid #344b69 !important;
    }
    [data-baseweb="popover"] [role="option"], [data-baseweb="menu"] li { color:#e6effb !important; }
    [data-baseweb="popover"] [role="option"]:hover, [data-baseweb="menu"] li:hover { background:#263c5b !important; }
    [data-testid="stSlider"] [data-baseweb="slider"] { color:#38bdf8 !important; }
    [data-testid="stSlider"] [data-testid="stTickBar"] { background:#425875 !important; }
    [data-testid="stSlider"] div[data-testid="stMarkdownContainer"] { color:#9fb2ca !important; }
    .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
        border-radius:11px !important; font-weight:750 !important; min-height:2.65rem;
        transition:all .18s ease;
    }
    div[data-testid="stButton"] button, .stDownloadButton > button {
        background:#17263d !important; color:#e8f2ff !important; border:1px solid #3a5272 !important;
        box-shadow:0 4px 12px rgba(0,0,0,.16) !important; opacity:1 !important;
    }
    div[data-testid="stButton"] button:hover, .stDownloadButton > button:hover {
        background:#203653 !important; border-color:#60a5fa !important; color:#fff !important;
        transform:translateY(-1px);
    }
    div[data-testid="stButton"] button p, .stDownloadButton button p { color:#e8f2ff !important; opacity:1 !important; }
    .stFormSubmitButton > button, div[data-testid="stButton"] button[kind="primary"] {
        background:linear-gradient(100deg,#2563eb,#0d9488) !important; border:0 !important; color:#fff !important;
    }
    .filter-reset button { background:#17263d !important; color:#e8f2ff !important; border:1px solid #405979 !important; border-radius:11px !important; min-height:43px !important; font-weight:750 !important; }
    .filter-reset button:hover { background:#203653 !important; border-color:#60a5fa !important; }
    div[role="radiogroup"] { gap:.45rem; }
    div[role="radiogroup"] label { background:#17263d !important; border:1px solid #344b69 !important; border-radius:11px !important; padding:.48rem .9rem; color:#cbd9ea !important; }
    div[role="radiogroup"] label:has(input:checked) { background:#173b5b !important; border-color:#38bdf8 !important; color:#eaf6ff !important; }
    [data-testid="stExpander"] { background:#111c2e !important; border:1px solid #2a3d58 !important; border-radius:13px !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary p { color:#e6effb !important; }
    /* Premium dark KPI cards */
    .kpi-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin:.55rem 0 1.35rem; }
    .kpi-card { position:relative; overflow:hidden; min-height:139px; background:linear-gradient(145deg,#17263d,#111c2e);
        border:1px solid #2b405d; border-radius:17px; padding:19px 20px 17px;
        box-shadow:0 10px 25px rgba(0,0,0,.2); }
    .kpi-card:before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:linear-gradient(180deg,#60a5fa,#2dd4bf); }
    .kpi-card:after { content:""; position:absolute; width:110px; height:110px; right:-48px; top:-55px; border-radius:50%; background:rgba(56,189,248,.07); }
    .kpi-top { display:flex; align-items:center; justify-content:space-between; gap:10px; }
    .kpi-label { color:#a9bdd5 !important; font-size:.79rem; line-height:1.35; font-weight:750; letter-spacing:.025em; }
    .kpi-icon { display:inline-flex; align-items:center; justify-content:center; width:34px; height:34px; border-radius:11px;
        background:#203b5d; color:#7dd3fc; font-size:1rem; font-weight:850; flex-shrink:0; }
    .kpi-value { color:#f4f8ff !important; font-size:1.95rem; line-height:1.15; font-weight:900; letter-spacing:-.045em; margin-top:14px; }
    .kpi-help { color:#8fa5bf !important; font-size:.73rem; margin-top:7px; line-height:1.35; }
    .kpi-grid.three { grid-template-columns:repeat(3,minmax(0,1fr)); }
    /* Plotly chart containers */
    div[data-testid="stPlotlyChart"] { background:#111c2e !important; border:1px solid #293e5a !important; border-radius:17px !important;
        padding:.45rem; box-shadow:0 10px 25px rgba(0,0,0,.17); }
    div[data-testid="stDataFrame"] { border:1px solid #2a3d58 !important; border-radius:13px; overflow:hidden; }
    [data-testid="stAlert"] { border-radius:13px !important; }
    @media (max-width:1000px) { .kpi-grid { grid-template-columns:repeat(2,minmax(0,1fr)); } }
    @media (max-width:560px) { .kpi-grid, .kpi-grid.three { grid-template-columns:1fr; } }
    @media (max-width:800px) { .block-container { padding-left:1rem; padding-right:1rem; } .hero { padding:1.25rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA LOADERS
# ============================================================
@st.cache_data(show_spinner="Loading EPA vehicle dataset...")
def load_dataset():
    files = sorted(DATA_DIR.glob("part-*.csv"))
    if not files:
        raise FileNotFoundError(
            f"No cleaned CSV files found in {DATA_DIR}. Run src/preprocess_data.py first."
        )
    return pd.concat([pd.read_csv(file) for file in files], ignore_index=True)


@st.cache_data(show_spinner="Loading test predictions...")
def load_predictions():
    path = OUTPUT_DIR / "test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(f"Test predictions not found: {path}")
    return pd.read_csv(path)


@st.cache_data
def load_metrics():
    path = OUTPUT_DIR / "model_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Model metrics not found: {path}")
    with path.open() as file:
        return json.load(file)


try:
    data = load_dataset()
    predictions = load_predictions()
    results = load_metrics()
except Exception as exc:
    st.error(f"Unable to load project files: {exc}")
    st.stop()

required_columns = {
    "engine_displacement", "no_cylinders", "no_gears", "transmission_desc",
    "drive_desc", "class", "fuel_usage_desc", "comb_mpg"
}
missing = required_columns - set(data.columns)
if missing:
    st.error(f"Cleaned dataset is missing required columns: {', '.join(sorted(missing))}")
    st.stop()

# ============================================================
# TOP NAVIGATION + CLEAN FILTER PANEL (NO SIDEBAR)
# ============================================================
vehicle_options = sorted(data["class"].dropna().unique().tolist())
fuel_options = sorted(data["fuel_usage_desc"].dropna().unique().tolist())
drive_options = sorted(data["drive_desc"].dropna().unique().tolist())
engine_min = float(data["engine_displacement"].min())
engine_max = float(data["engine_displacement"].max())

# Single-select slicers keep the filter area compact and easy to use.
class_choices = ["All vehicle classes"] + vehicle_options
fuel_choices = ["All fuel types"] + fuel_options
drive_choices = ["All drive types"] + drive_options
for key, default in {
    "filter_class": "All vehicle classes",
    "filter_fuel": "All fuel types",
    "filter_drive": "All drive types",
    "filter_engine": (engine_min, engine_max),
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Project title is intentionally placed before the filters so the purpose of the
# dashboard is immediately clear when it opens.
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">Cluster Computing · INT315 · Question 18</div>
      <h1>Automobile Fuel Efficiency Prediction System</h1>
      <p>Explore vehicle fuel economy, compare regression models and estimate combined MPG using EPA 2021 data.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    filter_title_col, reset_col = st.columns([5.5, 1.15], vertical_alignment="center")
    with filter_title_col:
        st.markdown('<div class="filter-panel-title">Filter vehicles</div><div class="filter-panel-subtitle">Choose a category or narrow the engine-size range. All filters start at the full dataset.</div>', unsafe_allow_html=True)
    with reset_col:
        st.markdown('<div class="filter-reset">', unsafe_allow_html=True)
        reset_clicked = st.button("↺  Reset filters", use_container_width=True, type="secondary")
        st.markdown('</div>', unsafe_allow_html=True)

    if reset_clicked:
        st.session_state["filter_class"] = "All vehicle classes"
        st.session_state["filter_fuel"] = "All fuel types"
        st.session_state["filter_drive"] = "All drive types"
        st.session_state["filter_engine"] = (engine_min, engine_max)
        st.rerun()

    f1, f2, f3, f4 = st.columns([1.15, 1.15, 1.05, 1.55], gap="medium", vertical_alignment="bottom")
    with f1:
        st.selectbox("Vehicle class", class_choices, key="filter_class")
    with f2:
        st.selectbox("Fuel type", fuel_choices, key="filter_fuel")
    with f3:
        st.selectbox("Drive type", drive_choices, key="filter_drive")
    with f4:
        st.slider("Engine displacement (L)", min_value=engine_min, max_value=engine_max,
                  key="filter_engine", step=0.1)

page = st.radio(
    "Dashboard navigation",
    ["Overview", "Vehicle Analysis", "Model Performance", "Fuel Efficiency Prediction", "Dataset Explorer"],
    horizontal=True,
    label_visibility="collapsed",
    key="dashboard_page",
)
st.markdown("<div style='height:.25rem'></div>", unsafe_allow_html=True)

selected_class = st.session_state["filter_class"]
selected_fuel = st.session_state["filter_fuel"]
selected_drive = st.session_state["filter_drive"]
engine_range = st.session_state["filter_engine"]
filtered_data = data[
    ((data["class"] == selected_class) if selected_class != "All vehicle classes" else True)
    & ((data["fuel_usage_desc"] == selected_fuel) if selected_fuel != "All fuel types" else True)
    & ((data["drive_desc"] == selected_drive) if selected_drive != "All drive types" else True)
    & data["engine_displacement"].between(engine_range[0], engine_range[1])
].copy()

prediction_required = {
    "engine_displacement", "no_cylinders", "no_gears", "transmission_desc",
    "drive_desc", "class", "fuel_usage_desc", "comb_mpg", "prediction"
}
if prediction_required.issubset(predictions.columns):
    filtered_predictions = predictions[
        ((predictions["class"] == selected_class) if selected_class != "All vehicle classes" else True)
        & ((predictions["fuel_usage_desc"] == selected_fuel) if selected_fuel != "All fuel types" else True)
        & ((predictions["drive_desc"] == selected_drive) if selected_drive != "All drive types" else True)
        & predictions["engine_displacement"].between(engine_range[0], engine_range[1])
    ].copy()
else:
    filtered_predictions = pd.DataFrame()

# Shared Plotly layout
PLOT_TEMPLATE = "plotly_dark"
PLOT_BG = "#111c2e"
GRID_COLOR = "#293e58"


def polish_chart(fig, height=390):
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(family="Inter, Arial, sans-serif", color="#dbe7f5", size=13),
        title=dict(font=dict(size=16, color="#f0f6ff"), x=0.02, xanchor="left"),
        margin=dict(l=22, r=22, t=62, b=42),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    font=dict(color="#c5d5e8", size=11)),
        hoverlabel=dict(bgcolor="#203653", bordercolor="#45617f", font_color="#ffffff"),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                     tickfont=dict(color="#a9bdd2", size=11),
                     title_font=dict(color="#d6e4f4", size=12))
    fig.update_yaxes(gridcolor=GRID_COLOR, zerolinecolor=GRID_COLOR,
                     tickfont=dict(color="#a9bdd2", size=11),
                     title_font=dict(color="#d6e4f4", size=12))
    return fig


def page_heading(title, description):
    st.markdown(f'<div class="section-heading">{title}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-note">{description}</div>', unsafe_allow_html=True)


def render_kpis(items, columns=4):
    """Render consistent, high-contrast KPI cards without Streamlit's default metric styling."""
    grid_class = "kpi-grid three" if columns == 3 else "kpi-grid"
    cards = []
    for label, value, icon, note in items:
        cards.append(
            '<div class="kpi-card">'
            '<div class="kpi-top">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-icon">{icon}</div>'
            '</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-help">{note}</div>'
            '</div>'
        )
    st.markdown(f'<div class="{grid_class}">{"".join(cards)}</div>', unsafe_allow_html=True)


@st.cache_resource
def get_spark_session():
    """Reuse the project's local Spark session for interactive predictions."""
    return (
        SparkSession.builder
        .appName("AutomobileFuelEfficiencyDashboard")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.hadoop.fs.defaultFS", "file:///")
        .getOrCreate()
    )


@st.cache_resource
def get_rf_model():
    """Load the already-trained Random Forest pipeline; never retrain here."""
    return PipelineModel.load(str(MODEL_DIR))


def predict_vehicle(engine, cylinders, gears, transmission, drive, vehicle_class, fuel):
    """Run one row through the saved PySpark pipeline and return predicted MPG."""
    spark = get_spark_session()
    model = get_rf_model()
    columns = [
        "engine_displacement", "no_cylinders", "no_gears",
        "transmission_desc", "drive_desc", "class", "fuel_usage_desc",
    ]
    row = [(float(engine), int(cylinders), int(gears), str(transmission),
            str(drive), str(vehicle_class), str(fuel))]
    input_df = spark.createDataFrame(row, columns)
    return float(model.transform(input_df).select("prediction").first()[0])


def record_prediction(source, engine, cylinders, gears, transmission, drive,
                      vehicle_class, fuel, predicted_mpg):
    """Keep prediction history for this active Streamlit session."""
    st.session_state.setdefault("prediction_history", [])
    st.session_state["prediction_history"].insert(0, {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Source": source,
        "Engine displacement (L)": float(engine),
        "Cylinders": int(cylinders),
        "Gears": int(gears),
        "Transmission": str(transmission),
        "Drive type": str(drive),
        "Vehicle class": str(vehicle_class),
        "Fuel type": str(fuel),
        "Predicted MPG": round(float(predicted_mpg), 3),
    })
    # Keep the latest 100 predictions to avoid unbounded session growth.
    st.session_state["prediction_history"] = st.session_state["prediction_history"][:100]


def add_automatic_insights(frame, complete_data):
    """Show descriptive, filter-aware observations without implying causation."""
    if frame.empty:
        st.info("Choose different filters to generate insights.")
        return
    avg_mpg = float(frame["comb_mpg"].mean())
    overall_mpg = float(complete_data["comb_mpg"].mean())
    delta = avg_mpg - overall_mpg
    if delta > 0.05:
        comparison = f"{delta:.2f} MPG above the full-dataset average"
    elif delta < -0.05:
        comparison = f"{abs(delta):.2f} MPG below the full-dataset average"
    else:
        comparison = "approximately equal to the full-dataset average"
    class_avg = frame.groupby("class")["comb_mpg"].mean().sort_values(ascending=False)
    top_class = str(class_avg.index[0]) if not class_avg.empty else "Not available"
    top_class_mpg = float(class_avg.iloc[0]) if not class_avg.empty else float("nan")
    fuel_counts = frame["fuel_usage_desc"].value_counts()
    common_fuel = str(fuel_counts.index[0]) if not fuel_counts.empty else "Not available"
    correlation = frame[["engine_displacement", "comb_mpg"]].corr().iloc[0, 1] if len(frame) > 1 else float("nan")
    insight_items = [
        f"The current filters include **{len(frame):,} vehicles** with an average combined efficiency of **{avg_mpg:.2f} MPG**, {comparison}.",
        f"**{top_class}** has the highest average MPG among the selected vehicle classes (**{top_class_mpg:.2f} MPG**).",
        f"**{common_fuel}** is the most frequent fuel category in the current selection ({int(fuel_counts.iloc[0]):,} records).",
    ]
    if pd.notna(correlation):
        strength = "positive" if correlation > 0.1 else "negative" if correlation < -0.1 else "weak"
        insight_items.append(
            f"Engine displacement and combined MPG have a **{strength} descriptive correlation** in this selection (r = {correlation:.2f}); this is an association, not proof of causation."
        )
    with st.container(border=True):
        st.markdown("#### Automatic insights")
        for item in insight_items:
            st.markdown(f"- {item}")


def get_feature_importance(model, sample_data):
    """Read impurity-based importances and feature labels from the saved Spark pipeline metadata."""
    rf_stage = next((stage for stage in reversed(model.stages)
                     if hasattr(stage, "featureImportances")), None)
    if rf_stage is None:
        raise RuntimeError("No Random Forest stage with featureImportances was found in the saved pipeline.")
    importances = np.asarray(rf_stage.featureImportances.toArray(), dtype=float)
    feature_columns = [
        "engine_displacement", "no_cylinders", "no_gears",
        "transmission_desc", "drive_desc", "class", "fuel_usage_desc",
    ]
    spark = get_spark_session()
    first = sample_data.iloc[0]
    row = [(float(first["engine_displacement"]), int(first["no_cylinders"]),
            int(first["no_gears"]), str(first["transmission_desc"]),
            str(first["drive_desc"]), str(first["class"]), str(first["fuel_usage_desc"]))]
    sample_spark = spark.createDataFrame(row, feature_columns)
    transformed = model.transform(sample_spark)
    metadata = transformed.schema["features"].metadata.get("ml_attr", {})
    attrs = metadata.get("attrs", {})
    names_by_index = {}
    for group in attrs.values():
        for attribute in group:
            names_by_index[int(attribute["idx"])] = str(attribute.get("name", f"Feature {int(attribute['idx']) + 1}"))
    names = [names_by_index.get(i, f"Encoded feature {i + 1}") for i in range(len(importances))]
    result = pd.DataFrame({"Feature": names, "Importance": importances})
    return result.sort_values("Importance", ascending=False).head(20)


# ============================================================
# OVERVIEW
# ============================================================
if page == "Overview":
    page_heading("Project overview", "A live summary of the vehicles currently selected using the filters above.")

    if filtered_data.empty:
        st.warning("No vehicles match the selected filters. Adjust the filter selections above.")
        st.stop()

    selected_avg_mpg = float(filtered_data["comb_mpg"].mean())
    full_avg_mpg = float(data["comb_mpg"].mean())
    mpg_delta = selected_avg_mpg - full_avg_mpg
    mpg_note = f"{mpg_delta:+.2f} MPG vs full dataset"
    selected_engine = float(filtered_data["engine_displacement"].mean())
    full_engine = float(data["engine_displacement"].mean())
    engine_note = f"{selected_engine - full_engine:+.2f} L vs full dataset"
    render_kpis([
        ("Vehicles selected", f"{len(filtered_data):,}", "▦", f"{len(filtered_data) / len(data) * 100:.1f}% of all records"),
        ("Average combined MPG", f"{selected_avg_mpg:.2f}", "↗", mpg_note),
        ("Average engine size", f"{selected_engine:.2f} L", "⚙", engine_note),
        ("Highest MPG", f'{filtered_data["comb_mpg"].max():.2f}', "★", "Best observed record in this selection"),
    ])

    st.divider()
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Fuel efficiency distribution")
        fig = px.histogram(
            filtered_data, x="comb_mpg", nbins=20,
            labels={"comb_mpg": "Combined fuel efficiency (MPG)", "count": "Number of vehicles"},
            color_discrete_sequence=["#3182ce"],
        )
        fig.update_layout(bargap=0.12, showlegend=False)
        fig.update_traces(hovertemplate="MPG: %{x}<br>Vehicles: %{y}<extra></extra>")
        polish_chart(fig, 360)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption("A histogram showing how vehicle fuel-efficiency values are distributed.")

    with right:
        st.markdown("#### Vehicle distribution by fuel type")
        fuel_counts = (filtered_data["fuel_usage_desc"].value_counts()
                       .rename_axis("Fuel type").reset_index(name="Vehicles"))
        fig = px.pie(
            fuel_counts, names="Fuel type", values="Vehicles", hole=0.38,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        # Keep small fuel categories readable: show percentages on the donut
        # and put full fuel names in a legend instead of squeezing labels inside.
        fig.update_traces(textposition="inside", textinfo="percent",
                          insidetextorientation="horizontal",
                          hovertemplate="%{label}<br>Vehicles: %{value}<br>Share: %{percent}<extra></extra>")
        fig.update_layout(
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.02,
                        font=dict(size=10, color="#334155")),
            margin=dict(l=12, r=105, t=25, b=18),
        )
        polish_chart(fig, 360)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption("The pie chart compares the number of vehicles using each fuel type.")

    st.markdown("#### Average MPG by vehicle class")
    avg_class = (filtered_data.groupby("class", as_index=False)
                 .agg(avg_mpg=("comb_mpg", "mean"), vehicles=("comb_mpg", "size"))
                 .sort_values("avg_mpg", ascending=False).head(10).sort_values("avg_mpg", ascending=True))
    fig = px.bar(
        avg_class, x="avg_mpg", y="class", orientation="h", text="avg_mpg",
        labels={"avg_mpg": "Average combined MPG", "class": "Vehicle class", "vehicles": "Vehicle records"},
        color="avg_mpg", color_continuous_scale=["#93c5fd", "#2563eb"],
        hover_data={"vehicles": True, "avg_mpg": ":.2f"},
    )
    fig.update_traces(texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Average combined MPG")
    polish_chart(fig, 430)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("Top 10 vehicle classes by average fuel efficiency. Values are average combined MPG.")

    st.divider()
    add_automatic_insights(filtered_data, data)

    st.divider()
    page_heading("Project at a glance", "The workflow and techniques used in this project.")
    info1, info2 = st.columns(2, gap="large")
    with info1:
        st.markdown(
            """
            <div class="info-panel">
            <b>Data and processing</b><br><br>
            • EPA 2021 automobile dataset<br>
            • 1,108 vehicle records in the cleaned dataset<br>
            • PySpark DataFrame preprocessing<br>
            • StringIndexer, OneHotEncoder and VectorAssembler
            </div>
            """, unsafe_allow_html=True
        )
    with info2:
        st.markdown(
            """
            <div class="info-panel">
            <b>Machine learning</b><br><br>
            • Linear Regression and Random Forest Regression<br>
            • 80:20 training and testing split<br>
            • RMSE, MAE and R² evaluation<br>
            • Test-set predictions and interactive MPG estimation
            </div>
            """, unsafe_allow_html=True
        )

# ============================================================
# VEHICLE ANALYSIS
# ============================================================
elif page == "Vehicle Analysis":
    page_heading("Vehicle analysis", "Explore how vehicle category, engine size and fuel type relate to combined MPG.")
    if filtered_data.empty:
        st.warning("No vehicles match the selected filters. Adjust the filter selections above.")
        st.stop()

    st.caption(f"Showing {len(filtered_data):,} of {len(data):,} dataset records.")

    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("#### Average MPG by vehicle class")
        class_avg = (filtered_data.groupby("class", as_index=False)
                     .agg(avg_mpg=("comb_mpg", "mean"), vehicles=("comb_mpg", "size"))
                     .sort_values("avg_mpg", ascending=False).head(10).sort_values("avg_mpg", ascending=True))
        fig = px.bar(
            class_avg, x="avg_mpg", y="class", orientation="h", text="avg_mpg",
            color="avg_mpg", color_continuous_scale=["#93c5fd", "#1d4ed8"],
            labels={"avg_mpg": "Average combined MPG", "class": "Vehicle class", "vehicles": "Vehicle records"},
            hover_data={"vehicles": True, "avg_mpg": ":.2f"},
        )
        fig.update_traces(texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="")
        polish_chart(fig, 420)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption("Horizontal bar chart comparing average fuel efficiency across the 10 highest-MPG classes.")

    with right:
        st.markdown("#### Number of vehicles by fuel type")
        fuel_counts = (filtered_data["fuel_usage_desc"].value_counts()
                       .rename_axis("Fuel type").reset_index(name="Vehicles"))
        # fuel_counts intentionally uses the display column "Fuel type".
        # Reference that exact column name in Plotly to avoid a runtime error.
        fig = px.bar(
            fuel_counts, x="Fuel type", y="Vehicles", text="Vehicles",
            labels={"Fuel type": "Fuel type", "Vehicles": "Number of vehicles"},
            color="Fuel type", color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_tickangle=-25, xaxis_title="Fuel type")
        polish_chart(fig, 420)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        st.caption("Column chart showing how many dataset records belong to each fuel type.")

    st.markdown("#### Engine size vs average fuel efficiency")
    engine_trend = (filtered_data.assign(engine_group=filtered_data["engine_displacement"].round(1))
                    .groupby("engine_group", as_index=False)
                    .agg(avg_mpg=("comb_mpg", "mean"), vehicles=("comb_mpg", "size"))
                    .sort_values("engine_group"))
    fig = px.line(
        engine_trend, x="engine_group", y="avg_mpg", markers=True,
        labels={"engine_group": "Engine displacement (L)", "avg_mpg": "Average combined MPG", "vehicles": "Vehicle records"},
        color_discrete_sequence=["#0f766e"],
        hover_data={"vehicles": True, "avg_mpg": ":.2f"},
    )
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    polish_chart(fig, 390)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("Line chart of average MPG for engine displacement values rounded to 0.1 L. Each point summarizes vehicles in that group.")

    st.markdown("#### Individual vehicle records: engine size and MPG")
    color_by = st.selectbox(
        "Colour points by",
        ["fuel_usage_desc", "drive_desc", "class"],
        format_func=lambda x: {"class": "Vehicle class", "fuel_usage_desc": "Fuel type", "drive_desc": "Drive type"}[x],
        key="scatter_color",
    )
    fig = px.scatter(
        filtered_data, x="engine_displacement", y="comb_mpg", color=color_by,
        hover_data=["no_cylinders", "no_gears", "transmission_desc"],
        labels={"engine_displacement": "Engine displacement (L)", "comb_mpg": "Combined MPG",
                "class": "Vehicle class", "fuel_usage_desc": "Fuel type", "drive_desc": "Drive type"},
        opacity=0.55, render_mode="webgl", color_discrete_sequence=px.colors.qualitative.Safe,
    )
    polish_chart(fig, 430)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    st.caption("Each dot represents one vehicle record. Use the legend to compare categories.")

    st.divider()
    st.markdown("#### Vehicle class comparison")
    st.caption("Compare two vehicle classes using the current fuel, drive and engine filters. The global vehicle-class slicer is intentionally ignored in this comparison so both classes remain selectable.")
    comparison_base = data[
        ((data["fuel_usage_desc"] == selected_fuel) if selected_fuel != "All fuel types" else True)
        & ((data["drive_desc"] == selected_drive) if selected_drive != "All drive types" else True)
        & data["engine_displacement"].between(engine_range[0], engine_range[1])
    ].copy()
    comparison_classes = sorted(comparison_base["class"].dropna().unique().tolist())
    if len(comparison_classes) < 2:
        st.info("At least two vehicle classes are needed for a comparison. Widen the fuel, drive or engine filters.")
    else:
        compare_left, compare_right = st.columns(2, gap="large")
        with compare_left:
            class_a = st.selectbox("First vehicle class", comparison_classes, index=0, key="compare_class_a")
        with compare_right:
            default_b = 1 if len(comparison_classes) > 1 else 0
            class_b = st.selectbox("Second vehicle class", comparison_classes, index=default_b, key="compare_class_b")
        if class_a == class_b:
            st.warning("Choose two different vehicle classes to compare.")
        else:
            comparison_summary = (comparison_base[comparison_base["class"].isin([class_a, class_b])]
                                  .groupby("class", as_index=False)
                                  .agg(average_mpg=("comb_mpg", "mean"),
                                       average_engine_l=("engine_displacement", "mean"),
                                       average_cylinders=("no_cylinders", "mean"),
                                       vehicle_records=("comb_mpg", "size")))
            compare_kpis = []
            for _, item in comparison_summary.iterrows():
                compare_kpis.append((str(item["class"]), f'{item["average_mpg"]:.2f} MPG', "🚘",
                                     f'{int(item["vehicle_records"]):,} records · {item["average_engine_l"]:.2f} L average engine'))
            render_kpis(compare_kpis, columns=3)
            fig = px.bar(comparison_summary, x="class", y="average_mpg", color="class",
                         text="average_mpg", barmode="group",
                         color_discrete_sequence=["#60a5fa", "#2dd4bf"],
                         labels={"class": "Vehicle class", "average_mpg": "Average combined MPG"},
                         hover_data={"average_engine_l": ":.2f", "average_cylinders": ":.1f", "vehicle_records": True})
            fig.update_traces(texttemplate="%{y:.2f} MPG", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_rangemode="tozero")
            polish_chart(fig, 350)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            st.download_button("Download vehicle comparison CSV",
                               comparison_summary.to_csv(index=False).encode("utf-8"),
                               file_name="vehicle_class_comparison.csv", mime="text/csv")

    st.divider()
    add_automatic_insights(filtered_data, data)

# ============================================================
# MODEL PERFORMANCE
# ============================================================
elif page == "Model Performance":
    page_heading("Model performance", "Compare regression models and inspect prediction errors on the held-out test set.")

    comparison = pd.DataFrame([
        {"Model": name, "RMSE": values["RMSE"], "MAE": values["MAE"], "R² Score": values["R2"]}
        for name, values in results.items()
    ])
    if "Random Forest" not in results:
        st.error("Random Forest metrics are missing from model_results.json.")
        st.stop()
    rf = results["Random Forest"]

    render_kpis([
        ("Random Forest · RMSE", f'{rf["RMSE"]:.4f}', "◎", "Lower prediction error is better"),
        ("Random Forest · MAE", f'{rf["MAE"]:.4f}', "↔", "Average absolute error in MPG"),
        ("Random Forest · R²", f'{rf["R2"]:.4f}', "★", "Closer to 1.0 is better"),
    ], columns=3)

    st.divider()
    st.markdown("#### Regression model score comparison")
    st.caption("Column charts compare the two trained models. Lower RMSE and MAE are better; higher R² is better.")
    metric_left, metric_right = st.columns(2, gap="large")
    with metric_left:
        error_scores = comparison.melt(id_vars="Model", value_vars=["RMSE", "MAE"],
                                       var_name="Metric", value_name="Score")
        fig = px.bar(
            error_scores, x="Metric", y="Score", color="Model", barmode="group", text="Score",
            color_discrete_map={"Linear Regression": "#93a4b8", "Random Forest": "#0f9d91"},
            labels={"Score": "Error (MPG)"},
            category_orders={"Metric": ["RMSE", "MAE"]},
        )
        fig.update_traces(texttemplate="%{y:.3f}", textposition="outside", cliponaxis=False)
        fig.update_layout(legend_title_text="Model", yaxis_rangemode="tozero")
        polish_chart(fig, 360)
        st.markdown("**Prediction error metrics**")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with metric_right:
        r2_scores = comparison[["Model", "R² Score"]].copy()
        fig = px.bar(
            r2_scores, x="Model", y="R² Score", color="Model", text="R² Score",
            color_discrete_map={"Linear Regression": "#93a4b8", "Random Forest": "#0f9d91"},
            labels={"R² Score": "R² score"},
        )
        fig.update_traces(texttemplate="%{y:.4f}", textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, yaxis=dict(range=[0, 1], tickformat=".1f"))
        polish_chart(fig, 360)
        st.markdown("**Model goodness-of-fit**")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    left, right = st.columns([1, 1.4], gap="large")
    with left:
        st.markdown("#### Evaluation results")
        st.dataframe(comparison.round(4), use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="info-panel"><b>How to read the scores</b><br>'
            'RMSE penalizes larger errors more strongly. MAE is the average absolute error in MPG. '
            'R² measures the proportion of target variation explained by the model. '
            'For RMSE and MAE, lower is better; for R², higher is better.</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown("#### Actual vs predicted fuel efficiency")
        if filtered_predictions.empty:
            st.warning("No test predictions match the current filters, or the prediction file does not contain the required columns.")
        else:
            fig = px.scatter(
                filtered_predictions, x="comb_mpg", y="prediction",
                hover_data=["class", "engine_displacement", "no_cylinders", "no_gears"],
                labels={"comb_mpg": "Actual MPG", "prediction": "Predicted MPG"},
                opacity=0.68, render_mode="webgl",
                color_discrete_sequence=["#2563eb"],
            )
            low = float(min(filtered_predictions["comb_mpg"].min(), filtered_predictions["prediction"].min()))
            high = float(max(filtered_predictions["comb_mpg"].max(), filtered_predictions["prediction"].max()))
            fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="Perfect prediction",
                                     line=dict(color="#0f766e", dash="dash", width=2.5)))
            fig.update_layout(showlegend=True)
            polish_chart(fig, 390)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    st.divider()
    st.markdown("#### Prediction error distribution")
    if not filtered_predictions.empty:
        residuals = filtered_predictions["comb_mpg"] - filtered_predictions["prediction"]
        residual_df = pd.DataFrame({"Prediction error (MPG)": residuals})
        fig = px.histogram(residual_df, x="Prediction error (MPG)", nbins=24,
                           color_discrete_sequence=["#0f9d91"], opacity=0.88)
        fig.add_vline(x=0, line_dash="dash", line_color="#dc2626", line_width=2,
                      annotation_text="Zero error", annotation_position="top")
        fig.update_layout(bargap=0.08)
        polish_chart(fig, 320)
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        render_kpis([
            ("Selected test records", f"{len(filtered_predictions):,}", "▦", "Test-set vehicles after filters"),
            ("Filtered RMSE", f"{np.sqrt(np.mean(residuals ** 2)):.4f}", "◎", "Lower prediction error is better"),
            ("Filtered MAE", f"{np.mean(np.abs(residuals)):.4f}", "↔", "Average absolute error in MPG"),
        ], columns=3)
        st.caption("Positive errors mean the model underpredicted MPG; negative errors mean it overpredicted MPG.")

    st.markdown(
        '<div class="info-panel"><b>Metric note:</b> This is a regression task because the target, combined MPG, is a continuous number. '
        'Accuracy, precision, recall and F1-score are classification metrics and are not valid evaluation scores for this regression model. '
        'RMSE, MAE and R² are the appropriate metrics used here.</div>',
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("#### Random Forest feature importance")
    st.caption("Feature importance is taken from the saved Random Forest model. It describes how the fitted trees used the encoded features; it does not establish causation.")
    if filtered_data.empty:
        st.info("Select at least one vehicle record to inspect feature metadata.")
    else:
        try:
            with st.spinner("Reading feature importance from the saved Spark model..."):
                rf_pipeline = get_rf_model()
                importance_df = get_feature_importance(rf_pipeline, filtered_data)
            fig = px.bar(importance_df.sort_values("Importance", ascending=True),
                         x="Importance", y="Feature", orientation="h", text="Importance",
                         color="Importance", color_continuous_scale=["#1e3a5f", "#38bdf8", "#2dd4bf"],
                         labels={"Importance": "Relative importance", "Feature": "Encoded model feature"})
            fig.update_traces(texttemplate="%{x:.4f}", textposition="outside", cliponaxis=False)
            fig.update_layout(coloraxis_showscale=False, yaxis_title="")
            polish_chart(fig, max(360, 24 * len(importance_df)))
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            st.download_button("Download feature importance CSV", importance_df.to_csv(index=False).encode("utf-8"),
                               file_name="random_forest_feature_importance.csv", mime="text/csv")
        except Exception as exc:
            st.warning(f"Feature importance could not be displayed: {exc}")

# LIVE PREDICTION
# ============================================================
elif page == "Fuel Efficiency Prediction":
    page_heading("Fuel efficiency prediction", "Generate live MPG estimates, test what-if changes and review recent predictions.")

    st.markdown(
        '<div class="info-panel"><b>Prediction engine:</b> The dashboard uses the existing saved PySpark Random Forest pipeline. '
        'It does not retrain the model when you change vehicle specifications.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Live vehicle prediction")
    with st.form("prediction_form", border=True):
        st.markdown("##### Vehicle specifications")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            engine = st.number_input("Engine displacement (L)", min_value=0.5, max_value=10.0, value=2.0, step=0.1, key="live_engine")
            cylinders = st.selectbox("Number of cylinders", sorted(data["no_cylinders"].dropna().unique().tolist()), key="live_cylinders")
            gears = st.selectbox("Number of gears", sorted(data["no_gears"].dropna().unique().tolist()), key="live_gears")
            transmission = st.selectbox("Transmission", sorted(data["transmission_desc"].dropna().unique().tolist()), key="live_transmission")
        with c2:
            drive = st.selectbox("Drive type", sorted(data["drive_desc"].dropna().unique().tolist()), key="live_drive")
            vehicle_class = st.selectbox("Vehicle class", vehicle_options, key="live_class")
            fuel = st.selectbox("Fuel type", fuel_options, key="live_fuel")
        submitted = st.form_submit_button("Estimate fuel efficiency", type="primary", use_container_width=True)

    if submitted:
        try:
            with st.spinner("Running the saved Random Forest pipeline..."):
                prediction_value = predict_vehicle(engine, cylinders, gears, transmission, drive, vehicle_class, fuel)
            record_prediction("Live prediction", engine, cylinders, gears, transmission, drive, vehicle_class, fuel, prediction_value)
            st.success("Prediction completed and added to this session's history.")
            result_col, detail_col = st.columns([1, 2], gap="large")
            with result_col:
                render_kpis([("Estimated combined efficiency", f"{prediction_value:.2f} MPG", "↗", "Saved Random Forest model estimate")], columns=3)
            with detail_col:
                st.markdown(
                    f'<div class="info-panel"><b>Vehicle submitted</b><br>Engine: {engine:.1f} L · Cylinders: {cylinders} · Gears: {gears}<br>Transmission: {transmission}<br>Class: {vehicle_class}<br>Fuel: {fuel}<br>Drive: {drive}</div>',
                    unsafe_allow_html=True,
                )
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")

    st.divider()
    st.markdown("#### What-if prediction simulator")
    st.caption("Keep the vehicle category and drivetrain the same, then change engine displacement to see how the trained model's estimate responds. This is a model scenario, not a real-world guarantee.")
    with st.form("what_if_form", border=True):
        w1, w2 = st.columns(2, gap="large")
        with w1:
            baseline_engine = st.slider("Baseline engine displacement (L)", min_value=engine_min, max_value=engine_max,
                                        value=float(min(max(2.0, engine_min), engine_max)), step=0.1, key="whatif_base_engine")
            scenario_engine = st.slider("Scenario engine displacement (L)", min_value=engine_min, max_value=engine_max,
                                        value=float(min(max(3.0, engine_min), engine_max)), step=0.1, key="whatif_scenario_engine")
            whatif_cylinders = st.selectbox("Cylinders", sorted(data["no_cylinders"].dropna().unique().tolist()), key="whatif_cylinders")
            whatif_gears = st.selectbox("Gears", sorted(data["no_gears"].dropna().unique().tolist()), key="whatif_gears")
        with w2:
            whatif_transmission = st.selectbox("Transmission", sorted(data["transmission_desc"].dropna().unique().tolist()), key="whatif_transmission")
            whatif_drive = st.selectbox("Drive type", sorted(data["drive_desc"].dropna().unique().tolist()), key="whatif_drive")
            whatif_class = st.selectbox("Vehicle class", vehicle_options, key="whatif_class")
            whatif_fuel = st.selectbox("Fuel type", fuel_options, key="whatif_fuel")
        run_scenario = st.form_submit_button("Run what-if comparison", type="primary", use_container_width=True)

    if run_scenario:
        try:
            with st.spinner("Comparing baseline and scenario using Random Forest..."):
                baseline_prediction = predict_vehicle(baseline_engine, whatif_cylinders, whatif_gears,
                                                       whatif_transmission, whatif_drive, whatif_class, whatif_fuel)
                scenario_prediction = predict_vehicle(scenario_engine, whatif_cylinders, whatif_gears,
                                                      whatif_transmission, whatif_drive, whatif_class, whatif_fuel)
            difference = scenario_prediction - baseline_prediction
            c1, c2, c3 = st.columns(3)
            c1.metric("Baseline prediction", f"{baseline_prediction:.2f} MPG")
            c2.metric("Scenario prediction", f"{scenario_prediction:.2f} MPG")
            c3.metric("Scenario change", f"{difference:+.2f} MPG")
            scenario_comparison = pd.DataFrame([
                {"Scenario": "Baseline", "Engine displacement (L)": baseline_engine, "Predicted MPG": baseline_prediction},
                {"Scenario": "What-if", "Engine displacement (L)": scenario_engine, "Predicted MPG": scenario_prediction},
            ])
            fig = px.bar(scenario_comparison, x="Scenario", y="Predicted MPG", color="Scenario", text="Predicted MPG",
                         color_discrete_sequence=["#60a5fa", "#2dd4bf"],
                         hover_data={"Engine displacement (L)": ":.1f"})
            fig.update_traces(texttemplate="%{y:.2f} MPG", textposition="outside", cliponaxis=False)
            fig.update_layout(showlegend=False, yaxis_rangemode="tozero")
            polish_chart(fig, 330)
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            record_prediction("What-if baseline", baseline_engine, whatif_cylinders, whatif_gears,
                              whatif_transmission, whatif_drive, whatif_class, whatif_fuel, baseline_prediction)
            record_prediction("What-if scenario", scenario_engine, whatif_cylinders, whatif_gears,
                              whatif_transmission, whatif_drive, whatif_class, whatif_fuel, scenario_prediction)
            st.download_button("Download what-if comparison CSV", scenario_comparison.to_csv(index=False).encode("utf-8"),
                               file_name="what_if_prediction_comparison.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"What-if simulation failed: {exc}")

    st.divider()
    st.markdown("#### Prediction history")
    history = st.session_state.get("prediction_history", [])
    if history:
        history_df = pd.DataFrame(history)
        st.caption(f"Showing the latest {len(history_df)} predictions from this browser session. History resets when the Streamlit session is cleared.")
        st.dataframe(history_df, use_container_width=True, hide_index=True, height=300)
        history_csv = history_df.to_csv(index=False).encode("utf-8")
        st.download_button("Download prediction history CSV", history_csv,
                           file_name="automobile_prediction_history.csv", mime="text/csv")
        if st.button("Clear prediction history", key="clear_prediction_history"):
            st.session_state["prediction_history"] = []
            st.rerun()
    else:
        st.info("No predictions have been recorded in this session yet. Run a live prediction or what-if comparison to start the history.")

# ============================================================
# DATASET EXPLORER
# ============================================================
elif page == "Dataset Explorer":
    page_heading("Dataset explorer", "Inspect, summarize and export the records selected by the filters above.")
    if filtered_data.empty:
        st.warning("No records match the selected filters.")
        st.stop()

    render_kpis([("Records currently displayed", f"{len(filtered_data):,}", "▦", "Records available in this view")], columns=3)
    search = st.text_input("Search within the displayed records", placeholder="Search vehicle class, fuel type, drive type...")
    view = filtered_data.copy()
    if search.strip():
        mask = view.astype(str).apply(
            lambda col: col.str.contains(search.strip(), case=False, na=False)
        ).any(axis=1)
        view = view[mask]

    st.caption(f"Showing {len(view):,} matching records.")
    st.dataframe(view, use_container_width=True, hide_index=True, height=420)

    download_col, stats_col = st.columns([1, 2], gap="large")
    with download_col:
        st.download_button(
            "Download filtered CSV",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="filtered_automobile_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with stats_col:
        st.markdown("#### Numeric summary")
        numeric_cols = ["engine_displacement", "no_cylinders", "no_gears", "comb_mpg"]
        st.dataframe(view[numeric_cols].describe().round(2), use_container_width=True)

    st.download_button("Download dataset summary CSV",
                       view[numeric_cols].describe().round(2).to_csv().encode("utf-8"),
                       file_name="automobile_dataset_summary.csv", mime="text/csv")
    if not view.empty:
        add_automatic_insights(view, data)

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    '<div class="small-muted" style="text-align:center;">Automobile Fuel Efficiency Prediction System '
    '· Apache Spark · PySpark MLlib · Random Forest · Streamlit</div>',
    unsafe_allow_html=True,
)
