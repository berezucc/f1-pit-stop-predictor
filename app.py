"""F1 pit-stop decision dashboard.

Pick a track, a tyre compound and the current lap state; the CatBoost model
trained in train_app_model.py returns the probability the car boxes on the next
lap and a PIT / STAY OUT call. A strategy curve sweeps tyre life so you can see
where the model flips its decision.
"""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from catboost import CatBoostClassifier

HERE = Path(__file__).resolve().parent

COMPOUND_COLOR = {
    "SOFT": "#e10600",
    "MEDIUM": "#f5c518",
    "HARD": "#e8e8e8",
    "INTERMEDIATE": "#43b02a",
    "WET": "#0067ad",
}


@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model(str(HERE / "model.cbm"))
    meta = json.loads((HERE / "meta.json").read_text())
    return model, meta


def predict(model, meta, row):
    df = pd.DataFrame([row])[meta["features"]]
    return float(model.predict_proba(df)[:, 1][0])


def strategy_curve(model, meta, base, race_len):
    """Pit probability as tyre life sweeps, holding everything else fixed."""
    rows = []
    for tl in range(1, meta["tyrelife_max"] + 1):
        r = dict(base, TyreLife=tl)
        r["RaceProgress"] = r["LapNumber"] / race_len
        rows.append({"TyreLife": tl, "p": predict(model, meta, r)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page + style
st.set_page_config(page_title="F1 Pit-Stop Predictor", page_icon="🏎️", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
      .stApp { background: #0e0e10; }
      h1, h2, h3, h4, h5, label, p, span, div { color: #f2f2f2; }
      .block-container { padding-top: 2.2rem; max-width: 1200px; }
      .hero { border-left: 5px solid #e10600; padding: .2rem 0 .2rem 1rem; margin-bottom:.4rem; }
      .hero h1 { font-size: 2.1rem; font-weight: 800; letter-spacing:-.5px; margin:0; }
      .hero p  { color:#9a9a9f; margin:.2rem 0 0; font-size:.95rem; }
      .card { background:#161619; border:1px solid #242428; border-radius:16px;
              padding:1.4rem 1.6rem; }
      .verdict { font-size:3rem; font-weight:800; letter-spacing:-1px; margin:.2rem 0; }
      .prob { font-size:3.4rem; font-weight:800; letter-spacing:-2px; }
      .sub  { color:#9a9a9f; font-size:.85rem; text-transform:uppercase; letter-spacing:1.5px; }
      .metric-pill { background:#161619; border:1px solid #242428; border-radius:12px;
                     padding:.8rem 1rem; text-align:center; }
      .metric-pill .v { font-size:1.5rem; font-weight:700; }
      .metric-pill .k { color:#9a9a9f; font-size:.72rem; text-transform:uppercase; letter-spacing:1px; }
      [data-testid="stSidebar"] { background:#121214; border-right:1px solid #242428; }
      .stSlider [data-baseweb="slider"] div[role="slider"] { background:#e10600; }
    </style>
    """,
    unsafe_allow_html=True,
)

model, meta = load_model()
THRESHOLD = meta.get("threshold", 0.5)  # data-driven cutoff from train script

st.markdown(
    '<div class="hero"><h1>🏎️ F1 PIT-STOP PREDICTOR</h1>'
    "<p>CatBoost over 2022–2025 race data · will the car box on the next lap?</p></div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------- input panel
with st.sidebar:
    st.markdown("### Race situation")
    year = st.selectbox("Season", meta["years"], index=len(meta["years"]) - 1)
    race = st.selectbox("Grand Prix", meta["races"],
                        index=meta["races"].index("Bahrain Grand Prix")
                        if "Bahrain Grand Prix" in meta["races"] else 0)
    compound = st.select_slider("Tyre compound", options=meta["compounds"], value="MEDIUM")
    position = st.slider("Track position", 1, meta["position_max"], 5)

    race_len = meta["race_len"].get(race, 60)
    st.markdown("### Lap state")
    lap = st.slider("Lap number", 1, race_len, min(race_len // 3, race_len))
    stint = st.slider("Stint", 1, meta["stint_max"], 1)
    tyre_life = st.slider("Tyre life (laps on this set)", 1, meta["tyrelife_max"],
                          min(15, meta["tyrelife_max"]))

row = {
    "Race": race, "Compound": compound, "Year": year,
    "LapNumber": lap, "Stint": stint, "TyreLife": tyre_life,
    "Position": position, "RaceProgress": lap / race_len,
}
p = predict(model, meta, row)
will_pit = p >= THRESHOLD
color = "#e10600" if will_pit else "#43b02a"
verdict = "🟥 BOX, BOX" if will_pit else "🟩 STAY OUT"

# -------------------------------------------------------------------- results
left, right = st.columns([1, 1.15], gap="large")

with left:
    st.markdown(
        f"""
        <div class="card" style="border-color:{color}">
          <div class="sub">Decision · lap {lap} of {race_len}</div>
          <div class="verdict" style="color:{color}">{verdict}</div>
          <div class="prob" style="color:{color}">{p*100:.1f}%</div>
          <div class="sub">pit-next-lap probability · threshold {THRESHOLD*100:.0f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(p, 1.0))

    chip = COMPOUND_COLOR[compound]
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-pill"><div class="v" style="color:{chip}">{compound[:4]}</div>'
                '<div class="k">compound</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-pill"><div class="v">{tyre_life}</div>'
                '<div class="k">tyre laps</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-pill"><div class="v">{row["RaceProgress"]*100:.0f}%</div>'
                '<div class="k">race done</div></div>', unsafe_allow_html=True)

with right:
    st.markdown("##### Strategy curve — pit probability vs tyre life")
    curve = strategy_curve(model, meta, row, race_len)
    base = (
        alt.Chart(curve)
        .mark_area(
            line={"color": "#e10600", "strokeWidth": 2},
            color=alt.Gradient(
                gradient="linear",
                stops=[alt.GradientStop(color="#e1060055", offset=0),
                       alt.GradientStop(color="#e1060000", offset=1)],
                x1=1, x2=1, y1=0, y2=1),
        )
        .encode(
            x=alt.X("TyreLife:Q", title="Tyre life (laps)"),
            y=alt.Y("p:Q", title="Pit probability", scale=alt.Scale(domain=[0, 1])),
        )
    )
    rule = alt.Chart(pd.DataFrame({"y": [THRESHOLD]})).mark_rule(
        color="#9a9a9f", strokeDash=[4, 4]).encode(y="y:Q")
    now = alt.Chart(pd.DataFrame({"x": [tyre_life]})).mark_rule(
        color="#f5c518", strokeWidth=2).encode(x="x:Q")
    st.altair_chart(
        (base + rule + now).properties(height=300).configure_view(
            fill="#161619", stroke="#242428").configure_axis(
            gridColor="#222", labelColor="#9a9a9f", titleColor="#9a9a9f"),
        use_container_width=True,
    )
    st.caption("Yellow line = current tyre life · dashed line = decision threshold")

# --------------------------------------------------------------- model card
st.markdown("---")
imp = pd.DataFrame(
    sorted(meta["importances"].items(), key=lambda kv: kv[1]),
    columns=["feature", "importance"],
)
mc1, mc2 = st.columns([1, 1.15], gap="large")
with mc1:
    a, b, c = st.columns(3)
    a.markdown(f'<div class="metric-pill"><div class="v">{meta["oof_auc"]:.3f}</div>'
               '<div class="k">OOF ROC-AUC</div></div>', unsafe_allow_html=True)
    b.markdown(f'<div class="metric-pill"><div class="v">{meta["base_rate"]*100:.1f}%</div>'
               '<div class="k">base pit rate</div></div>', unsafe_allow_html=True)
    c.markdown(f'<div class="metric-pill"><div class="v">{meta["n_rows"]//1000}k</div>'
               '<div class="k">training laps</div></div>', unsafe_allow_html=True)
    st.caption("5-fold out-of-fold AUC on a single-row model. The full Kaggle "
               "ensemble (sequence features + stacking) reaches 0.949; this is its "
               "interactive cousin, scoring one lap at a time.")
with mc2:
    st.markdown("##### What drives the call")
    chart = (
        alt.Chart(imp)
        .mark_bar(color="#e10600", cornerRadiusEnd=4)
        .encode(
            x=alt.X("importance:Q", title=None),
            y=alt.Y("feature:N", sort="-x", title=None),
        )
        .properties(height=220)
        .configure_view(fill="#161619", stroke="#242428")
        .configure_axis(gridColor="#222", labelColor="#cfcfcf", titleColor="#9a9a9f")
    )
    st.altair_chart(chart, use_container_width=True)
