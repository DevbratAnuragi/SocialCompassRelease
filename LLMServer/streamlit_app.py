import json, time
from pathlib import Path
import pandas as pd
import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
import json


LOG_PATH = Path("ui_logs/hase_events.jsonl")

st.set_page_config(page_title="SocialCompass Runtime", layout="wide")

st.markdown("""
<style>
.badge {padding:6px 10px;border-radius:8px;font-weight:600;}
.badge.ok {background:#e6ffed;color:#046307;}
.badge.warn {background:#fff5e6;color:#92400e;}
.badge.err {background:#ffe6e6;color:#7a0619;}
.badge.info {background:#e6f0ff;color:#003b8e;}
.flowbox {border:1px solid #eee;border-radius:12px;padding:12px;background:#fafafa;}
.step {display:inline-block;margin-right:8px;padding:6px 10px;border-radius:999px;background:#f0f0f0;}
.step.on {background:#d1fae5;color:#065f46;font-weight:600;}
.step.off{opacity:0.4;}
</style>
""", unsafe_allow_html=True)

st.title("🌡️ SocialCompass Live Runtime")

def read_tail_jsonl(path: Path, max_lines: int = 1000):
    if not path.exists():
        return []
    # read last max_lines quickly
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()[-max_lines:]
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except Exception:
            pass
    return out

# auto-refresh every 1s
st_autorefresh = st.experimental_user
st.sidebar.markdown("### Controls")
interval = st.sidebar.slider("Refresh (ms)", 200, 3000, 1000, 100)
limit    = st.sidebar.slider("Max log rows", 100, 5000, 1000, 100)


placeholder = st.empty()
time.sleep(0.05)  # small delay so first render is clean

def badge(text, kind="info"):
    return f'<span class="badge {kind}">{text}</span>'

def flow_steps(latest_status: str):
    # Order of statuses we care about
    steps = [
        ("models_loaded", "Models Loaded"),
        ("server_listening", "Server Listening"),
        ("client_connected", "Client Connected"),
        ("window_rx", "Window RX"),
        ("embeddings_done", "Embeddings"),
        ("scores", "Scored"),
        ("decision", "Decision → TX"),
    ]
    html = ['<div class="flowbox">']
    seen = False
    for key, label in steps:
        active = (key == latest_status) or seen
        css = "step on" if active else "step off"
        html.append(f'<span class="{css}">{label}</span>')
        if key == latest_status:  # after this step, dim following
            seen = True
    html.append("</div>")
    return "\n".join(html)

# main loop-ish render (streamlit reruns on each interaction/refesh)
events = read_tail_jsonl(LOG_PATH, max_lines=limit)


if len(events) == 0:
    st.info("No events yet. Start the server and send a window from Beam Pro.")
    st.stop()
import numpy as np
import pandas as pd
import altair as alt
from pathlib import Path
import json

# read original thresholds from first "models_loaded" event (fallbacks if missing)
def _orig_thresholds(evts):
    ml = next((e for e in evts if e.get("status")=="models_loaded"), {})
    return float(ml.get("theta_social", 0.50)), float(ml.get("theta_arousal", 0.0))

theta_social_orig, theta_arousal_orig = _orig_thresholds(events)

st.sidebar.markdown("### Threshold overrides")
# initialize session state
st.session_state.setdefault("theta_social_override", None)
st.session_state.setdefault("theta_arousal_override", None)

colA, colB = st.sidebar.columns(2)
with colA:
    use_social = st.checkbox("Override θ_social", value=st.session_state["theta_social_override"] is not None)
with colB:
    use_arousal = st.checkbox("Override θ_arousal", value=st.session_state["theta_arousal_override"] is not None)

if use_social:
    st.session_state["theta_social_override"] = st.sidebar.slider(
        "θ_social", 0.0, 1.0, 
        value=float(st.session_state["theta_social_override"] or theta_social_orig), step=0.01
    )
if use_arousal:
    st.session_state["theta_arousal_override"] = st.sidebar.slider(
        "θ_arousal", -1.5, 1.5,
        value=float(st.session_state["theta_arousal_override"] or theta_arousal_orig), step=0.01
    )

if st.sidebar.button("Reset thresholds"):
    st.session_state["theta_social_override"] = None
    st.session_state["theta_arousal_override"] = None

# active thresholds used by the dashboard
theta_social = float(st.session_state["theta_social_override"] if use_social else theta_social_orig)
theta_arousal = float(st.session_state["theta_arousal_override"] if use_arousal else theta_arousal_orig)
OVERRIDE_PATH = Path("ui_logs/threshold_override.json")

if st.sidebar.button("Apply thresholds to server"):
    OVERRIDE_PATH.parent.mkdir(parents=True, exist_ok=True)
    override = {"theta_social": theta_social, "theta_arousal": theta_arousal}
    OVERRIDE_PATH.write_text(json.dumps(override), encoding="utf-8")
    st.sidebar.success("Override written for server.")
st.sidebar.caption(f"Active θ_social = **{theta_social:.3f}**, θ_arousal = **{theta_arousal:.3f}** (UI only)")


# current status
latest = events[-1]
status = latest.get("status", "unknown")

cols = st.columns([2,3,3,3])
with cols[0]:
    st.markdown("**Server State**")
    if status in ("server_starting", "models_loaded", "server_listening"):
        st.markdown(badge(status.replace("_"," ").title(), "info"), unsafe_allow_html=True)
    elif status in ("client_connected", "window_rx", "embeddings_done", "scores"):
        st.markdown(badge(status.replace("_"," ").title(), "ok"), unsafe_allow_html=True)
    elif status == "decision":
        st.markdown(badge("Decision Emitted", "ok"), unsafe_allow_html=True)
    else:
        st.markdown(badge(status, "warn"), unsafe_allow_html=True)

with cols[1]:
    # Last decision (if present)
    last_decisions = [e for e in reversed(events) if e.get("status")=="decision"]
    if last_decisions:
        dec = last_decisions[0]
        hase = bool(dec.get("hase", False))
        txt  = dec.get("text","")
        st.markdown("**Latest Decision**")
        st.markdown(badge("HASE DETECTED" if hase else "No HASE", "ok" if hase else "warn"), unsafe_allow_html=True)
        st.write(txt)

with cols[2]:
    # Last window summary
    last_rx = next((e for e in reversed(events) if e.get("status")=="window_rx"), None)
    if last_rx:
        st.markdown("**Last Window**")
        st.write(f"t0 = `{last_rx.get('t0', '-')}`")
        st.write(f"IMU = `{last_rx.get('imu_shape', '-')}` @ `{last_rx.get('imu_hz', '-')}` Hz")
        st.write(f"MFCC = `{last_rx.get('mfcc_shape', '-')}`")

with cols[3]:
    # Guidance
    st.markdown("**Tip**")
    st.write("Open this dashboard while your server is running. Each 20s window will update charts and status.")

# flow visualization
st.markdown(flow_steps(status), unsafe_allow_html=True)

# charts for scores over time
# build score frame
df_scores = pd.DataFrame(
    [(e.get("ts"), e.get("p_social"), e.get("s_arousal"))
     for e in events if e.get("status")=="scores"],
    columns=["ts","p_social","s_arousal"]
).dropna()

if len(df_scores):
    # stable x index (nice for live updates)
    df_scores["idx"] = np.arange(len(df_scores))

    left, right = st.columns(2)

    # ---- p_social with θ_social rule ----
    with left:
        st.subheader("p_social over time")
        base = alt.Chart(df_scores).mark_line().encode(
            x=alt.X("idx:Q", title="window #"),
            y=alt.Y("p_social:Q", title="p_social", scale=alt.Scale(domain=[0,1]))
        )
        rule = alt.Chart(pd.DataFrame({"y":[theta_social]})).mark_rule(color="red").encode(y="y:Q")
        st.altair_chart(base + rule, use_container_width=True)

    # ---- s_arousal with θ_arousal rule ----
    with right:
        st.subheader("s_arousal over time")
        # auto domain with small margins
        ymin = float(min(-0.5, df_scores["s_arousal"].min() - 0.05))

        ymax = float(max( 0.05, df_scores["s_arousal"].max() + 0.05))

        base = alt.Chart(df_scores).mark_line().encode(
            x=alt.X("idx:Q", title="window #"),
            y=alt.Y("s_arousal:Q", title="s_arousal", scale=alt.Scale(domain=[ymin, ymax]))
        )
        rule = alt.Chart(pd.DataFrame({"y":[theta_arousal]})).mark_rule(color="red").encode(y="y:Q")
        st.altair_chart(base + rule, use_container_width=True)


# rolling table with color highlights
st.subheader("Recent events")

df = pd.DataFrame(events)

# ── Latest decision badge (above the table)
last_dec = next((e for e in reversed(events) if e.get("status") == "decision"), None)
if last_dec:
    hase = bool(last_dec.get("hase", False))
    txt  = last_dec.get("text", "")
    st.markdown(
        f"""<div class="flowbox">
              <span class="badge {'ok' if hase else 'warn'}">
                {'HASE DETECTED' if hase else 'No HASE'}
              </span>
              &nbsp; {txt}
            </div>""",
        unsafe_allow_html=True
    )

# ── Pretty hase column (don’t turn NaN into ✅)
import numpy as np
def pretty_hase(x):
    if x is True:  return "✅"
    if x is False: return "❌"
    if x is None:  return ""
    if isinstance(x, float) and np.isnan(x): return ""
    return ""

if "hase" in df.columns:
    df["hase"] = df["hase"].apply(pretty_hase)

# (optional) tidy order
cols_order = ["ts","status","device","url","remote","t0","imu_shape","mfcc_shape","imu_hz","p_social","s_arousal","hase","text"]
df = df[[c for c in cols_order if c in df.columns]]

# find last decision row index
mask = df["status"].eq("decision") if "status" in df.columns else None
last_idx = df[mask].index.max() if mask is not None and mask.any() else None

def highlight_last(row):
    if last_idx is not None and row.name == last_idx:
        return ['background-color: #ECFDF5'] * len(row)  # soft green
    return [''] * len(row)

styled = df.tail(200).style.apply(highlight_last, axis=1)
st.dataframe(styled, use_container_width=True)


# auto refresh
auto = st.sidebar.checkbox("Auto refresh", value=True)

if auto:
    import time
    time.sleep(interval/1000.0)
    st.rerun()
