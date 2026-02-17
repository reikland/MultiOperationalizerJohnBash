import os

import streamlit as st

from forecast_brief.client import llm_raw
from forecast_brief.formatter import build_final_txt
from forecast_brief.pipeline import run_questions_per_cluster, run_tier2_three_calls
from forecast_brief.prompts_core import SYSTEM_T1, SYSTEM_T3, USER_T1, USER_T3
from forecast_brief.utils import add_months, clean, ensure_nonempty_geo, split_clusters, today_utc_date

st.set_page_config(page_title="Forecasting Brief (RAW, public-resolvable questions)", layout="wide")
st.title("Forecasting brief (RAW) — Tier 2 = :online×3 chained, Tier 4 = 2–3 public-resolvable questions per cluster")

with st.sidebar:
    st.header("OpenRouter")
    api_key = st.text_input("OPENROUTER_API_KEY", value=os.getenv("OPENROUTER_API_KEY", ""), type="password")

    st.divider()
    st.header("Models")
    base_model = st.text_input("Base model (Tier 1 & Tier 3)", value="openai/gpt-5.2-chat")
    sector_model = st.text_input("Sector model (Tier 2) — forced :online", value="openai/gpt-5.2-chat")
    question_model = st.text_input("Question model (Tier 4, per cluster)", value="openai/gpt-5.2-chat")

    st.divider()
    st.header("Settings")
    geo = st.text_input("Primary geography", value="EU (France/Benelux/Germany/Italy)")
    horizon = st.slider("Horizon (months)", 3, 36, 18, 1)
    n_clusters = st.slider("Target clusters (5–10)", 5, 10, 7, 1)
    risk = st.selectbox("Risk appetite", ["low", "medium", "high"], index=1)

    st.divider()
    st.header("Controls")
    temperature = st.slider("Temperature", 0.0, 1.2, 0.20, 0.05)
    max_tokens = st.slider("Max tokens per call", 700, 2600, 1700, 100)
    stream_debug = st.checkbox("Stream intermediate raw text (noisy)", value=False)
    show_intermediate = st.checkbox("Show intermediate outputs", value=True)

company = st.text_area(
    "Company description (English recommended). 2–12 lines.",
    height=220,
    placeholder="Example: B2B SaaS for invoice-to-cash automation in EU; compliance (PDP/PPF), OCR, payments...",
)

colA, colB = st.columns([1, 1])
run_btn = colA.button("Generate brief", type="primary", use_container_width=True)
clear_btn = colB.button("Clear", use_container_width=True)

status = st.empty()
progress = st.progress(0)

if clear_btn:
    st.session_state.clear()
    st.rerun()

for key in ["final_txt", "t1", "t2_full", "t2_snapshot", "t3", "t4"]:
    st.session_state.setdefault(key, "")


def must_have() -> bool:
    if not clean(api_key):
        st.error("Missing OpenRouter API key.")
        return False
    if not clean(company):
        st.error("Missing company description.")
        return False
    return True


if run_btn and must_have():
    try:
        geo_value = ensure_nonempty_geo(geo)
        deadline_date = add_months(today_utc_date(), int(horizon)).isoformat()
        stream_target = st if stream_debug else None

        progress.progress(0.1)
        status.info("Tier 1/4 — Axes…")
        t1 = clean(llm_raw(api_key, base_model, SYSTEM_T1, USER_T1.format(company=company, geo=geo_value, horizon=horizon, risk=risk), temperature, max_tokens, stream_target))
        st.session_state.t1 = t1

        progress.progress(0.35)
        status.info("Tier 2/4 — Sector research (:online × 3 calls, chained)…")
        t2_full, t2_snapshot = run_tier2_three_calls(
            llm=llm_raw,
            api_key=api_key,
            sector_model_raw=sector_model,
            company=company,
            tier1_text=t1,
            geo=geo_value,
            horizon=horizon,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_to=stream_target,
        )
        st.session_state.t2_full = t2_full
        st.session_state.t2_snapshot = t2_snapshot

        progress.progress(0.6)
        status.info("Tier 3/4 — Clusters…")
        t3 = clean(
            llm_raw(
                api_key,
                base_model,
                SYSTEM_T3,
                USER_T3.format(company=company, tier1_text=t1, tier2_snapshot=t2_snapshot, geo=geo_value, horizon=horizon, n_clusters=n_clusters),
                temperature,
                max_tokens,
                stream_target,
            )
        )
        clusters = split_clusters(t3)
        if len(clusters) < 5:
            status.warning("Tier 3 produced too few clusters; repairing…")
            t3 = clean(
                llm_raw(
                    api_key,
                    base_model,
                    "You are a strict formatter and strategist. Output RAW TEXT only.",
                    f"Rewrite Tier 3 with 5–10 clusters and required format. Target: {n_clusters}.\n\nFaulty Tier 3:\n{t3}",
                    0.2,
                    min(1700, max_tokens),
                    None,
                )
            )
            clusters = split_clusters(t3)
        st.session_state.t3 = t3

        progress.progress(0.8)
        status.info("Tier 4/4 — Questions (2–3 public-resolvable questions per cluster)…")
        t4 = run_questions_per_cluster(
            llm=llm_raw,
            api_key=api_key,
            question_model=question_model,
            company=company,
            tier1_text=t1,
            tier2_snapshot=t2_snapshot,
            clusters=clusters,
            geo=geo_value,
            horizon=horizon,
            risk=risk,
            deadline_date=deadline_date,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_to=stream_target,
        )
        st.session_state.t4 = t4
        st.session_state.final_txt = build_final_txt(company, geo_value, horizon, risk, deadline_date, t1, t2_full, t2_snapshot, t3, t4)

        progress.progress(1.0)
        status.success("Done.")
    except Exception as exc:
        status.error(f"Error: {exc}")
        progress.progress(0)

st.subheader("Output TXT (RAW)")
st.text_area("TXT", value=st.session_state.final_txt, height=520)

if st.session_state.final_txt:
    st.download_button(
        "Download .txt",
        data=st.session_state.final_txt.encode("utf-8"),
        file_name="forecasting_brief_public_resolvable_raw.txt",
        mime="text/plain",
        use_container_width=True,
    )

if show_intermediate and st.session_state.t1:
    st.subheader("Intermediate outputs (RAW)")
    for label, key in [("Tier 1", "t1"), ("Tier 2 (all calls)", "t2_full"), ("Tier 2 (final snapshot)", "t2_snapshot"), ("Tier 3", "t3"), ("Tier 4 (questions)", "t4")]:
        with st.expander(label, expanded=False):
            st.text(st.session_state[key])
