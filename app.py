# app.py — Streamlit + OpenRouter (RAW output)
# Goal: You paste a short company description -> output a forecasting brief:
#   Tier 1 (axes) -> Tier 2 (sector research, 3 calls with :online, chained) -> Tier 3 (5–10 clusters)
#   Tier 4 (ONE CALL PER CLUSTER): 2–3 PUBLICLY RESOLVABLE + decision-useful questions (no internal company metrics)
#
# Install:
#   python3 -m pip install streamlit requests
# Run:
#   streamlit run app.py
#
# Env:
#   export OPENROUTER_API_KEY="..."

import os
import re
import json
import textwrap
from datetime import datetime, date, timedelta
from typing import Dict, List, Tuple, Optional

import requests
import streamlit as st

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


# -----------------------------
# Date helpers
# -----------------------------
def today_utc_date() -> date:
    return datetime.utcnow().date()

def now_utc_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

def last_day_of_month(y: int, m: int) -> int:
    first = date(y, m, 1)
    if m == 12:
        next_first = date(y + 1, 1, 1)
    else:
        next_first = date(y, m + 1, 1)
    return (next_first - timedelta(days=1)).day

def add_months(d: date, months: int) -> date:
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    ld = last_day_of_month(y, m)
    return date(y, m, min(d.day, ld))


# -----------------------------
# Text helpers
# -----------------------------
def clean(s: str) -> str:
    return (s or "").strip()

def wrap(txt: str, width: int = 96) -> str:
    out = []
    for para in (txt or "").splitlines():
        if not para.strip():
            out.append("")
        else:
            out.extend(textwrap.fill(para, width=width).splitlines())
    return "\n".join(out).strip()

def ensure_nonempty_geo(geo: str) -> str:
    g = clean(geo)
    return g if g else "Global"

def ensure_online_model(model: str) -> str:
    m = clean(model)
    if not m:
        return "openai/gpt-5.2-chat:online"
    return m if m.endswith(":online") else (m + ":online")

def extract_between(text: str, start_marker: str, end_marker: Optional[str] = None) -> str:
    if not text:
        return ""
    i = text.find(start_marker)
    if i < 0:
        return ""
    sub = text[i + len(start_marker):]
    if end_marker:
        j = sub.find(end_marker)
        if j >= 0:
            sub = sub[:j]
    return sub.strip()

def split_clusters(tier3_text: str) -> List[Tuple[str, str]]:
    """
    Each cluster begins with: CLUSTER: <Name>
    Returns list of (name, block_text).
    """
    t = tier3_text or ""
    starts = [(m.start(), m.group(1).strip()) for m in re.finditer(r"^CLUSTER:\s*(.+?)\s*$", t, flags=re.MULTILINE)]
    blocks: List[Tuple[str, str]] = []
    for idx, (pos, name) in enumerate(starts):
        end = starts[idx + 1][0] if idx + 1 < len(starts) else len(t)
        block = t[pos:end].strip()
        if name and block:
            blocks.append((name, block))
    return blocks

def has_required_question_fields(qtext: str) -> bool:
    required = [
        "QUESTION FOR CLUSTER:",
        "TITLE:",
        "TYPE:",
        "QUESTION (one sentence):",
        "BACKGROUND",
        "CLUSTER CONTEXT",
        "FERMI STARTING POINT",
        "WHAT TO CONSIDER",
        "ANSWER GUIDANCE",
        "AMBIGUITY CHECKS",
    ]
    return all(r in (qtext or "") for r in required)

def split_question_blocks(qtext: str) -> List[str]:
    """
    Split Tier 4 output into question blocks starting with "QUESTION FOR CLUSTER:".
    """
    t = qtext or ""
    starts = [m.start() for m in re.finditer(r"^QUESTION FOR CLUSTER:\s*.+\s*$", t, flags=re.MULTILINE)]
    if not starts:
        return []
    blocks: List[str] = []
    for i, pos in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(t)
        blocks.append(t[pos:end].strip())
    return [b for b in blocks if b]

def count_question_blocks(qtext: str) -> int:
    return len(re.findall(r"^QUESTION FOR CLUSTER:\s*.+$", qtext or "", flags=re.MULTILINE))

def all_blocks_have_required_fields(qtext: str) -> bool:
    blocks = split_question_blocks(qtext)
    if not blocks:
        return False
    return all(has_required_question_fields(b) for b in blocks)

def looks_internal_metric_question(qtext: str) -> bool:
    """
    Heuristic: flag questions that require internal company data rather than public resolution.
    We allow "public company filings" but user wants SME/private typical; so bias hard against internal.
    """
    bad_phrases = [
        "company’s", "company's", "our", "customer base", "by arr", "arr", "nrr", "churn",
        "gross margin", "cac", "sales cycle", "internal", "audited", "go-live across customers",
        "net interest margin on embedded invoice financing", "new arr", "contribution margin attributable",
        "percentage of invoices processed in the prior 90 days across the company’s active customer base",
        "documented evidence", "at least 30% of the company",
    ]
    t = (qtext or "").lower()
    return any(p.lower() in t for p in bad_phrases)


# -----------------------------
# OpenRouter client (RAW)
# -----------------------------
def openrouter_chat(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    stream: bool,
    timeout: int = 120,
):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        # Optional but often recommended by OpenRouter:
        "HTTP-Referer": "http://localhost",
        "X-Title": "ForecastingBriefRAW",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": bool(stream),
    }
    r = requests.post(OPENROUTER_URL, headers=headers, data=json.dumps(payload), stream=stream, timeout=timeout)
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = {"error": r.text[:2000]}
        raise RuntimeError(f"OpenRouter HTTP {r.status_code}: {err}")

    if not stream:
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def gen():
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                chunk = line[len("data: "):].strip()
                if chunk == "[DONE]":
                    break
                try:
                    j = json.loads(chunk)
                    delta = j["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except Exception:
                    continue

    return gen()

def llm_raw(
    api_key: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    stream_to: Optional[st.delta_generator.DeltaGenerator] = None,
) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if stream_to is None:
        return openrouter_chat(api_key, model, msgs, temperature, max_tokens, stream=False)

    ph = stream_to.empty()
    acc = ""
    for part in openrouter_chat(api_key, model, msgs, temperature, max_tokens, stream=True):
        acc += part
        ph.text(acc)
    return acc


# -----------------------------
# PROMPTS (RAW)
# -----------------------------
SYSTEM_T1 = """You are a senior forecasting editor for professional forecasters.

Goal: From a short company description, produce a high-signal set of uncertainty axes and clarifying questions.

Hard rules:
- Output RAW TEXT only (no JSON, no markdown code blocks).
- Use the EXACT section markers below, in this exact order.
- Use concise bullets; avoid fluff and repetition.

Format (must match exactly):
=== TIER 1: AXES ===
AXES (15-30):
- ...
KEYWORDS (10-20):
- ...
IMPLICIT ASSUMPTIONS (5-12):
- ...
CLARIFYING QUESTIONS (3-8):
- ...
END TIER 1
"""

USER_T1 = """Company description:
{company}

Settings:
- Primary geography: {geo}
- Horizon: {horizon} months
- Risk appetite: {risk} (low/medium/high)

Write Tier 1 now.
"""

SYSTEM_T2 = """You are an industry research scout WITH WEB ACCESS.

This is a 3-call chained process:
- Call 1: establish sector map and key public drivers.
- Call 2: add new non-redundant details (do NOT repeat call 1).
- Call 3: add new non-redundant details (do NOT repeat calls 1–2).
Each call MUST incorporate the prior snapshot (do not lose information).

Important: The user DOES NOT CARE about collecting links; Tier 2 is only to enrich knowledge.
- Prefer factual, public, stable information (regulators, standards, official statistics, product documentation norms).
- You MAY include URLs, but only if you can keep them on one line; otherwise omit.
- Avoid hallucinated specifics (exact dates or requirements) unless you are confident they are public and stable.

Hard rules:
- Output RAW TEXT only (no JSON, no markdown code blocks).
- Use EXACT markers and structure below.

Format (must match exactly):
=== TIER 2 CALL {call_idx}/{call_total}: SECTOR RESEARCH ===
UPDATED SNAPSHOT (use this whole block for downstream tiers):
SECTOR DEFINITION (1-3 sentences):
...
VALUE CHAIN (5-10):
- ...
DEMAND DRIVERS (5-10):
- ...
SUPPLY CONSTRAINTS (3-8):
- ...
KEY METRICS (6-12):
- ...
RECENT TRENDS (5-10, dated when possible):
- ...
COMPETITOR ARCHETYPES (4-8):
- ...
REGULATORY / MACRO HOOKS (4-8):
- ...
PUBLIC RESOLUTION ANCHORS (5-12, no URLs required):
- Name the public dataset / regulator page / product doc type that would resolve key uncertainties.
- ...

END TIER 2 CALL {call_idx}
"""

USER_T2 = """You are Tier 2 CALL {call_idx} of {call_total}.

Company description:
{company}

Upstream Tier 1 output (reuse its keywords/axes; do not ignore):
{tier1_text}

Primary geography: {geo}
Horizon: {horizon} months

PRIOR UPDATED SNAPSHOT (from previous calls; extend it, do not lose detail):
{prior_snapshot}

Task:
- Extend and improve the UPDATED SNAPSHOT with NEW, non-redundant details.
- Keep it focused on public, decision-relevant factors.
Return in the required format only.
"""

SYSTEM_T3 = """You are a sectorization engine.

Goal: Convert Tier 1 + Tier 2 snapshot into 5–10 MECE-ish clusters that will each yield ONE forecasting question call (but 2–3 questions inside that call).

Hard rules:
- Output RAW TEXT only (no JSON, no markdown code blocks).
- Use EXACT markers/structure below.
- Each cluster must start with: "CLUSTER: <Name>" on its own line.
- Provide 5–10 clusters, decision-relevant, no duplicates.

Format (must match exactly):
=== TIER 3: CLUSTERS ===
CLUSTER: <Name 1>
SCOPE:
- includes: ...
- excludes: ...
WHY IT MATTERS:
...
LEADING INDICATORS (3-6):
- ...
TIME HORIZON (months): N

CLUSTER: <Name 2>
...

END TIER 3
"""

USER_T3 = """Company description:
{company}

Tier 1 output:
{tier1_text}

Tier 2 UPDATED SNAPSHOT:
{tier2_snapshot}

Settings:
- Primary geography: {geo}
- Horizon: {horizon} months
- Target clusters: {n_clusters}

Write Tier 3 now.
"""

# Tier 4: public-resolvable, decision-useful, one call per cluster, 2–3 questions per call
SYSTEM_Q = """You are a forecasting question writer for professional forecasters.

Generate EXACTLY 2 OR 3 distinct high-quality questions for the given cluster only.

PRIMARY CONSTRAINT (must follow):
- Each question MUST be publicly resolvable from public information (regulators, official statistics, standards bodies, product documentation, press releases, public policy documents).
- DO NOT ask about the company's internal metrics (ARR, churn, customer adoption %, margins, DSO impact in their customer base, NIM on their book, etc.).
- Use an external measurable proxy that is still decision-relevant for the company (regulatory timelines, number of accredited providers, ERP GA feature availability, official stats on instant payments, SME insolvency indices, lending standards surveys, major competitor product launches, etc.).

DIVERSITY REQUIREMENT:
- The 2–3 questions must NOT be near-duplicates: use different public proxies, datasets, or mechanisms per question.

Quality bar (for each question):
- Unambiguous: units, geography, deadline, what counts.
- Decision-useful: tie back to Strategy A (compliance/PDP) vs Strategy B (payments/finance) in Cluster Context.
- Include a Fermi starting point (base rate) with rationale.
- Include a PUBLIC resolution hook inside Answer Guidance (name the dataset/page/type; no URL required).

Hard rules:
- Output RAW TEXT only (no JSON, no markdown code blocks).
- Use the EXACT structure below for EACH question block, repeated 2–3 times.
- Do NOT add any wrapper text, headings, numbering, or commentary outside the blocks.
- Use the DEADLINE_DATE provided and do not exceed it.
- Allowed types: BINARY / NUMERIC / MULTIPLE_CHOICE.

Format (must match exactly, repeat 2–3 times):
QUESTION FOR CLUSTER: <Cluster Name>
TITLE:
TYPE: <BINARY|NUMERIC|MULTIPLE_CHOICE>
QUESTION (one sentence):
BACKGROUND (shared baseline, 3-6 sentences):
...
CLUSTER CONTEXT (2-4 sentences):
...
FERMI STARTING POINT (rough prior + rationale):
...
WHAT TO CONSIDER (5-8):
- ...
ANSWER GUIDANCE (units / interpretation / MC options):
...
AMBIGUITY CHECKS (3-5):
- ...
"""

USER_Q = """DEADLINE_DATE: {deadline_date}

Company description:
{company}

Tier 1 output (use it):
{tier1_text}

Tier 2 UPDATED SNAPSHOT (use it):
{tier2_snapshot}

CLUSTER_NAME: {cluster_name}

THIS CLUSTER (write questions ONLY for this cluster; use its scope/indicators):
{cluster_block}

Settings:
- Primary geography: {geo}
- Horizon: {horizon} months
- Risk appetite: {risk}

Write 2–3 question blocks now (no extra text).
"""


# -----------------------------
# Tier runners
# -----------------------------
def run_tier2_three_calls(
    api_key: str,
    sector_model_raw: str,
    company: str,
    tier1_text: str,
    geo: str,
    horizon: int,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> Tuple[str, str]:
    sector_model = ensure_online_model(sector_model_raw)
    prior_snapshot = "—"
    call_texts: List[str] = []

    for call_idx in (1, 2, 3):
        sys = SYSTEM_T2.format(call_idx=call_idx, call_total=3)
        usr = USER_T2.format(
            call_idx=call_idx,
            call_total=3,
            company=company,
            tier1_text=tier1_text,
            geo=geo,
            horizon=horizon,
            prior_snapshot=prior_snapshot,
        )

        txt = clean(llm_raw(
            api_key=api_key,
            model=sector_model,
            system=sys,
            user=usr,
            temperature=max(0.15, temperature),
            max_tokens=max_tokens,
            stream_to=st if stream else None,
        ))

        # update snapshot for next call
        snap = extract_between(
            txt,
            "UPDATED SNAPSHOT (use this whole block for downstream tiers):",
            f"END TIER 2 CALL {call_idx}",
        )
        if snap:
            prior_snapshot = snap.strip()

        call_texts.append(txt)

    tier2_full = "\n\n".join(call_texts).strip()
    final_snapshot = prior_snapshot.strip() if prior_snapshot else "—"
    return tier2_full, final_snapshot

def run_questions_per_cluster(
    api_key: str,
    question_model: str,
    company: str,
    tier1_text: str,
    tier2_snapshot: str,
    clusters: List[Tuple[str, str]],
    geo: str,
    horizon: int,
    risk: str,
    deadline_date: str,
    temperature: float,
    max_tokens: int,
    stream: bool = False,
) -> str:
    q_blocks: List[str] = []

    for name, block in clusters:
        usr = USER_Q.format(
            deadline_date=deadline_date,
            company=company,
            tier1_text=tier1_text,
            tier2_snapshot=tier2_snapshot,
            cluster_name=name,
            cluster_block=block,
            geo=geo,
            horizon=horizon,
            risk=risk,
        )

        raw = clean(llm_raw(
            api_key=api_key,
            model=question_model,
            system=SYSTEM_Q,
            user=usr,
            temperature=temperature,
            max_tokens=max_tokens,
            stream_to=st if stream else None,
        ))

        n_q = count_question_blocks(raw)
        blocks = split_question_blocks(raw)

        # Repair triggers:
        # - must be exactly 2 or 3 blocks
        # - each block must have required fields
        # - must avoid internal metrics
        needs_repair = (
            (n_q < 2 or n_q > 3) or
            (not blocks) or
            (not all_blocks_have_required_fields(raw)) or
            looks_internal_metric_question(raw)
        )

        if needs_repair:
            repair_sys = "You are a strict forecasting editor and format enforcer. Output RAW TEXT only."
            repair_user = f"""Rewrite the output so that:
1) It contains EXACTLY 2 OR 3 DISTINCT question blocks (not duplicates).
2) EACH block matches the EXACT required structure and includes all required fields.
3) ALL questions are PUBLICLY RESOLVABLE (no internal company metrics, no ARR/churn/customer-base measures).
4) EACH question remains decision-useful for Strategy A (compliance/PDP) vs Strategy B (payments/finance).
5) EACH question includes a PUBLIC resolution hook in Answer Guidance (dataset/page/type; no URL needed).
Use DEADLINE_DATE {deadline_date} and do not exceed it.
Cluster name: {name}

IMPORTANT CONTEXT (must be used, do not ignore):
Company description:
{company}

Tier 1 output:
{tier1_text}

Tier 2 UPDATED SNAPSHOT:
{tier2_snapshot}

Cluster block:
{block}

Faulty output:
{raw}
"""
            raw = clean(llm_raw(
                api_key=api_key,
                model=question_model,
                system=repair_sys,
                user=repair_user,
                temperature=0.2,
                max_tokens=min(2000, max_tokens),
                stream_to=None,
            ))

        q_blocks.append(raw)

    return "\n\n".join(q_blocks).strip()

def build_final_txt(
    company: str,
    geo: str,
    horizon: int,
    risk: str,
    deadline_date: str,
    t1: str,
    t2_full: str,
    t2_snapshot: str,
    t3: str,
    t4_questions: str,
) -> str:
    out = []
    out.append("FORECASTING BRIEF — RAW OUTPUT (Axes → Online sector research → Clusters → Public-resolvable questions)")
    out.append(f"Generated: {now_utc_str()}")
    out.append(f"Primary geography: {geo}")
    out.append(f"Horizon: {horizon} months")
    out.append(f"Risk appetite: {risk}")
    out.append(f"Deadline date (horizon cap): {deadline_date}")
    out.append("")
    out.append("=== INPUT COMPANY DESCRIPTION ===")
    out.append(wrap(company))
    out.append("")
    out.append(t1.strip())
    out.append("")
    out.append("=== TIER 2 (ALL 3 CALLS RAW OUTPUT) ===")
    out.append(t2_full.strip())
    out.append("")
    out.append("=== TIER 2 (FINAL UPDATED SNAPSHOT USED DOWNSTREAM) ===")
    out.append(t2_snapshot.strip() if t2_snapshot else "—")
    out.append("")
    out.append(t3.strip())
    out.append("")
    out.append("=== TIER 4: QUESTIONS (2–3 per cluster, ONE CALL PER CLUSTER, PUBLICLY RESOLVABLE) ===")
    out.append(t4_questions.strip())
    out.append("")
    return "\n".join(out)


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Forecasting Brief (RAW, public-resolvable questions)", layout="wide")
st.title("Forecasting brief (RAW) — Tier 2 = :online×3 chained, Tier 4 = 2–3 public-resolvable questions per cluster (1 call/cluster)")

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
    placeholder="Example: B2B SaaS for invoice-to-cash automation in EU; compliance (PDP/PPF), OCR, payments..."
)

colA, colB = st.columns([1, 1])
run_btn = colA.button("Generate brief", type="primary", use_container_width=True)
clear_btn = colB.button("Clear", use_container_width=True)

status = st.empty()
progress = st.progress(0)

if clear_btn:
    st.session_state.clear()
    st.rerun()

for k in ["final_txt", "t1", "t2_full", "t2_snapshot", "t3", "t4"]:
    if k not in st.session_state:
        st.session_state[k] = ""

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
        st.session_state.final_txt = ""

        g = ensure_nonempty_geo(geo)
        deadline = add_months(today_utc_date(), int(horizon))
        deadline_date = deadline.isoformat()

        progress.progress(0.08)
        status.info("Tier 1/4 — Axes…")
        t1 = clean(llm_raw(
            api_key=api_key,
            model=base_model,
            system=SYSTEM_T1,
            user=USER_T1.format(company=company, geo=g, horizon=horizon, risk=risk),
            temperature=temperature,
            max_tokens=max_tokens,
            stream_to=st if stream_debug else None,
        ))
        st.session_state.t1 = t1

        progress.progress(0.35)
        status.info("Tier 2/4 — Sector research (:online × 3 calls, chained; links optional)…")
        t2_full, t2_snapshot = run_tier2_three_calls(
            api_key=api_key,
            sector_model_raw=sector_model,
            company=company,
            tier1_text=t1,
            geo=g,
            horizon=horizon,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream_debug,
        )
        st.session_state.t2_full = t2_full
        st.session_state.t2_snapshot = t2_snapshot

        progress.progress(0.58)
        status.info("Tier 3/4 — Clusters…")
        t3 = clean(llm_raw(
            api_key=api_key,
            model=base_model,
            system=SYSTEM_T3,
            user=USER_T3.format(
                company=company,
                tier1_text=t1,
                tier2_snapshot=t2_snapshot,
                geo=g,
                horizon=horizon,
                n_clusters=n_clusters,
            ),
            temperature=temperature,
            max_tokens=max_tokens,
            stream_to=st if stream_debug else None,
        ))

        clusters = split_clusters(t3)
        if len(clusters) < 5:
            status.warning("Tier 3 produced too few clusters; repairing…")
            t3 = clean(llm_raw(
                api_key=api_key,
                model=base_model,
                system="You are a strict formatter and strategist. Output RAW TEXT only.",
                user=f"""Rewrite Tier 3 properly with 5–10 clusters and the required format.
Keep consistent with Tier 1 and Tier 2 snapshot. Target clusters: {n_clusters}

Faulty Tier 3:
{t3}
""",
                temperature=0.2,
                max_tokens=min(1700, max_tokens),
                stream_to=None,
            ))
            clusters = split_clusters(t3)

        st.session_state.t3 = t3

        progress.progress(0.72)
        status.info("Tier 4/4 — Questions (1 call per cluster; 2–3 PUBLICLY RESOLVABLE questions per call)…")
        t4 = run_questions_per_cluster(
            api_key=api_key,
            question_model=question_model,
            company=company,
            tier1_text=t1,
            tier2_snapshot=t2_snapshot,
            clusters=clusters,
            geo=g,
            horizon=horizon,
            risk=risk,
            deadline_date=deadline_date,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream_debug,
        )
        st.session_state.t4 = t4

        final_txt = build_final_txt(
            company=company,
            geo=g,
            horizon=horizon,
            risk=risk,
            deadline_date=deadline_date,
            t1=t1,
            t2_full=t2_full,
            t2_snapshot=t2_snapshot,
            t3=t3,
            t4_questions=t4,
        )
        st.session_state.final_txt = final_txt

        progress.progress(1.0)
        status.success("Done. Tier 4 is 2–3 public-resolvable questions per cluster (no internal company metrics).")
    except Exception as e:
        status.error(f"Error: {e}")
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
    with st.expander("Tier 1", expanded=False):
        st.text(st.session_state.t1)
    with st.expander("Tier 2 (all calls)", expanded=False):
        st.text(st.session_state.t2_full)
    with st.expander("Tier 2 (final snapshot)", expanded=False):
        st.text(st.session_state.t2_snapshot)
    with st.expander("Tier 3", expanded=False):
        st.text(st.session_state.t3)
    with st.expander("Tier 4 (questions)", expanded=False):
        st.text(st.session_state.t4)

