from forecast_brief.utils import now_utc_str, wrap


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
    return "\n".join(
        [
            "FORECASTING BRIEF — RAW OUTPUT (Axes → Online sector research → Clusters → Public-resolvable questions)",
            f"Generated: {now_utc_str()}",
            f"Primary geography: {geo}",
            f"Horizon: {horizon} months",
            f"Risk appetite: {risk}",
            f"Deadline date (horizon cap): {deadline_date}",
            "",
            "=== INPUT COMPANY DESCRIPTION ===",
            wrap(company),
            "",
            t1.strip(),
            "",
            "=== TIER 2 (ALL 3 CALLS RAW OUTPUT) ===",
            t2_full.strip(),
            "",
            "=== TIER 2 (FINAL UPDATED SNAPSHOT USED DOWNSTREAM) ===",
            t2_snapshot.strip() if t2_snapshot else "—",
            "",
            t3.strip(),
            "",
            "=== TIER 4: QUESTIONS (2–3 per cluster, ONE CALL PER CLUSTER, PUBLICLY RESOLVABLE) ===",
            t4_questions.strip(),
            "",
        ]
    )
