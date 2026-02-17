from forecast_brief.pipeline import run_questions_per_cluster, run_tier2_three_calls


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return self.responses.pop(0)


def test_run_tier2_three_calls_uses_latest_snapshot():
    responses = [
        "=== TIER 2 CALL 1/3: SECTOR RESEARCH ===\nUPDATED SNAPSHOT (use this whole block for downstream tiers):\nA\nEND TIER 2 CALL 1",
        "=== TIER 2 CALL 2/3: SECTOR RESEARCH ===\nUPDATED SNAPSHOT (use this whole block for downstream tiers):\nB\nEND TIER 2 CALL 2",
        "=== TIER 2 CALL 3/3: SECTOR RESEARCH ===\nUPDATED SNAPSHOT (use this whole block for downstream tiers):\nC\nEND TIER 2 CALL 3",
    ]
    fake = FakeLLM(responses)
    full, snapshot = run_tier2_three_calls(fake, "k", "model", "co", "tier1", "EU", 12, 0.2, 1000)
    assert snapshot == "C"
    assert "CALL 1/3" in full and "CALL 3/3" in full


def test_run_questions_per_cluster_repairs_bad_output():
    bad = "QUESTION FOR CLUSTER: A\nTITLE:\nTYPE: BINARY\nQUESTION (one sentence):\n"
    good = """QUESTION FOR CLUSTER: A
TITLE:
TYPE: BINARY
QUESTION (one sentence):
BACKGROUND (shared baseline, 3-6 sentences):
CLUSTER CONTEXT (2-4 sentences):
FERMI STARTING POINT (rough prior + rationale):
WHAT TO CONSIDER (5-8):
- x
ANSWER GUIDANCE (units / interpretation / MC options):
AMBIGUITY CHECKS (3-5):
- x
QUESTION FOR CLUSTER: A
TITLE:
TYPE: NUMERIC
QUESTION (one sentence):
BACKGROUND (shared baseline, 3-6 sentences):
CLUSTER CONTEXT (2-4 sentences):
FERMI STARTING POINT (rough prior + rationale):
WHAT TO CONSIDER (5-8):
- x
ANSWER GUIDANCE (units / interpretation / MC options):
AMBIGUITY CHECKS (3-5):
- x"""
    fake = FakeLLM([bad, good])
    result = run_questions_per_cluster(
        fake,
        "k",
        "m",
        "co",
        "t1",
        "snapshot",
        [("A", "CLUSTER: A")],
        "EU",
        12,
        "medium",
        "2030-01-01",
        0.2,
        1000,
    )
    assert "TYPE: NUMERIC" in result
    assert len(fake.calls) == 2
