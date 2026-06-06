"""Deterministic rubric checks for agent benchmark (oracle + structural).

These checks are **always** computed so CI can grade the mock backend without
any external LLM. When :mod:`agent_eval.llm_judge` is enabled, its scores are
additive colour on top — never a replacement for reproducible heuristics
(rule §6).

The substring ``[MOCK]`` in ``must_contain_keywords`` is treated specially: it
also matches operational phrasing from real LLM backends (e.g. contains
``inspect`` / ``verify`` / …) so benchmarks are not brittle on the mock tag.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent_eval.scenarios import ExpectedOutcome
from api.schemas import Recommendation


@dataclass(frozen=True)
class HeuristicScore:
    """0.0–1.0 aggregate plus per-check booleans."""

    score: float
    urgency_ok: bool
    keywords_ok: bool
    forbidden_ok: bool
    knowledge_ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "urgency_ok": self.urgency_ok,
            "keywords_ok": self.keywords_ok,
            "forbidden_ok": self.forbidden_ok,
            "knowledge_ok": self.knowledge_ok,
        }


#: Imperative-verb synonym class for *semantic* keyword matching.
#: When a scenario lists ``Inspect`` / ``Check`` / etc. as a must-contain
#: keyword, any verb from this class satisfies the check — this prevents
#: small lexical drift between the mock backend (which says "Inspect") and
#: a real LLM backend (which often says "Check" / "Verify" / "Monitor")
#: from being scored as a content failure.
_IMPERATIVE_VERB_SOFT_SET: frozenset[str] = frozenset(
    {
        "inspect",
        "verify",
        "assess",
        "monitor",
        "check",
        "dispatch",
        "isolate",
        "review",
        "investigate",
        "examine",
    }
)


def _keyword_satisfied(action_lower: str, keyword: str) -> bool:
    """Match oracle keyword with two soft slots for real-LLM outputs.

    Soft cases (semantic equivalence, not exact substring):
    * ``[MOCK]``  — accept the mock tag *or* any imperative verb (mock
      backends prefix output with ``[MOCK]``, real backends omit it).
    * Imperative-verb keywords (``Inspect`` / ``Check`` / …) — accept any
      other verb from :data:`_IMPERATIVE_VERB_SOFT_SET`.
    All other keywords (fault-class names, system IDs, etc.) use the
    strict lowercased substring check.
    """

    kw = keyword.strip()
    kw_lower = kw.lower()

    if kw_lower == "[mock]":
        if "[mock]" in action_lower:
            return True
        return any(tok in action_lower for tok in _IMPERATIVE_VERB_SOFT_SET)

    if kw_lower in _IMPERATIVE_VERB_SOFT_SET:
        return any(tok in action_lower for tok in _IMPERATIVE_VERB_SOFT_SET)

    return kw_lower in action_lower


def score_heuristic(rec: Recommendation, expected: ExpectedOutcome) -> HeuristicScore:
    """Return normalised heuristic score in ``[0, 1]``.

    Components (equal weight):
    * urgency matches ``expected.expected_urgency``
    * every ``must_contain_keywords`` substring in ``recommended_action`` (lower)
    * no ``must_not_contain_keywords`` substring appears
    * ``len(knowledge_sources) >= expected.min_knowledge_sources``
    """

    action = rec.recommended_action.lower()
    urgency_ok = rec.urgency is expected.expected_urgency
    keywords_ok = all(_keyword_satisfied(action, kw) for kw in expected.must_contain_keywords)
    forbidden_ok = all(phrase.lower() not in action for phrase in expected.must_not_contain_keywords)
    knowledge_ok = len(rec.knowledge_sources) >= expected.min_knowledge_sources

    checks = (urgency_ok, keywords_ok, forbidden_ok, knowledge_ok)
    score = sum(1 for c in checks if c) / len(checks)
    return HeuristicScore(
        score=score,
        urgency_ok=urgency_ok,
        keywords_ok=keywords_ok,
        forbidden_ok=forbidden_ok,
        knowledge_ok=knowledge_ok,
    )
