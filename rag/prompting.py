"""Prompt rendering for the cloud agent.

For MVP we keep prompts in Python strings rather than separate ``.j2`` files
to minimize moving parts. The :class:`PromptBuilder` exposes one public
method per prompt the agent uses; future polish-phase prompts can move to
files in ``rag/prompts/`` without changing call sites.
"""
from __future__ import annotations

from collections.abc import Iterable

from jinja2 import Environment, StrictUndefined

from api.schemas import Alert
from rag.retrieval import RetrievedChunk

# StrictUndefined raises on missing variables — catches typos in prompt rendering.
_JINJA_ENV = Environment(undefined=StrictUndefined, autoescape=False)


_RECOMMENDATION_TEMPLATE = _JINJA_ENV.from_string(
    """You are AgentPV, a fault-response assistant for solar PV / BESS sites.

# Alert
- system_id: {{ alert.system_id }}
- system_type: {{ alert.system_type.value }}
- fault_class: {{ alert.fault_class }}
- severity: {{ alert.severity.value }}
- confidence: {{ "%.2f"|format(alert.confidence) }}

# Latest sensor snapshot
{% for k, v in alert.sensor_snapshot.items() %}- {{ k }}: {{ v }}
{% endfor %}

# Retrieved knowledge ({{ chunks|length }} chunk(s))
{% for c in chunks %}
## [{{ loop.index }}] {{ c.chunk.title }} — {{ c.chunk.section or "intro" }}
({{ c.chunk.source }}, score={{ "%.3f"|format(c.score) }})
{{ c.chunk.text }}
{% endfor %}

# Task
Produce a single concrete recommended action that an on-call operator can
execute within the time horizon dictated by the severity. Cite the
retrieved chunks by their numeric ids in square brackets. Do not invent
sources or numbers.
"""
)


class PromptBuilder:
    """Render prompts from typed inputs."""

    def render_recommendation_prompt(
        self,
        alert: Alert,
        chunks: Iterable[RetrievedChunk],
    ) -> str:
        chunks_list = list(chunks)
        return _RECOMMENDATION_TEMPLATE.render(alert=alert, chunks=chunks_list)
