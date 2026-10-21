"""Render an eval run as markdown.

The JSON report is the machine-readable record; this is the one a human reads when a number
moves. It leads with the configuration, because the first question about any changed metric is
"what else changed?".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from evals.runners.run_rag import RunReport


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render_markdown(report: RunReport) -> str:
    lines: list[str] = []
    add = lines.append

    add(f"# Eval report — `{report.profile.name}`")
    add("")
    add(f"> {report.profile.description}")
    add("")
    add(
        f"{report.started_at:%Y-%m-%d %H:%M UTC} · {len(report.rows)} rows · "
        f"{report.duration_seconds:.0f}s · ${report.total_cost_usd:.4f}"
    )
    add("")

    add("## Configuration")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| index fingerprint | `{report.index_fingerprint}` |")
    add(f"| index slot | {report.profile.index_version} |")
    add(
        f"| chunking | {report.chunk_tokens} tokens, {report.chunk_overlap} overlap, "
        f"chunker v{report.chunker_version} |"
    )
    add(f"| embedding model | `{report.embedding_model}` |")
    add(f"| retrieval | vector-only, k={report.profile.top_k} |")
    add(f"| answer model | `{report.answer_model}` |")
    add(f"| prompt versions | {', '.join(f'{k}={v}' for k, v in report.prompt_versions.items())} |")
    add("")

    add("## Retrieval")
    add("")
    add(
        f"| Recall@{report.profile.top_k} | **Context recall** | Hit rate@{report.profile.top_k} "
        "| MRR | Context precision |"
    )
    add("|---|---|---|---|---|")
    add(
        f"| {_pct(report.retrieval.recall_at_k)} | **{_pct(report.retrieval.context_recall)}** | "
        f"{_pct(report.retrieval.hit_rate_at_k)} | "
        f"{report.retrieval.mrr:.3f} | {_pct(report.retrieval.context_precision)} |"
    )
    add("")
    add(
        f"The generator received **{report.retrieval.mean_context_units:.1f} passages per row** "
        f"on average, from {report.profile.top_k} retrieved chunks."
    )
    add("")
    add(
        "**Context recall is the number to compare across chunking schemes.** Recall@k is "
        "measured on the retrieved units, so it moves with their size: halve the chunk and the "
        "same retriever returns less text under the same k, which reads as a regression and is "
        "partly an artefact. Context recall is measured on the passages actually sent to the "
        "model after parent expansion — the text the answer depends on — and means the same "
        "thing whether the index is flat or parent-child."
    )
    add("")
    add(
        "Recall is quote-level: of the quotes a question cites, the fraction that appear in some "
        "retrieved chunk. Hit rate is the weaker question of whether *anything* relevant came "
        "back. Context precision is the share of retrieved chunks that held evidence — the cost "
        "side, since every chunk is context the model pays to read."
    )
    add("")

    add("## Generation")
    add("")
    if report.judges:
        add("| Judge | Mean | Rows scored | Failed | Prompt |")
        add("|---|---|---|---|---|")
        for judge in report.judges:
            add(
                f"| {judge.name} | {judge.mean_score:.3f} | {judge.scored_rows} | "
                f"{judge.failed_rows} | {judge.prompt_version} |"
            )
        add("")
        add(f"_{report.judge_caveat}_")
    else:
        add("No judges were run.")
    add("")

    add("## Grounding (mechanical)")
    add("")
    add(
        "A deterministic cross-check on the faithfulness judge: a digit figure in an answer "
        "either appears in the passages it was given or it does not. It cannot be talked round, "
        "which the judge can — but absence alone is a question, not a verdict, because this "
        "corpus has a ledger whose totals are arithmetic and a correct answer computes figures "
        "that appear nowhere in the text."
    )
    add("")
    add("| | Rows |")
    add("|---|---|")
    add(
        f"| asserting a figure absent from the context | {report.rows_with_unsupported_figures}"
        f"/{len(report.rows)} |"
    )
    add(
        f"| ...**and** missing some of their cited evidence | "
        f"{report.rows_with_unverifiable_figures}/{len(report.rows)} |"
    )
    add(f"| figure support rate | {_pct(report.figure_support_rate)} |")
    add("")

    unverifiable = [r for r in report.rows if r.has_unverifiable_figure]
    if unverifiable:
        add("### Unaccountable figures")
        add("")
        add(
            "A figure neither present in the retrieved passages nor derivable from them had to "
            "come from somewhere else. **Compare the last column.**"
        )
        add("")
        add(f"| Row | Category | Recall@{report.profile.top_k} | Figures | Judge faithfulness |")
        add("|---|---|---|---|---|")
        for row in unverifiable:
            faith = row.judges.score("faithfulness")
            shown = f"{faith:.2f}" if faith is not None else "—"
            add(
                f"| `{row.id}` | {row.category} | {_pct(row.retrieval.recall)} | "
                f"{', '.join(row.grounding.unsupported)} | {shown} |"
            )
        add("")

    arithmetic = [
        r for r in report.rows if not r.grounding.grounded and not r.has_unverifiable_figure
    ]
    if arithmetic:
        add("### Figures absent but accountable (probably arithmetic)")
        add("")
        add(
            "Every cited quote for these rows was retrieved, so the figure is most likely "
            "computed from the passages rather than invented. Listed for completeness, not as "
            "a finding."
        )
        add("")
        add("| Row | Figures |")
        add("|---|---|")
        for row in arithmetic:
            add(f"| `{row.id}` | {', '.join(row.grounding.unsupported)} |")
        add("")

    add("## Refusal")
    add("")
    add("| | Rows | Rate |")
    add("|---|---|---|")
    add(
        f"| correctly refused (unanswerable) | {report.refusal.correctly_refused}"
        f"/{report.refusal.unanswerable_rows} | {_pct(report.refusal.refusal_recall)} |"
    )
    add(
        f"| falsely refused (answerable) | {report.refusal.falsely_refused}"
        f"/{report.refusal.answerable_rows} | {_pct(report.refusal.false_refusal_rate)} |"
    )
    add("")
    add(
        "Two numbers, never one: a model that refuses everything scores perfectly on the first "
        "row and terribly on the second, and a model that answers everything does the reverse."
    )
    add("")

    add("## By category")
    add("")
    add(
        f"| Category | Rows | Recall@{report.profile.top_k} | MRR | Refusal correct | "
        "Faithfulness | Correctness |"
    )
    add("|---|---|---|---|---|---|---|")
    for cat in report.by_category:
        faith = f"{cat.faithfulness:.2f}" if cat.faithfulness is not None else "—"
        correct = f"{cat.correctness:.2f}" if cat.correctness is not None else "—"
        add(
            f"| {cat.category} | {cat.rows} | {_pct(cat.recall_at_k)} | {cat.mrr:.3f} | "
            f"{_pct(cat.refusal_correct)} | {faith} | {correct} |"
        )
    add("")

    add("## Cost and latency")
    add("")
    add("| | |")
    add("|---|---|")
    add(f"| total | ${report.total_cost_usd:.4f} |")
    add(f"| per row | ${report.total_cost_usd / max(len(report.rows), 1):.5f} |")
    add(f"| answer latency p50 | {report.latency_p50_ms:.0f} ms |")
    add(f"| answer latency p95 | {report.latency_p95_ms:.0f} ms |")
    add("")
    add("| tokens | input | output |")
    add("|---|---|---|")
    add(f"| answer calls | {report.answer_input_tokens:,} | {report.answer_output_tokens:,} |")
    add(f"| judge calls | {report.judge_input_tokens:,} | {report.judge_output_tokens:,} |")
    add(f"| **total** | **{report.total_input_tokens:,}** | **{report.total_output_tokens:,}** |")
    add("")
    cached_share = report.total_cached_tokens / max(report.total_input_tokens, 1)
    add(
        f"{report.total_cached_tokens:,} input tokens ({cached_share:.1%}) were served from the "
        "provider's prefix cache at the cached rate. Two runs doing identical work can differ "
        "substantially on cost from this alone, so read a cost delta against this number before "
        "reading it as a regression."
    )
    add("")

    worst = sorted(
        report.rows,
        key=lambda r: (
            r.judges.score("faithfulness") if r.judges.score("faithfulness") is not None else 1.0,
            r.retrieval.recall,
        ),
    )[:10]
    add("## Lowest-faithfulness rows")
    add("")
    add(
        "Where to look first. Low faithfulness with low recall is a retrieval problem; low "
        "faithfulness with high recall is the model ignoring what it was given."
    )
    add("")
    add(f"| Row | Category | Recall@{report.profile.top_k} | Faith | Refused | Dropped cites |")
    add("|---|---|---|---|---|---|")
    for row in worst:
        faith = row.judges.score("faithfulness")
        add(
            f"| `{row.id}` | {row.category} | {_pct(row.retrieval.recall)} | "
            f"{faith:.2f} | {row.refused} | {row.dropped_citations or '—'} |"
            if faith is not None
            else f"| `{row.id}` | {row.category} | {_pct(row.retrieval.recall)} | — | "
            f"{row.refused} | {row.dropped_citations or '—'} |"
        )
    add("")
    return "\n".join(lines) + "\n"
