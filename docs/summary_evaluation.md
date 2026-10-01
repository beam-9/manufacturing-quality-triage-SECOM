# Summary evaluation

Evaluation uses 20 fixed unit cases at evenly spaced validation score ranks and five validation batch capacities (10%–50%). Selection uses scores, not known outcomes. This is a prompt-development/evaluation set, not an independent generalization benchmark. Historical labels are absent from the evidence supplied to Ollama.

Earlier free-prose versions exposed two issues: an over-broad lexical validator rejected harmless descriptions, and schema-valid prose could still make dimensionally incorrect claims (for example comparing log-odds contributions with measurement percentiles). Development attempts are preserved in `artifacts/summary_development_*`; local SQLite retains failed attempts. Some version 1.2 outputs also contained unsupported ranking claims and a Unicode fraction that bypassed a digit-only rule. These outputs are not used by the final demo.

**Version 2.1 is a constrained report composer, not free-form prose generation.** Ollama selects and orders short finding codes/evidence references/checks; Python renders fixed interpretation wording. Each finding has a machine-checkable predicate: selected status, contribution sign, missing input, value outside its training 5th–95th percentile band, or batch policy. Every cited fact must support the selected predicate. Generic context wording is available when evidence is insufficient. Python computes and renders numerical facts. Short codes reduce generation latency compared with v2.0's repeated full sentences. This trades linguistic variety for verifiable semantics, while preserving local LLM integration and workflow automation.

Final measured counts, model/runtime versions, latencies and fallbacks are saved in `artifacts/summary_evaluation.json` after the live evaluation finishes. Per-case evidence, output and automatic checks are in `summary_evaluation_cases.json`. Automated schema/reference/scope validity does not prove semantic fidelity. A human must review each draft before relying on it.

## Semantic-review rubric

For each case, check the cited evidence supports the interpretation, no unobserved quantity or physical meaning is asserted, model contribution is not treated as causality, the population matches, uncertainty is retained, and suggested checks are useful and allowed. Mark unsupported statements as rejected even if automatic checks passed. Agent-assisted review is recorded separately from human approval; generated drafts remain unreviewed for the end user.

## Failure behavior

- Invalid JSON, unknown references, inconsistent scope, unsupported numeric/physical/causal claims: one repair attempt, then labeled template.
- Missing local server or model: immediate labeled template, preserving the error.
- Timeout: bounded request, one retry/repair, then template; the total two-attempt workflow can exceed a single timeout.
- Hosted mode: saved evidence-matching draft or transparent template; no model calls or persistent review writes.
- Changed population/capacity/model evidence: different hash; no stale brief reuse.

The workflow is a constrained report generator, not an open-ended chatbot or an autonomous quality agent. Validity/fallback rates and latency are measured only for the evaluated fixtures, not claimed universally.
