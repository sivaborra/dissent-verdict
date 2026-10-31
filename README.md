# dissent-verdict

Panel JSON in. Rubric-scored verdict and report out. Dissent preserved, never averaged. Humans sign off.

Panel Judge scores competing answers to one question against the same stated
rubric, preserves dissent instead of averaging it away, and leaves the final
decision to a human who signs off. It is a generic, file-based tool — panels in,
a verdict and a report out.

## Install

```sh
pip install .
```

Python 3.10+. No other dependencies.

## Run

Validate a panel, then judge it with exactly one route:

```sh
python3 -m panel_judge validate --panel my-panel.json
python3 -m panel_judge judge --panel my-panel.json --heuristic --out-dir out/my-panel/
python3 -m panel_judge judge --panel my-panel.json --assessments my-assessments.json --out-dir out/my-panel/
python3 -m panel_judge judge --panel my-panel.json --ollama-model qwen3:14b --out-dir out/my-panel/
```

Each run writes `verdict.json` and `report.md` to the output directory.
`--heuristic` is a deterministic rule-based route. `--assessments` synthesises
a verdict from assessments you supply. `--ollama-model` judges each panelist
with a local Ollama model, saving each assessment atomically as it passes, so
a run can resume instead of restarting.

Without a human sign-off, a verdict is a draft, not a decision. Sign-off
re-derives the verdict from its recorded inputs and refuses a hand-edited one.

Every report carries this disclaimer:

> "Scores rate the answers against the stated criteria. They are not a guarantee of success, market outcome, or regulatory outcome."

## Customize

**Your panel.** A panel JSON file: a question, panelists with written answers,
and a rubric. The default rubric scores five criteria 0–5, each weighted 0.2:
`evidence_strength`, `capital_efficiency`, `regulatory_runway`,
`differentiation`, `risk_mitigation`. Supply your own rubric to change the
criteria and weights. Before any verdict, each answer is scored against three
checkpoints: evidence, feasibility, and risk.

**Your verdict shape.** Grouped strategies take the per-criterion maximum
across their members, never an average. The verdict is `majority`,
`plurality`, or `no-consensus` — no-consensus is a real verdict: the panel
did not agree, and a human must resolve it. Dissent stays in the report with
its scores, reasons, and fallback.

**Your judge.** Any judge that produces assessments in the assessment format
(panelist id, scores 0–5 with reasons per criterion, plus provenance) plugs
in through `--assessments` without touching the core.

## Keys and models

- Local Ollama: no key. Judge models of 14b parameters and above are
  recommended; 8b models complete but score weakly and are not recommended
  for production use.
- Cloud: any provider key, supplied through the environment only — for
  example `OPENROUTER_API_KEY`. A key is never in code, never printed, and
  never stored in verdict state.

## Add a judge — including Jev

The assessment format is the whole interface: write a small adapter that
produces assessments, and the core does the rest.

Jev is a decision-only model: it returns a typed choice, score, or
probability with a confidence value, and no rationale. The adapter pattern
for it is: ask Jev one typed question per rubric criterion, map each
score plus its confidence into that criterion's assessment entry, use
high-confidence Jev scores as a fast pre-judge, and escalate low-confidence
criteria to a reasoning judge (Ollama 14b+ locally, or a cloud model via
any provider key), which supplies the reasons the report requires.

This is an adapter pattern only. No Jev code and no Jev key are included
in this repository, and a Jev-only assessment — scores and confidence,
no reasons — must never be presented as a fully reasoned verdict on its own.

## License

MIT — anyone may use it freely, with any key or any local model.
