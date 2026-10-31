# Heuristic mapping annex — normative for the heuristic scorer

Heuristic mapping version: heuristic-mapping-v1

This annex is the normative description of the heuristic scorer in
`panel_judge/core.py` (`heuristic_scores`, `_risk_register`,
`_market_set`, and the strength/weakness display fallback in
`synthesise`). It is documentation, not code, and is not imported at
runtime; the code constant `HEURISTIC_MAPPING_VERSION` is the runtime
version. The annex header version, the code constant, and
`verdict.state.heuristic_mapping_version` on heuristic runs must agree
or the build fails its own consistency check.

Version rule: `heuristic-mapping-v<n>`, a single integer counter
starting at v1 for the v1.0 annex (the v0.2 mapping was unversioned;
v1 records the v0.2 scoring behaviour unchanged — no pattern, threshold,
cap, token list, or rule below changed during the v1.0 implementation).
The version bumps by 1 whenever ANY of the following changes: a feature
pattern or word list for any criterion; a count threshold or score cap
for any criterion; the evidence numbers rule; the risk risk-plus-response
rule; the market-token list used for same-label dissent detection; the
risk-register extraction rules (sentence splitting, fatal-risk test); or
any fallback that derives a strength/weakness display from scores. A
change to a code comment, a reason string's wording that does not change
which score is produced, or a pure refactor that produces identical
scores on the full test corpus does NOT bump the version — but that
identical-scores claim must be proven by running the fixture + synthetic
corpus before and after, and recorded in the merge notes. When the
version bumps, `tool_version` must also bump by at least a PATCH, so a
heuristic verdict's provenance (`tool_version` +
`heuristic_mapping_version`) always identifies its mapping.

## Standing limitation (verbatim in substance)

The scorer rewards specificity and structure; it cannot tell whether a
cited precedent is real or a regulation actually applies; it is a
reproducible baseline, not a substitute for human judgement.

## Counting basis

Counts are regex matches (`re.findall`, case-insensitive) over the
answer text as judged (post-truncation). Numbers are counted by the
`_NUM_RE` pattern: a digit run optionally followed by a decimal part and
optionally a unit (`%`, `million`, `billion`, `£`, `$`, `€`).

## Per-criterion signals and count → score tables

### evidence_strength — signals: `_NUM_RE` numeric tokens + `_FEATURES["evidence_strength"]`

Word pattern (by name in code, `evidence_strength` feature regex):
`sold|sales|priced|price|from $|from £|from €|million|billion|per year|units|models? fell|shrinking|study|trial|report|named|precedent`
(plus the numeric-token count, which is a separate signal).

Numbers rule (dominates): numbers ≥ 4 → 4; numbers ≥ 2 → 3; numbers
≥ 1 → 2; numbers = 0 → 1. Then: if evidence-word matches ≥ 3, add 1,
capped at 5. Evidence words can therefore add at most 1, and only on
top of the numbers-based score. There is no path to 0 for
evidence_strength under this mapping (an answer with no numbers and no
evidence words scores 1).

### capital_efficiency — signals: `_FEATURES["capital_efficiency"]`

Word pattern: `bought-in|bought in|outsource|outsourc|build-to-order|confirmed orders?|deposits?|low-volume|low volume|few hundred|small|disciplined|avoid|instead of developing|low capital|direct|asset-light`.

Count → score: ≥ 6 → 4; ≥ 4 → 3; ≥ 2 → 2; ≥ 1 → 1; 0 → 0. Cap 4 (no
path to 5 under this mapping).

### regulatory_runway — signals: `_FEATURES["regulatory_runway"]`

Word pattern (generic regulatory language only — NO domain-specific
regulation names, so the scorer is not biased toward one panel's subject
matter): `regulat|exemption|homologation|legal|legally|route to (market|sale)|ban|tolerant markets?`.

Count → score: same table as capital_efficiency (≥ 6 → 4; ≥ 4 → 3;
≥ 2 → 2; ≥ 1 → 1; 0 → 0). Cap 4.

### differentiation — signals: `_FEATURES["differentiation"]`

Word pattern: `scarcity|unique|distinct|exclusive|heritage|analogue|analog|emotional|collectib|rarity|premium|driver-focused|purist|last|honest|craftsmanship|distinctive|niche`.

Count → score: same table (≥ 6 → 4; ≥ 4 → 3; ≥ 2 → 2; ≥ 1 → 1; 0 → 0).
Cap 4.

### risk_mitigation — signals: `_FEATURES["risk_mitigation"]` + risk-plus-response rule

Word pattern: `risk|biggest risk|mitigat|fallback|pivot|cap(?:ped)? output|tolerant markets?|future fuels?|hybrid-ready|staged|validate|second market|diversif`.
Base count → score uses the same table (≥ 6 → 4; ≥ 4 → 3; ≥ 2 → 2;
≥ 1 → 1; 0 → 0). Risk-plus-response rule, applied after the table:
a lone "risk" mention without a response word caps the score at 1;
risk + response lifts the score to at least 3 (within the table, i.e.
`max(table_score, 3)` when both are present). Response words are
searched answer-wide (not sentence-local): `pivot|fallback|mitigat|cap(?:ped)?|deposit|second market|hybrid-ready|staged|validate`.

## Market-token list (used for same-label dissent detection, not scoring)

Full list, in code order (`_MARKET_TOKENS`): `united states`,
`middle east`, `europe`, `australia`, `asia`, `japan`, `africa`,
`united kingdom`, `california`, `china`, `india`, `brazil`,
`southeast asia`, `north america`. Matching is case-insensitive
substring matching against the answer text as judged. A same-label
panelist is material dissent only when its market set is NON-EMPTY and
disjoint from the comparison group's market set; an empty set is never
dissent (regression: empty-market false dissent, fixed in v0.2).

## Risk register (`_risk_register`)

One entry per risk-naming sentence in the group's answers. Sentences
are split on `(?<=[.!?])\s+` (sentence-ending punctuation followed by
whitespace). A sentence counts as risk-naming when it contains "risk"
(case-insensitive substring). `response_in_answers` is the first
sentence in the WHOLE answer (answer-wide search, not sentence-local)
containing a response word (see the risk response list above), or
`null` if none. `fatal` is `true` only conservatively: the risk-naming
sentence itself must match the removal pattern (removes the legal route
/ sole supply / core market — code `_REMOVAL_RE`) AND the same sentence
must contain the words "no pivot". The "no pivot" test is
sentence-local; the response search is answer-wide. The DESIGN fatal
definition ("no pivot stated in the answers") is implemented as this
sentence-local test, which is the conservative reading: a pivot stated
elsewhere in the answer does not by itself make the sentence non-fatal
unless the sentence states it, and the register still records the
answer-wide response separately. Duplicate risk sentences across group
members are merged, with `named_by` accumulating panelist IDs. Empty
array when no risk is named.

## Strength / weakness display fallback

When a supplied assessment omits `strengths`/`weaknesses` (or a
heuristic per-panelist entry carries empty lists), the verdict derives
the display from scores only: strengths = criterion IDs scored ≥ 4,
weaknesses = criterion IDs scored ≤ 2. This fallback never invents
prose: it lists criterion IDs. A supplied non-empty strengths or
weaknesses array is used verbatim. An empty derived list renders in
the report as "none noted".

## Alias table (frozen in v1.0, recorded separately)

The short-criterion alias table used by the supplied route
(`evidence` → `evidence_strength`, `efficiency` →
`capital_efficiency`, `regulatory` → `regulatory_runway`,
`differentiation` → `differentiation`, `risk_mitigation` →
`risk_mitigation`) is versioned as `alias-table-v1`, recorded in state
as `alias_table_version` on supplied runs. It is frozen in v1.0: it is
NOT versioned with this heuristic annex, because
`heuristic_mapping_version` is `null` on supplied runs — exactly where
the alias table is used. Any change to the alias table requires a
design amendment that also bumps `tool_version` MINOR. An alias not in
this table is an unknown criterion, not a guess.

## Comparability

Heuristic scores are comparable only between runs with the same
`heuristic_mapping_version` (and the same `tool_version`,
`rubric_version`, panel, and budget/truncation state). Never compare
an `overall_score` across different `rubric_version`,
`tool_version`, or `heuristic_mapping_version` values, or across
different judging routes or assessors (heuristic 3.4 vs supplied 4.6
on the same panel are scores from different judges under different
limits, not comparable quality measures).
