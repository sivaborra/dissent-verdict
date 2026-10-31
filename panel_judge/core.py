#!/usr/bin/env python3
"""Core judging logic for the generic panel-judge library.

Pure Python standard library. No network, no model, no live data.
The judge only sees the answers in the panel file (and, for the
supplied-assessments route, the scores a separate assessor wrote).

See DESIGN.md for the full specification this implements.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

try:  # Single source of truth: pyproject.toml; __init__ falls back to its constant.
    from . import __version__ as _PKG_VERSION
except Exception:  # pragma: no cover - direct file import fallback
    _PKG_VERSION = "0.2.0"
TOOL_VERSION = f"panel-judge {_PKG_VERSION}"

# v1.0 heuristic-mapping annex version (docs/HEURISTIC-MAPPING.md).
# Scoring behaviour is identical to v0.2 core.py; the annex is v1 because
# the v0.2 mapping was unversioned.
HEURISTIC_MAPPING_VERSION = "heuristic-mapping-v1"
# The short-criterion alias table is FROZEN in v1.0 (see DESIGN §5 / §1.3):
# any change requires a design amendment that also bumps tool_version MINOR.
ALIAS_TABLE_VERSION = "alias-table-v1"

# Fixed canonical checkpoint mapping (DESIGN §5.2, immutable).
CANONICAL_CHECKPOINT_MAPPING = {
    "evidence_strength": "evidence",
    "differentiation": "evidence",
    "capital_efficiency": "feasibility",
    "regulatory_runway": "feasibility",
    "risk_mitigation": "risk",
}
# Verdict top-level keys that a reader must never invent or drop (G22:
# order is presentation only; presence is normative).
VERDICT_TOP_LEVEL_KEYS = {
    "panel_id", "question", "verdict", "rubric", "checkpoints",
    "checkpoint_disagreements", "per_panelist", "dissent", "disclaimer",
    "state", "verification",
}
# Criterion IDs that must never be used as added criteria (verdict-field confusion).
RESERVED_CRITERION_IDS = VERDICT_TOP_LEVEL_KEYS | {"rubric_schema_version", "criteria",
                                                   "checkpoint_mapping", "assessments"}
DISCLAIMER = (
    "Scores rate the answers against the stated criteria. "
    "They are not a guarantee of success, market outcome, or regulatory outcome."
)

CRITERIA = (
    "evidence_strength",
    "capital_efficiency",
    "regulatory_runway",
    "differentiation",
    "risk_mitigation",
)

# Short names produced by panel_judge/ollama_judge.py, mapped to the
# canonical criterion IDs in DESIGN.md.
CRITERION_ALIASES = {
    "evidence": "evidence_strength",
    "efficiency": "capital_efficiency",
    "regulatory": "regulatory_runway",
    "differentiation": "differentiation",
    "risk_mitigation": "risk_mitigation",
}

CHECKPOINTS = (
    ("evidence", ("evidence_strength", "differentiation"),
     "Is there real, specific support for this strategy, and is it actually distinct?"),
    ("feasibility", ("capital_efficiency", "regulatory_runway"),
     "Can this actually be built, funded, and sold legally, and for how long?"),
    ("risk", ("risk_mitigation",),
     "What are the biggest risks, are they named, and does the strategy answer them?"),
)

SOURCES = ("model", "human", "ollama", "other")
ANSWER_KINDS = ("summary", "full")

DEFAULT_MAX_PANEL_CHARS = 250_000
DEFAULT_MAX_ANSWER_CHARS = 50_000
DEFAULT_MAX_ASSESSMENTS_CHARS = 250_000

DEFAULT_RUBRIC = [
    {"id": "evidence_strength", "name": "Evidence strength",
     "description": "Claims backed by specific, checkable evidence: named precedents, numbers, markets, mechanisms.",
     "weight": 0.2},
    {"id": "capital_efficiency", "name": "Capital / resource efficiency",
     "description": "Little upfront capital and few scarce resources relative to the goal: bought-in components, build-to-order, low volumes, deposits.",
     "weight": 0.2},
    {"id": "regulatory_runway", "name": "Regulatory / legal runway",
     "description": "A named, currently available legal route to market and a plausible life before that route closes.",
     "weight": 0.2},
    {"id": "differentiation", "name": "Differentiation",
     "description": "Distinct from incumbents and from other panelists in a way buyers would pay for.",
     "weight": 0.2},
    {"id": "risk_mitigation", "name": "Risk mitigation",
     "description": "Biggest risks named, with a concrete response to each (pivot, cap, deposit model, second market).",
     "weight": 0.2},
]


class JudgeError(Exception):
    """A named validation / judging failure. No partial verdict is written."""


# ---------------------------------------------------------------- validation

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _loads_no_duplicates(text, what):
    """json.loads that rejects duplicate object keys (review P1-5).

    Plain json.loads keeps the last duplicate, so {"evidence": 5,
    "evidence": 0} would silently become 0. A duplicate key is a
    malformed input here, never a merge.
    """
    def _no_dup(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise JudgeError(f"{what} has a duplicate key: {key!r}")
            out[key] = value
        return out
    try:
        return json.loads(text, object_pairs_hook=_no_dup)
    except JudgeError:
        raise
    except json.JSONDecodeError as exc:
        raise JudgeError(f"{what} is not valid JSON: {exc}") from exc


def canonical_panel_sha256(panel):
    """SHA-256 binding an assessments file to the panel it judged
    (review P0-2). Canonical form: question + panelist id/answer pairs
    in panel order, JSON with sorted keys. Extra panel fields (model,
    source, strategy_label, metadata) are NOT part of the binding, so
    the helper's panel form and the core panel form hash identically
    for the same question and answers. Truncated answers bind to the
    truncated text as loaded by core; a panel truncated differently
    is a different panel for this hash.
    """
    canon = {
        "question": panel["question"],
        "panelists": [{"id": p["id"], "answer": p["answer"]}
                      for p in panel["panelists"]],
    }
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def _round9(value):
    """Round a float comparison operand to 9 decimals (review P1-13).

    With six equal-weight criteria (1/6 each) a true 0.5 margin reads
    as 0.4999999999999998 in binary floats and would falsely force
    no-consensus. Rounding before margin/equality comparisons removes
    that artefact; the five-criterion 0.2 case was already clean.
    """
    return round(float(value), 9)


# Truncation marker: the marker INCLUDING its single separating space is
# 23 characters (" [truncated to budget]"); a truncated answer including
# marker + separator is at most the applicable limit (G17, P1-6).
TRUNCATION_MARKER = " [truncated to budget]"


def _reject_nul(value, what):
    if isinstance(value, str) and "\x00" in value:
        raise JudgeError(f"{what} contains a NUL byte, which is rejected")


def load_panel(path, max_panel_chars, max_answer_chars, allow_truncate):
    """Load and validate a panel file. Returns (panel, raw_bytes, warnings, truncated_info)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JudgeError(f"panel file is not valid UTF-8: {exc}")
    if len(text) > max_panel_chars and not allow_truncate:
        raise JudgeError(
            f"panel file exceeds --max-panel-chars ({max_panel_chars}): {len(text)} characters")
    panel = _loads_no_duplicates(text, "panel file")
    if not isinstance(panel, dict):
        raise JudgeError("panel file must contain a JSON object")
    # DESIGN G7 (v0.2): closed key sets. Any other key -- including any
    # verdict key -- is a validation error. A verdict-like key inside
    # metadata does NOT trigger this rule (metadata is free-form).
    allowed_top = {"panel_id", "question", "panelists", "metadata"}
    unknown_top = sorted(set(panel) - allowed_top)
    if unknown_top:
        raise JudgeError(
            f"panel file has unknown top-level key(s): {', '.join(unknown_top)} "
            "(allowed: panel_id, question, panelists, optional metadata; "
            "verdict/score keys are never allowed in a panel file)")
    for field in ("panel_id", "question", "panelists"):
        if field not in panel:
            raise JudgeError(f"panel file is missing {field!r}")
    if not isinstance(panel["panel_id"], str) or not panel["panel_id"].strip():
        raise JudgeError("panel_id must be a non-empty string")
    if not isinstance(panel["question"], str) or not panel["question"].strip():
        raise JudgeError("question must be a non-empty string")
    panelists = panel["panelists"]
    if not isinstance(panelists, list) or len(panelists) < 2:
        raise JudgeError("panelists must be an array of at least 2 entries")
    seen_ids, seen_answers, warnings = set(), {}, []
    truncated_info = {"truncated": False, "original_answer_chars": {},
                      "truncated_answer_chars": {}, "truncated_panelist_ids": []}
    for i, p in enumerate(panelists, start=1):
        if not isinstance(p, dict):
            raise JudgeError(f"panelist {i} is not a JSON object")
        allowed_panelist = {"id", "model", "source", "answer", "answer_kind",
                            "strategy_label", "is_dissent"}
        unknown_p = sorted(set(p) - allowed_panelist)
        if unknown_p:
            raise JudgeError(
                f"panelist {i} has unknown key(s): {', '.join(unknown_p)} "
                "(allowed: id, model, source, answer, optional answer_kind, "
                "strategy_label, is_dissent)")
        for field in ("id", "model", "source", "answer"):
            if field not in p or not isinstance(p[field], str) or not p[field].strip():
                raise JudgeError(f"panelist {i} needs a non-empty string {field!r}")
            _reject_nul(p[field], f"panelist {i} field {field!r}")
        if p["id"] in seen_ids:
            raise JudgeError(f"duplicate panelist id: {p['id']!r}")
        seen_ids.add(p["id"])
        if p["source"] not in SOURCES:
            raise JudgeError(f"panelist {p['id']!r}: source must be one of {SOURCES}, got {p['source']!r}")
        if "answer_kind" in p and p["answer_kind"] not in ANSWER_KINDS:
            raise JudgeError(f"panelist {p['id']!r}: answer_kind must be one of {ANSWER_KINDS}")
        if "strategy_label" in p and (not isinstance(p["strategy_label"], str) or not p["strategy_label"].strip()):
            raise JudgeError(f"panelist {p['id']!r}: strategy_label must be a non-empty string")
        if "strategy_label" in p:
            _reject_nul(p["strategy_label"], f"panelist {p['id']!r} strategy_label")
        if "is_dissent" in p and not isinstance(p["is_dissent"], bool):
            raise JudgeError(f"panelist {p['id']!r}: is_dissent must be a boolean")
        if len(p["answer"]) > max_answer_chars:
            if not allow_truncate:
                raise JudgeError(
                    f"panelist {p['id']!r} answer exceeds --max-answer-chars ({max_answer_chars}): "
                    f"{len(p['answer'])} characters")
            if max_answer_chars < len(TRUNCATION_MARKER):
                raise JudgeError(
                    f"--max-answer-chars ({max_answer_chars}) is smaller than the truncation "
                    f"marker ({len(TRUNCATION_MARKER)} characters); cannot truncate")
            truncated_info["truncated"] = True
            truncated_info["truncated_panelist_ids"].append(p["id"])
            truncated_info["original_answer_chars"][p["id"]] = len(p["answer"])
            p["answer"] = p["answer"][: max_answer_chars - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER
            truncated_info["truncated_answer_chars"][p["id"]] = len(p["answer"])
        if p["answer"] in seen_answers:
            warnings.append(
                f"duplicate answer text for panelists {seen_answers[p['answer']]!r} and {p['id']!r} "
                "(warning only: two panelists can genuinely converge)")
        else:
            seen_answers[p["answer"]] = p["id"]
    if panel.get("metadata") is not None and not isinstance(panel.get("metadata"), dict):
        raise JudgeError("metadata, if present, must be an object")
    _reject_nul(panel["panel_id"], "panel_id")
    _reject_nul(panel["question"], "question")
    # G17 panel-total fit test (v1.0 fix): --allow-truncate permits a
    # panel-total overage ONLY if the post-truncation total fits.
    # panel_chars_after_truncation = panel_chars - sum(original - truncated).
    # This check runs whenever the original file was over budget, including
    # when no answer was truncated (then nothing was saved, so it fails).
    if len(text) > max_panel_chars:
        after = len(text) - sum(
            truncated_info["original_answer_chars"][pid] - truncated_info["truncated_answer_chars"][pid]
            for pid in truncated_info["truncated_panelist_ids"])
        if after > max_panel_chars:
            raise JudgeError(
                f"panel file still exceeds --max-panel-chars ({max_panel_chars}) after "
                f"answer truncation: {after} characters")
    # G16 case-variant label warning (warning only; groups stay separate).
    # One warning per distinct case-variant pair.
    labels = []
    for p in panelists:
        lab = p.get("strategy_label")
        if isinstance(lab, str) and lab.strip():
            labels.append(lab.strip())
    seen_labels = []
    for lab in labels:
        if lab not in seen_labels:
            seen_labels.append(lab)
    for i, a in enumerate(seen_labels):
        for b in seen_labels[i + 1:]:
            if a.lower() == b.lower() and a != b:
                warnings.append(
                    f"labels {a!r} and {b!r} differ only by case and were treated as different groups")
    return panel, raw, warnings, truncated_info


_SLUG_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_CHECKPOINT_IDS = ("evidence", "feasibility", "risk")


def load_rubric(path):
    """Load a rubric file, or return the default.

    Returns (rubric_list, rubric_version, checkpoint_mapping, is_extended).
    Rubric list order: canonical five first in DESIGN order, then added
    criteria in rubric-file order (DESIGN §5.2).

    v1.0 (DESIGN §5): canonical IDs are frozen (rename forbidden), but
    ADDITION is allowed: total criteria at most 8 (at most 3 added),
    added IDs are slugs that must not collide with canonical IDs, the
    alias table, or verdict top-level keys, and every added criterion
    must map to exactly one checkpoint. Canonical entries in a declared
    mapping, if present, must equal the fixed canonical mapping; they
    may be omitted (fixed mapping applies). Weights are all-or-none
    over ALL criteria and must sum to 1.0 within 0.01 (absent -> 1/N).
    Object-form top-level keys are closed: criteria, rubric_schema_version,
    checkpoint_mapping only. Per-criterion keys are closed: id, name,
    description, weight only. Schema version must be 1 for canonical
    rubrics and 2 for extended rubrics. v1.0 imposes no size limit on
    the rubric file (stated, not silent).
    """
    if path is None:
        return ([dict(c) for c in DEFAULT_RUBRIC], "default-v1",
                dict(CANONICAL_CHECKPOINT_MAPPING), False)
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JudgeError(f"rubric file is not valid UTF-8: {exc}")
    data = _loads_no_duplicates(text, "rubric file")
    schema_version = None
    declared_mapping = None
    if isinstance(data, dict):
        unknown_top = sorted(set(data) - {"criteria", "rubric_schema_version", "checkpoint_mapping"})
        if unknown_top:
            raise JudgeError(f"rubric file has unknown top-level key(s): {', '.join(unknown_top)}")
        if not isinstance(data.get("criteria"), list):
            raise JudgeError("rubric file must be a list of criteria or an object with a 'criteria' list")
        criteria = data["criteria"]
        schema_version = data.get("rubric_schema_version")
        if schema_version is not None and schema_version not in (1, 2):
            raise JudgeError("rubric_schema_version must be 1 (canonical) or 2 (extended)")
        declared_mapping = data.get("checkpoint_mapping")
        if declared_mapping is not None and not isinstance(declared_mapping, dict):
            raise JudgeError("rubric checkpoint_mapping, if present, must be an object")
    elif isinstance(data, list):
        criteria = data
    else:
        raise JudgeError("rubric file must be a list of criteria or an object with a 'criteria' list")
    if not criteria:
        raise JudgeError("rubric must contain at least one criterion")
    seen = set()
    by_id = {}
    for c in criteria:
        if not isinstance(c, dict) or not all(k in c for k in ("id", "name", "description")):
            raise JudgeError("each rubric criterion needs id, name and description")
        unknown_c = sorted(set(c) - {"id", "name", "description", "weight"})
        if unknown_c:
            raise JudgeError(f"rubric criterion {c.get('id')!r} has unknown key(s): {', '.join(unknown_c)}")
        if not isinstance(c["id"], str) or not isinstance(c["name"], str) or not isinstance(c["description"], str):
            raise JudgeError("each rubric criterion needs string id, name and description")
        if not c["id"].strip() or not c["name"].strip() or not c["description"].strip():
            raise JudgeError("each rubric criterion needs non-empty id, name and description")
        _reject_nul(c["name"], f"rubric criterion {c['id']!r} name")
        _reject_nul(c["description"], f"rubric criterion {c['id']!r} description")
        if c["id"] in seen:
            raise JudgeError(f"duplicate rubric criterion id: {c['id']!r}")
        seen.add(c["id"])
        by_id[c["id"]] = c
    missing_canonical = [cid for cid in CRITERIA if cid not in seen]
    if missing_canonical:
        raise JudgeError(
            f"rubric is missing canonical criterion id(s): {', '.join(missing_canonical)}; "
            "removing or renaming a canonical criterion is invalid")
    added = [cid for cid in seen if cid not in CRITERIA]
    for cid in added:
        if not _SLUG_RE.match(cid):
            raise JudgeError(f"added rubric criterion id {cid!r} must be a slug matching ^[a-z][a-z0-9_]*$")
        if cid in CRITERION_ALIASES:
            raise JudgeError(f"added rubric criterion id {cid!r} collides with a criterion alias")
        if cid in RESERVED_CRITERION_IDS:
            raise JudgeError(f"added rubric criterion id {cid!r} collides with a verdict field name")
    if len(seen) > 8:
        raise JudgeError(f"extended rubric has {len(seen)} criteria; the cap is 8 total (at most 3 added)")
    is_extended = bool(added)
    if schema_version is not None and schema_version != (2 if is_extended else 1):
        raise JudgeError(
            f"rubric_schema_version {schema_version} does not match the rubric: "
            f"{'extended' if is_extended else 'canonical'} rubrics require schema version "
            f"{2 if is_extended else 1}")
    # Checkpoint mapping (P0-8): added entries required; canonical entries
    # optional but must equal the fixed mapping if present; unknown
    # criterion or checkpoint in a declared mapping is an error. The
    # duplicate-mapping error is deleted (unobservable after JSON parse).
    mapping = dict(CANONICAL_CHECKPOINT_MAPPING)
    if declared_mapping is not None:
        for cid, cp in declared_mapping.items():
            if cid not in seen:
                raise JudgeError(f"checkpoint_mapping names unknown criterion {cid!r}")
            if cp not in _CHECKPOINT_IDS:
                raise JudgeError(f"checkpoint_mapping for {cid!r} names unknown checkpoint {cp!r}")
            if cid in CANONICAL_CHECKPOINT_MAPPING:
                if cp != CANONICAL_CHECKPOINT_MAPPING[cid]:
                    raise JudgeError(
                        f"checkpoint_mapping for canonical criterion {cid!r} must be "
                        f"{CANONICAL_CHECKPOINT_MAPPING[cid]!r}, got {cp!r}")
            else:
                mapping[cid] = cp
    for cid in added:
        if declared_mapping is None or cid not in declared_mapping:
            raise JudgeError(f"added rubric criterion {cid!r} has no checkpoint_mapping entry")
    weights_present = [("weight" in c) for c in criteria]
    if any(weights_present) and not all(weights_present):
        raise JudgeError("rubric weights are all-or-none: present for all criteria or absent for all criteria")
    n = len(seen)
    ordered_ids = [cid for cid in CRITERIA if cid in by_id] + [c["id"] for c in criteria if c["id"] not in CRITERIA]
    total = 0.0
    final = []
    for cid in ordered_ids:
        c = by_id[cid]
        weight = float(c["weight"]) if "weight" in c else 1.0 / n
        if "weight" in c and (not isinstance(c["weight"], (int, float)) or isinstance(c["weight"], bool)):
            raise JudgeError(f"rubric criterion {cid!r}: weight must be a number")
        total += weight
        final.append({"id": cid, "name": c["name"], "description": c["description"], "weight": weight})
    if abs(total - 1.0) > 0.01:
        raise JudgeError(f"rubric weights must sum to 1.0, got {total:.3f}")
    version = ("custom-extended:" if is_extended else "custom:") + _sha256_bytes(raw)
    return final, version, mapping, is_extended


def _normalise_criterion(raw_id, panelist_id, rubric_ids=None):
    """Map a supplied criterion spelling to a canonical/extended ID.

    Added (extended-rubric) IDs are accepted verbatim when they are in
    the active rubric; canonical IDs and the frozen alias table are
    accepted otherwise. Anything else is an unknown criterion, never a guess.
    """
    if rubric_ids is not None and raw_id in rubric_ids:
        return raw_id
    if raw_id in CRITERIA:
        return raw_id
    if raw_id in CRITERION_ALIASES:
        return CRITERION_ALIASES[raw_id]
    raise JudgeError(f"assessment for {panelist_id!r}: unknown criterion {raw_id!r}")


_ASSESSMENT_ENVELOPE_KEYS = {
    "assessments", "assessor", "assessor_role", "assessor_model",
    "assessor_provider", "assessor_panelist_id", "prompt_version",
    "prompt_sha256", "temperature", "usage", "checkpoints",
    # review P0-2 / P1-2 / P1-6: panel binding, key-fix trace, and
    # helper context-window provenance. All optional except
    # panel_sha256 for a judge-model assessor (see load_assessments).
    # Re-review N-2: rubric_version (the assessor's anchors) and
    # window_limit are accepted here so a helper envelope can bind
    # its rubric; both are recorded in state, never silently equated
    # with the judging rubric (see synthesise warning below).
    "panel_sha256", "key_fixes", "num_ctx", "prompt_eval_count",
    "rubric_version", "window_limit",
}
_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _validate_usage(usage):
    """Provider-reported-only usage/cost (DESIGN §1.3, P1-2)."""
    if not isinstance(usage, dict):
        raise JudgeError("assessments 'usage', if present, must be an object")
    unknown = sorted(set(usage) - {"prompt_tokens", "completion_tokens", "total_tokens",
                                   "cost_amount", "cost_currency", "pricing_source"})
    if unknown:
        raise JudgeError(f"assessments 'usage' has unknown field(s): {', '.join(unknown)}")
    for tok in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if tok in usage and usage[tok] is not None:
            if not isinstance(usage[tok], int) or isinstance(usage[tok], bool) or usage[tok] < 0:
                raise JudgeError(f"assessments 'usage.{tok}' must be an integer >= 0")
    cost_amount = usage.get("cost_amount")
    cost_currency = usage.get("cost_currency")
    pricing_source = usage.get("pricing_source")
    if cost_amount is not None:
        if not isinstance(cost_amount, (int, float)) or isinstance(cost_amount, bool) or cost_amount < 0:
            raise JudgeError("assessments 'usage.cost_amount' must be a number >= 0 or null")
        if not isinstance(cost_currency, str) or not cost_currency.strip():
            raise JudgeError("assessments 'usage.cost_amount' requires a non-empty 'cost_currency'")
        if not isinstance(pricing_source, str) or not pricing_source.strip():
            raise JudgeError("assessments 'usage.cost_amount' requires a non-empty 'pricing_source'")
    if cost_currency is not None:
        if not isinstance(cost_currency, str) or not cost_currency.strip():
            raise JudgeError("assessments 'usage.cost_currency' must be a non-empty string or null")
        if cost_amount is None:
            raise JudgeError("assessments 'usage.cost_currency' without 'cost_amount' is invalid")
    if pricing_source is not None and (not isinstance(pricing_source, str) or not pricing_source.strip()):
        raise JudgeError("assessments 'usage.pricing_source', if present, must be a non-empty string or null")
    # cost_amount null + cost_currency null satisfies "both absent/null";
    # a null cost_amount does NOT trigger the pricing_source requirement.
    return usage


def load_assessments(path, panel, rubric, max_assessments_chars, checkpoint_mapping=None):
    """Load and validate a supplied-assessments file (DESIGN §1.3 envelope).

    Returns (per, sha256, assessor, assessor_role, supplied_checkpoints,
    prompt_version, assessor_panelist_id, assessments_chars, envelope)
    where envelope = {assessor_model, assessor_provider, prompt_sha256,
    temperature, usage}. An over-budget assessments file ALWAYS fails,
    even with --allow-truncate (no defined truncation path for scores).
    """
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JudgeError(f"assessments file is not valid UTF-8: {exc}")
    if len(text) > max_assessments_chars:
        raise JudgeError(
            f"assessments file exceeds --max-assessments-chars ({max_assessments_chars}): {len(text)} characters")
    data = _loads_no_duplicates(text, "assessments file")
    if not isinstance(data, dict) or not isinstance(data.get("assessments"), list):
        raise JudgeError("assessments file must be an object with an 'assessments' list")
    unknown_top = sorted(set(data) - _ASSESSMENT_ENVELOPE_KEYS)
    if unknown_top:
        raise JudgeError(
            f"assessments file has unknown top-level key(s): {', '.join(unknown_top)} "
            "(closed envelope: verification and any other key must not be smuggled in here)")
    # --- Envelope: decidable role/provider/prompt rules (P1-1) -----------
    assessor_role = data.get("assessor_role")
    if assessor_role not in ("judge-model", "human"):
        raise JudgeError("assessments 'assessor_role' is required and must be 'judge-model' or 'human'")
    assessor = data.get("assessor")
    if assessor is not None and (not isinstance(assessor, str) or not assessor.strip()):
        raise JudgeError("assessments 'assessor', if present, must be a non-empty string")
    if assessor is not None:
        _reject_nul(assessor, "assessments 'assessor'")
    if assessor_role == "judge-model" and not assessor:
        raise JudgeError("assessments 'assessor' is required for a judge-model assessor")
    assessor_model = data.get("assessor_model")
    if assessor_role == "judge-model":
        if not isinstance(assessor_model, str) or not assessor_model.strip():
            raise JudgeError("assessments 'assessor_model' is required (non-empty) for a judge-model assessor")
    else:
        if assessor_model is not None:
            raise JudgeError("assessments 'assessor_model' is forbidden with assessor_role 'human'")
    assessor_provider = data.get("assessor_provider")
    if assessor_model is not None:
        if not isinstance(assessor_provider, str) or not assessor_provider.strip():
            raise JudgeError("assessments 'assessor_provider' is required when 'assessor_model' is present")
    elif assessor_provider is not None:
        if not isinstance(assessor_provider, str) or not assessor_provider.strip():
            raise JudgeError("assessments 'assessor_provider', if present, must be a non-empty string")
        if assessor_role == "human" and assessor_provider.strip().lower() in ("openrouter", "ollama"):
            raise JudgeError(
                f"assessments 'assessor_provider' {assessor_provider!r} contradicts assessor_role 'human'")
    prompt_version = data.get("prompt_version")
    if assessor_role == "judge-model":
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise JudgeError("assessments 'prompt_version' is required (non-empty) for a judge-model assessor")
    elif prompt_version is not None and (not isinstance(prompt_version, str) or not prompt_version.strip()):
        raise JudgeError("assessments 'prompt_version', if present, must be a non-empty string")
    prompt_sha256 = data.get("prompt_sha256")
    if prompt_sha256 is not None and (not isinstance(prompt_sha256, str) or not _HEX64_RE.match(prompt_sha256)):
        raise JudgeError("assessments 'prompt_sha256', if present, must be 64 lowercase hex characters")
    temperature = data.get("temperature")
    if temperature is not None and (not isinstance(temperature, (int, float)) or isinstance(temperature, bool)):
        raise JudgeError("assessments 'temperature', if present, must be a number")
    # review P1-8: provenance is no longer optional for a judge-model
    # assessor. A judge-model verdict with a null prompt SHA or a
    # missing temperature must not be signable off (see cli.py).
    if assessor_role == "judge-model":
        if prompt_sha256 is None:
            raise JudgeError("assessments 'prompt_sha256' is required for a judge-model assessor")
        if temperature is None:
            raise JudgeError("assessments 'temperature' is required for a judge-model assessor")
    # review P0-2: panel binding. For a judge-model assessor the
    # envelope must carry the canonical panel SHA and it must equal
    # the panel as loaded here (post-truncation answers, see
    # canonical_panel_sha256). For a human assessor the field is
    # optional, but a present value that mismatches is still rejected
    # -- a wrong binding is wrong whoever asserted it.
    panel_sha256 = data.get("panel_sha256")
    if panel_sha256 is not None and (not isinstance(panel_sha256, str) or not _HEX64_RE.match(panel_sha256)):
        raise JudgeError("assessments 'panel_sha256', if present, must be 64 lowercase hex characters")
    if assessor_role == "judge-model" and panel_sha256 is None:
        raise JudgeError("assessments 'panel_sha256' is required for a judge-model assessor")
    if panel_sha256 is not None and panel_sha256 != canonical_panel_sha256(panel):
        raise JudgeError(
            "assessments 'panel_sha256' does not match this panel: the assessments "
            "were made for a different panel (different question or answers)")
    # Optional helper provenance (review P1-2 / P1-6), validated, never trusted blindly.
    key_fixes = data.get("key_fixes")
    if key_fixes is not None:
        if not isinstance(key_fixes, dict):
            raise JudgeError("assessments 'key_fixes', if present, must be an object keyed by panelist id")
        for pid, fixes in key_fixes.items():
            if pid not in [p["id"] for p in panel["panelists"]]:
                raise JudgeError(f"assessments 'key_fixes' names unknown panelist: {pid!r}")
            if not isinstance(fixes, list) or not all(isinstance(x, str) for x in fixes):
                raise JudgeError(f"assessments 'key_fixes' for {pid!r} must be a list of strings")
    for _num_key in ("num_ctx", "prompt_eval_count"):
        _num_val = data.get(_num_key)
        if _num_val is not None and (not isinstance(_num_val, int) or isinstance(_num_val, bool) or _num_val < 0):
            raise JudgeError(f"assessments '{_num_key}', if present, must be an integer >= 0")
    usage = None
    if "usage" in data and data["usage"] is not None:
        usage = _validate_usage(data["usage"])
    panel_ids = [p["id"] for p in panel["panelists"]]
    assessor_panelist_id = data.get("assessor_panelist_id")
    if assessor_panelist_id is not None:
        if not isinstance(assessor_panelist_id, str):
            raise JudgeError("assessments 'assessor_panelist_id', if present, must be a string or null")
        if assessor_panelist_id not in panel_ids:
            raise JudgeError(f"assessments 'assessor_panelist_id' names a panelist not in the panel: {assessor_panelist_id!r}")
    # --- Per-assessment entries (closed keys, identifier aliases) --------
    rubric_ids = [c["id"] for c in rubric]
    per = {}
    for entry in data["assessments"]:
        if not isinstance(entry, dict):
            raise JudgeError("an assessment entry is not a JSON object")
        unknown_e = sorted(set(entry) - {"panelist_id", "panelist", "id", "scores", "reasons",
                                         "strengths", "weaknesses"})
        if unknown_e:
            raise JudgeError(f"an assessment entry has unknown key(s): {', '.join(unknown_e)}")
        id_values = {k: entry[k] for k in ("panelist_id", "panelist", "id") if k in entry}
        if not id_values:
            raise JudgeError("an assessment entry has no panelist identifier "
                             "(panelist_id preferred; panelist and id accepted as aliases)")
        distinct = set(id_values.values())
        if len(distinct) > 1:
            raise JudgeError(
                f"assessment entry has differing identifier keys: {id_values!r}")
        pid = next(iter(distinct))
        if pid not in panel_ids:
            raise JudgeError(f"assessment for unknown panelist: {pid!r}")
        if pid in per:
            raise JudgeError(f"duplicate assessment for panelist: {pid!r}")
        scores_raw, reasons_raw = entry.get("scores"), entry.get("reasons")
        if not isinstance(scores_raw, dict) or not isinstance(reasons_raw, dict):
            raise JudgeError(f"assessment for {pid!r} needs 'scores' and 'reasons' objects")
        scores, reasons = {}, {}
        for raw_id, value in scores_raw.items():
            cid = _normalise_criterion(raw_id, pid, rubric_ids)
            # review P0-1: an alias and its canonical id in one entry
            # (e.g. "evidence": 1 and "evidence_strength": 5) must be
            # rejected, not silently merged with last-key-wins.
            if cid in scores:
                raise JudgeError(
                    f"assessment for {pid!r}: criterion {cid!r} is given twice "
                    f"(via alias {raw_id!r} and another key); refusing to merge")
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 5:
                raise JudgeError(f"assessment for {pid!r}: score for {cid!r} must be an integer 0-5")
            scores[cid] = value
        for raw_id, value in reasons_raw.items():
            cid = _normalise_criterion(raw_id, pid, rubric_ids)
            if cid in reasons:
                raise JudgeError(
                    f"assessment for {pid!r}: reason for {cid!r} is given twice "
                    f"(via alias {raw_id!r} and another key); refusing to merge")
            if not isinstance(value, str) or not value.strip():
                raise JudgeError(f"assessment for {pid!r}: reason for {cid!r} must be a non-empty string")
            _reject_nul(value, f"assessment for {pid!r} reason for {cid!r}")
            reasons[cid] = value.strip()
        for cid in rubric_ids:
            if cid not in scores or cid not in reasons:
                raise JudgeError(f"assessment for {pid!r} is missing score/reason for {cid!r}")
        for key in ("strengths", "weaknesses"):
            if key in entry and entry[key] is not None:
                if not isinstance(entry[key], list) or not all(isinstance(x, str) for x in entry[key]):
                    raise JudgeError(f"assessment for {pid!r}: {key!r} must be an array of strings")
        per[pid] = {
            "scores": scores,
            "reasons": reasons,
            "strengths": entry.get("strengths") or [],
            "weaknesses": entry.get("weaknesses") or [],
        }
    missing = [pid for pid in panel_ids if pid not in per]
    if missing:
        raise JudgeError(f"assessments are missing panelists: {', '.join(missing)}")
    # --- Supplied checkpoints: validated against the ACTIVE mapping ------
    if checkpoint_mapping is None:
        checkpoint_mapping = dict(CANONICAL_CHECKPOINT_MAPPING)
    checkpoint_criteria = {}
    for cid, cp_id in checkpoint_mapping.items():
        checkpoint_criteria.setdefault(cp_id, set()).add(cid)
    supplied_checkpoints = data.get("checkpoints", [])
    if not isinstance(supplied_checkpoints, list):
        raise JudgeError("assessments 'checkpoints', if present, must be a list")
    normalised_checkpoints = []
    for sc in supplied_checkpoints:
        if not isinstance(sc, dict):
            raise JudgeError("a supplied checkpoint entry is not a JSON object")
        unknown_sc = sorted(set(sc) - {"checkpoint_id", "scores", "notes", "strategy_label"})
        if unknown_sc:
            raise JudgeError(f"supplied checkpoint has unknown key(s): {', '.join(unknown_sc)}")
        if sc.get("checkpoint_id") not in checkpoint_criteria:
            raise JudgeError(f"unknown supplied checkpoint: {sc.get('checkpoint_id')!r} (must be evidence, feasibility or risk)")
        if not isinstance(sc.get("scores"), dict):
            raise JudgeError(f"supplied checkpoint {sc['checkpoint_id']!r} needs a 'scores' object")
        if not isinstance(sc.get("notes"), str) or not sc["notes"].strip():
            raise JudgeError(f"supplied checkpoint {sc['checkpoint_id']!r} needs a non-empty 'notes' string")
        _reject_nul(sc["notes"], f"supplied checkpoint {sc['checkpoint_id']!r} notes")
        norm_scores = {}
        for raw_id, value in sc["scores"].items():
            cid = _normalise_criterion(raw_id, "supplied checkpoint", rubric_ids)
            if cid in norm_scores:
                raise JudgeError(
                    f"supplied checkpoint {sc['checkpoint_id']!r}: criterion {cid!r} "
                    f"is given twice (via alias {raw_id!r} and another key); refusing to merge")
            if cid not in checkpoint_criteria[sc["checkpoint_id"]]:
                raise JudgeError(
                    f"supplied checkpoint {sc['checkpoint_id']!r}: criterion {cid!r} "
                    f"is not scored by that checkpoint")
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 5:
                raise JudgeError(
                    f"supplied checkpoint {sc['checkpoint_id']!r}: score for {cid!r} "
                    "must be an integer 0-5")
            norm_scores[cid] = value
        entry = {"checkpoint_id": sc["checkpoint_id"], "scores": norm_scores,
                 "notes": sc["notes"].strip()}
        if sc.get("strategy_label") is not None:
            if not isinstance(sc["strategy_label"], str) or not sc["strategy_label"].strip():
                raise JudgeError("supplied checkpoint 'strategy_label', if present, must be a non-empty string")
            entry["strategy_label"] = sc["strategy_label"].strip()
        normalised_checkpoints.append(entry)
    # Re-review N-2: the assessor's rubric version is bound, not
    # assumed. It is optional here so pre-v4 shipped files still
    # validate (their rubric is unknown, not claimed); when present
    # it must be a non-empty string, is recorded in the envelope,
    # and synthesise warns if it differs from the judging rubric.
    # A helper "ollama-builtin-v1" value is deliberately distinct
    # from core "default-v1": the helper guidance text differs from
    # the core rubric descriptions, so the two are never equated.
    rubric_version_assessor = data.get("rubric_version")
    if rubric_version_assessor is not None and (
            not isinstance(rubric_version_assessor, str) or not rubric_version_assessor.strip()):
        raise JudgeError("assessments 'rubric_version', if present, must be a non-empty string")
    window_limit = data.get("window_limit")
    if window_limit is not None and (
            not isinstance(window_limit, int) or isinstance(window_limit, bool) or window_limit <= 0):
        raise JudgeError("assessments 'window_limit', if present, must be a positive integer")
    envelope = {"assessor_model": assessor_model, "assessor_provider": assessor_provider,
                "prompt_sha256": prompt_sha256, "temperature": temperature, "usage": usage,
                "panel_sha256": panel_sha256, "key_fixes": key_fixes,
                "num_ctx": data.get("num_ctx"), "prompt_eval_count": data.get("prompt_eval_count"),
                "rubric_version": rubric_version_assessor, "window_limit": window_limit}
    return (per, _sha256_bytes(raw), assessor, assessor_role,
            normalised_checkpoints, prompt_version, assessor_panelist_id, len(text), envelope)


# ---------------------------------------------------------------- verification hook

_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?Z$")
_VERIFICATION_TOP_KEYS = {"verifier", "verified_at", "checks"}
_VERIFICATION_CHECK_KEYS = {"panelist_id", "claim_excerpt", "claim_is_verbatim",
                            "criterion_bearing", "source_ref", "status", "evidence_note"}
_VERIFICATION_STATUSES = ("verified", "contradicted", "unverifiable", "not_checked")


def load_verification(path, panel, rubric):
    """Load and validate a --verification file (DESIGN §3.2, annotation only).

    Validation only: verification NEVER feeds back into scores. Returns
    (record, sha256, path_as_supplied) where record is the verdict's
    top-level `verification` object. v1.0 imposes no size limit on the
    verification file (stated, not silent). A future verified_at is
    allowed and recorded verbatim. Verification carried in panel
    metadata is ignored for all purposes (only --verification counts).
    """
    from urllib.parse import urlparse
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise JudgeError(f"verification file is not valid UTF-8: {exc}")
    data = _loads_no_duplicates(text, "verification file")
    if not isinstance(data, dict):
        raise JudgeError("verification file must contain a JSON object")
    unknown_top = sorted(set(data) - _VERIFICATION_TOP_KEYS)
    if unknown_top:
        raise JudgeError(f"verification file has unknown top-level key(s): {', '.join(unknown_top)}")
    verifier = data.get("verifier")
    if not isinstance(verifier, str) or not verifier.strip():
        raise JudgeError("verification 'verifier' must be a non-empty string")
    _reject_nul(verifier, "verification 'verifier'")
    verified_at = data.get("verified_at")
    if verified_at is not None:
        if not isinstance(verified_at, str) or not _TIMESTAMP_RE.match(verified_at):
            raise JudgeError("verification 'verified_at', if present, must match the UTC ISO-8601 Z format")
        try:
            datetime.strptime(verified_at, "%Y-%m-%dT%H:%M:%S.%fZ" if "." in verified_at
                              else "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise JudgeError(f"verification 'verified_at' is not a valid timestamp: {exc}")
    checks_raw = data.get("checks")
    if not isinstance(checks_raw, list) or not checks_raw:
        raise JudgeError("verification 'checks' must be a non-empty array")
    panel_ids = [p["id"] for p in panel["panelists"]]
    answers = {p["id"]: p["answer"] for p in panel["panelists"]}
    rubric_ids = [c["id"] for c in rubric]
    counts = {"verified": 0, "contradicted": 0, "unverifiable": 0, "not_checked": 0}
    for i, chk in enumerate(checks_raw, start=1):
        if not isinstance(chk, dict):
            raise JudgeError(f"verification check {i} is not a JSON object")
        unknown_c = sorted(set(chk) - _VERIFICATION_CHECK_KEYS)
        if unknown_c:
            raise JudgeError(f"verification check {i} has unknown key(s): {', '.join(unknown_c)}")
        for field in _VERIFICATION_CHECK_KEYS:
            if field not in chk:
                raise JudgeError(f"verification check {i} is missing field {field!r}")
        pid = chk["panelist_id"]
        if pid not in panel_ids:
            raise JudgeError(f"verification check {i} names unknown panelist: {pid!r}")
        excerpt = chk["claim_excerpt"]
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise JudgeError(f"verification check {i}: claim_excerpt must be a non-empty string")
        _reject_nul(excerpt, f"verification check {i} claim_excerpt")
        verbatim = chk["claim_is_verbatim"]
        if not isinstance(verbatim, bool):
            raise JudgeError(f"verification check {i}: claim_is_verbatim must be a JSON boolean")
        if verbatim and excerpt not in answers[pid]:
            raise JudgeError(
                f"verification check {i}: verbatim claim_excerpt is not an exact substring "
                f"of panelist {pid!r}'s answer as judged")
        bearing = chk["criterion_bearing"]
        if bearing is not None and bearing not in rubric_ids:
            raise JudgeError(f"verification check {i}: criterion_bearing {bearing!r} is not in the active rubric")
        status = chk["status"]
        if status not in _VERIFICATION_STATUSES:
            raise JudgeError(f"verification check {i}: status must be one of {_VERIFICATION_STATUSES}")
        source_ref = chk["source_ref"]
        if status in ("verified", "contradicted"):
            if not isinstance(source_ref, str) or not source_ref.strip():
                raise JudgeError(
                    f"verification check {i}: source_ref is required (non-empty) for a {status} check")
        elif source_ref is not None and (not isinstance(source_ref, str)):
            raise JudgeError(f"verification check {i}: source_ref must be a string or null")
        if isinstance(source_ref, str) and "://" in source_ref:
            netloc = urlparse(source_ref).netloc
            if "@" in netloc:
                raise JudgeError(f"verification check {i}: source_ref URL must not contain credentials")
        if isinstance(source_ref, str):
            _reject_nul(source_ref, f"verification check {i} source_ref")
        note = chk["evidence_note"]
        if not isinstance(note, str) or not note.strip():
            raise JudgeError(f"verification check {i}: evidence_note must be a non-empty string")
        _reject_nul(note, f"verification check {i} evidence_note")
        counts[status] += 1
    # review P1-10: outcome taxonomy is unchanged (existing callers
    # and tests rely on "passed" for a fully verified set), but the
    # outcome is now labelled with its coverage, so a "passed" that
    # checked only one panelist cannot be read as panel-wide
    # verification. Coverage is recorded, not silently assumed.
    covered = sorted({chk["panelist_id"] for chk in checks_raw})
    uncovered = [pid for pid in panel_ids if pid not in covered]
    coverage = {"panelists_checked": covered, "panelists_unchecked": uncovered,
                "panelists_total": len(panel_ids),
                "coverage_complete": not uncovered}
    if counts["contradicted"] >= 1:
        outcome = "contradicted"
    elif (counts["verified"] >= 1 and counts["contradicted"] == 0
          and counts["unverifiable"] == 0 and counts["not_checked"] == 0):
        outcome = "passed"
    else:
        outcome = "mixed"
    record = {"verifier": verifier, "verified_at": verified_at,
              "checks": checks_raw, "summary_counts": counts,
              "verification_outcome": outcome,
              "verification_coverage": coverage,
              "scores_unchanged_by_verification": True}
    return record, _sha256_bytes(raw), str(path)


# ---------------------------------------------------------------- heuristic scorer

_NUM_RE = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|million|billion|£|\$|€)?", re.IGNORECASE)

_FEATURES = {
    "evidence_strength": (
        r"\b(sold|sales|priced|price|from \$|from £|from €|million|billion|per year|units|"
        r"models? fell|shrinking|study|trial|report|named|precedent)\b",
        r"\d",
    ),
    "capital_efficiency": (
        r"\b(bought-in|bought in|outsource|outsourc|build-to-order|confirmed orders?|"
        r"deposits?|low-volume|low volume|few hundred|small|disciplined|avoid|"
        r"instead of developing|low capital|direct|asset-light)\b",
    ),
    "regulatory_runway": (
        # Generic regulatory language only. No domain-specific regulation
        # names (e.g. a specific act or framework) — those would bias
        # the scorer toward one panel's subject matter.
        r"\b(regulat|exemption|homologation|legal|legally|"
        r"route to (market|sale)|ban|tolerant markets?)\b",
    ),
    "differentiation": (
        r"\b(scarcity|unique|distinct|exclusive|heritage|analogue|analog|emotional|"
        r"collectib|rarity|premium|driver-focused|purist|last|honest|craftsmanship|"
        r"distinctive|niche)\b",
    ),
    "risk_mitigation": (
        r"\b(risk|biggest risk|mitigat|fallback|pivot|cap(?:ped)? output|tolerant markets?|"
        r"future fuels?|hybrid-ready|staged|validate|second market|diversif)\b",
    ),
}


def _count(patterns, text):
    total = 0
    for pat in patterns:
        total += len(re.findall(pat, text, flags=re.IGNORECASE))
    return total


def heuristic_scores(answer):
    """Deterministic, transparent scorer. Rewards specificity and structure.

    Limits (stated in every report that uses it): it cannot tell whether
    a cited precedent is real or a regulation actually applies.
    Returns (scores dict, reasons dict, feature counts dict).
    """
    text = answer or ""
    scores, reasons, counts = {}, {}, {}
    # evidence: numbers are the strongest generic signal, plus named facts
    n_numbers = len(_NUM_RE.findall(text))
    n_evidence_words = _count(_FEATURES["evidence_strength"], text)
    counts["evidence_strength"] = n_numbers + n_evidence_words
    if n_numbers >= 4:
        ev = 4
    elif n_numbers >= 2:
        ev = 3
    elif n_numbers >= 1:
        ev = 2
    else:
        ev = 1
    if n_evidence_words >= 3 and ev < 5:
        ev = min(5, ev + 1)
    scores["evidence_strength"] = ev
    reasons["evidence_strength"] = (
        f"Heuristic: {n_numbers} numeric tokens and {n_evidence_words} evidence words in the answer; "
        "specificity rewarded, truth of cited facts not checked.")
    for cid in ("capital_efficiency", "regulatory_runway", "differentiation", "risk_mitigation"):
        c = _count(_FEATURES[cid], text)
        counts[cid] = c
        if c >= 6:
            s = 4
        elif c >= 4:
            s = 3
        elif c >= 2:
            s = 2
        elif c >= 1:
            s = 1
        else:
            s = 0
        # risk_mitigation needs both a named risk and a response; a lone
        # "risk" mention without mitigation words stays low.
        if cid == "risk_mitigation":
            has_risk = bool(re.search(r"\brisk\b", text, re.IGNORECASE))
            has_response = bool(re.search(
                r"\b(mitigat|fallback|pivot|cap(?:ped)?|tolerant|future fuels?|hybrid-ready|staged|validate)\b",
                text, re.IGNORECASE))
            if has_risk and not has_response:
                s = min(s, 1)
            elif has_risk and has_response and s < 3:
                s = 3
        scores[cid] = s
        reasons[cid] = f"Heuristic: {c} matching feature mentions for {cid} in the answer text."
    return scores, reasons, counts


# ---------------------------------------------------------------- verdict synthesis

def _weighted_mean(scores, rubric):
    return sum(scores[c["id"]] * c["weight"] for c in rubric)


def _strategy_groups(panel):
    """Group panelists by exact strategy_label (stripped, case-significant).

    A panelist with no strategy_label forms a singleton group of its own.
    Groups are never merged, even if two labels describe similar strategies.
    Returns a list, in first-appearance order, of dicts with keys:
    key (stable internal identity), label (str or None), members (panelist
    ids), panelist_id (the sole member's id for unlabeled groups, else None).
    """
    groups, by_key = [], {}
    for p in panel["panelists"]:
        raw_label = p.get("strategy_label")
        label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else None
        if label is not None:
            key = ("label", label)
        else:
            key = ("panelist", p["id"])
        if key not in by_key:
            grp = {"key": key, "label": label, "members": [],
                   "panelist_id": p["id"] if label is None else None}
            by_key[key] = grp
            groups.append(grp)
        by_key[key]["members"].append(p["id"])
    return groups


def _group_scores(member_ids, per_panelist, criteria_ids):
    """A strategy group's criterion score is its checkpoint score: for the
    heuristic route the group's answers scored as a whole, rewarding the
    best-supported evidence across members; for the supplied route the
    highest supplied per-panelist score within the group (DESIGN A1).
    Both reduce to the per-criterion maximum across the group's members,
    with supporters named so a strong score cannot hide behind a weak panel.
    criteria_ids is the active rubric's criterion list (canonical or extended).
    """
    out, supporters = {}, {}
    for cid in criteria_ids:
        best = max(per_panelist[pid]["scores"].get(cid, 0) for pid in member_ids)
        out[cid] = best
        supporters[cid] = [pid for pid in member_ids
                           if per_panelist[pid]["scores"].get(cid, 0) == best]
    return out, supporters


_MARKET_TOKENS = (
    "united states", "middle east", "europe", "australia", "asia",
    "japan", "africa", "united kingdom", "california", "china",
    "india", "brazil", "southeast asia", "north america",
)


def _market_set(answer):
    low = (answer or "").lower()
    return {tok for tok in _MARKET_TOKENS if tok in low}


_RESPONSE_WORDS_RE = re.compile(
    r"\b(pivot|fallback|mitigat|cap(?:ped)?|deposit|second market|hybrid-ready|staged|validate)\b",
    re.IGNORECASE)
_REMOVAL_RE = re.compile(
    r"remov\w*\s+(the\s+)?(legal\s+(route|path)|sole|core\s+market)|sole\s+(critical\s+)?supply",
    re.IGNORECASE)


def _risk_register(group_members, panel):
    """Heuristic risk register for one strategy group (DESIGN A1).

    One entry per named risk sentence in the group's answers. fatal is
    False unless the sentence states the risk removes the legal route /
    sole supply / core market AND states there is no pivot -- the
    conservative default is non-fatal. Empty array if no risk is named.
    """
    by_id = {p["id"]: p for p in panel["panelists"]}
    entries, seen = [], {}
    for pid in group_members:
        answer = by_id[pid]["answer"] or ""
        for sentence in re.split(r"(?<=[.!?])\s+", answer):
            sent = sentence.strip()
            if not sent or "risk" not in sent.lower():
                continue
            response = None
            for cand in re.split(r"(?<=[.!?])\s+", answer):
                if _RESPONSE_WORDS_RE.search(cand):
                    response = cand.strip()
                    break
            fatal = bool(_REMOVAL_RE.search(sent)) and "no pivot" in sent.lower()
            if sent in seen:
                seen[sent]["named_by"].append(pid)
            else:
                entry = {"risk": sent, "named_by": [pid],
                         "response_in_answers": response, "fatal": fatal}
                seen[sent] = entry
                entries.append(entry)
    return entries


_CHECKPOINT_QUESTIONS = {
    "evidence": "Is there real, specific support for this strategy, and is it actually distinct?",
    "feasibility": "Can this actually be built, funded, and sold legally, and for how long?",
    "risk": "What are the biggest risks, are they named, and does the strategy answer them?",
}


def synthesise(panel, rubric, per_panelist, judging_route, judge_model,
                panel_raw, panel_path, rubric_version, assessments_sha256,
                budget_chars, truncated_info, supplied_checkpoints=None,
                assessor=None, warnings=None, assessor_role=None,
                prompt_version=None, max_answer_chars=None,
                max_assessments_chars=None, rubric_file=None,
                assessments_file=None, ollama_endpoint=None,
                assessor_panelist_id=None, assessments_chars=None,
                checkpoint_mapping=None, envelope=None,
                verification_record=None, verification_sha256=None,
                verification_file=None):
    """Build the verdict. Checkpoint scores are computed FIRST, from the
    panel answers / supplied per-panelist scores, and recorded immutably;
    verdict scores are then taken from the recommended (or leading)
    group's checkpoint scores and are never derived the other way round
    (DESIGN A1). Verification, if given, is recorded only: it never
    enters this function before this point and changes no score here.
    Returns (verdict, warnings, excluded_flagged)."""
    if checkpoint_mapping is None:
        checkpoint_mapping = dict(CANONICAL_CHECKPOINT_MAPPING)
    rubric_ids = [c["id"] for c in rubric]
    panel_ids = [p["id"] for p in panel["panelists"]]
    by_id = {p["id"]: p for p in panel["panelists"]}
    groups = _strategy_groups(panel)

    # --- Checkpoint scores first, per (checkpoint, strategy group) -------
    group_data = {}
    for grp in groups:
        scores, supporters = _group_scores(grp["members"], per_panelist, rubric_ids)
        group_data[grp["key"]] = {
            "group": grp, "scores": scores, "supporters": supporters,
            "overall": _weighted_mean(scores, rubric),
        }
    ranked = sorted(group_data.values(),
                    key=lambda gd: (-_round9(gd["overall"]), gd["group"]["label"] or gd["group"]["panelist_id"] or ""))
    top = ranked[0]
    second = ranked[1] if len(ranked) > 1 else None
    # review P1-13: compare rounded overalls, not raw binary floats.
    margin = _round9(top["overall"] - second["overall"]) if second is not None else None
    tied_top = second is not None and _round9(top["overall"]) == _round9(second["overall"])

    # --- Consensus (DESIGN A2 + A3, no-consensus checked first) ----------
    if _round9(top["overall"]) < 2.5 or (margin is not None and margin < 0.5):
        consensus, recommended_key = "no-consensus", None
    else:
        recommended_key = top["group"]["key"]
        share = len(top["group"]["members"]) / len(panel_ids)
        largest_share = max(len(gd["group"]["members"]) for gd in ranked) / len(panel_ids)
        if share > 0.5:
            consensus = "majority"
        elif share == largest_share:
            consensus = "plurality"
        else:
            consensus = "minority"
    recommended_label = None
    if recommended_key is not None:
        recommended_label = group_data[recommended_key]["group"]["label"]
    if consensus == "no-consensus":
        leading_group = None if tied_top else top["group"]
        leading_label = leading_group["label"] if leading_group is not None else None
    else:
        leading_group = top["group"]
        leading_label = recommended_label
    verdict_group = top if consensus == "no-consensus" else group_data[recommended_key]
    verdict_scores = verdict_group["scores"]
    verdict_supporters = verdict_group["supporters"]

    # --- Checkpoint entries (one per checkpoint x group) -----------------
    # Criteria per checkpoint come from the ACTIVE mapping (canonical or
    # extended), in DESIGN criterion order, not from the canonical table.
    checkpoint_criteria_ordered = {}
    for cid in rubric_ids:
        checkpoint_criteria_ordered.setdefault(checkpoint_mapping[cid], []).append(cid)
    checkpoints = []
    for cp_id in _CHECKPOINT_IDS:
        criteria = checkpoint_criteria_ordered.get(cp_id, [])
        question = _CHECKPOINT_QUESTIONS[cp_id]
        for gd in group_data.values():
            grp = gd["group"]
            cp_scores = {cid: int(gd["scores"][cid]) for cid in criteria if cid in gd["scores"]}
            entry = {"checkpoint_id": cp_id, "strategy_label": grp["label"],
                     "scores": cp_scores,
                     "notes": (question + " Checkpoint scores for this strategy group are "
                               "computed from the group's answers (or supplied per-panelist "
                               "scores) before any verdict score, and are not derived from "
                               "verdict scores.")}
            if grp["label"] is None:
                entry["panelist_id"] = grp["panelist_id"]
            if cp_id == "risk":
                entry["risk_register"] = _risk_register(grp["members"], panel)
                if entry["risk_register"]:
                    entry["notes"] += " Named risks: " + " | ".join(
                        e["risk"] for e in entry["risk_register"]) + "."
                else:
                    entry["notes"] += " No risk explicitly named in this group's answers."
            checkpoints.append(entry)

    # --- Checkpoint disagreements (DESIGN A1 + G8) -----------------------
    disagreements = []
    # (i) Verdict departures: none in v0.2 -- verdict scores equal the
    # group's checkpoint scores by construction, so no departure entries.
    # (ii) A fatal risk the feasibility checkpoint's notes do not address.
    for gd in group_data.values():
        grp = gd["group"]
        risk_entry = next(e for e in checkpoints
                          if e["checkpoint_id"] == "risk" and e["strategy_label"] == grp["label"]
                          and (grp["label"] is not None or e.get("panelist_id") == grp["panelist_id"]))
        for risk in risk_entry["risk_register"]:
            if risk["fatal"]:
                disagreements.append({
                    "criterion_id": None,
                    "strategy_label": grp["label"],
                    "checkpoint_scores": None,
                    "resolution": "verdict records the fatal risk; feasibility notes do not address it",
                    "resolution_reason": (
                        f"Risk checkpoint names a fatal risk ({risk['risk']}) that the "
                        "feasibility checkpoint's notes do not address."),
                })
    # Supplied-checkpoint mismatches: any supplied score differing from
    # the synthesised score for its group (or the leading group when the
    # supplied entry names no strategy_label) is recorded, with the
    # synthesised score as the resolution.
    for sc in (supplied_checkpoints or []):
        if sc.get("strategy_label") is not None:
            target = next((gd for gd in group_data.values()
                           if gd["group"]["label"] == sc["strategy_label"]), None)
            targets = [target] if target is not None else []
        else:
            targets = [top]
        for gd in targets:
            for cid, supplied_score in sc["scores"].items():
                synth = int(gd["scores"][cid])
                if supplied_score != synth:
                    disagreements.append({
                        "criterion_id": cid,
                        "strategy_label": gd["group"]["label"],
                        "checkpoint_scores": {"supplied": supplied_score, "synthesised": synth},
                        "resolution": synth,
                        "resolution_reason": (
                            f"Supplied {sc['checkpoint_id']} checkpoint score {supplied_score} "
                            f"differs from the synthesised score {synth} (highest supplied "
                            "per-panelist score within the group); synthesised score stands."),
                    })

    # --- Verdict rubric / summary (DESIGN A7) ----------------------------
    if consensus == "no-consensus" and tied_top:
        rubric_out, overall = [], None
    else:
        rubric_out = []
        for c in rubric:
            cid = c["id"]
            supporters = verdict_supporters.get(cid, [])
            if consensus == "no-consensus":
                reason = (f"Leading (not recommended) group's checkpoint score for {cid}, "
                          f"best-supported by {', '.join(supporters) if supporters else 'none'}; "
                          "see per-panelist reasons for the underlying scores.")
            else:
                reason = (f"Best-supported version across the recommended strategy's panelists "
                          f"({', '.join(supporters) if supporters else 'none'}); "
                          "see per-panelist reasons for the underlying scores.")
            rubric_out.append({"criterion_id": cid, "score": int(verdict_scores[cid]),
                               "weight": c["weight"], "reason": reason})
        overall = round(verdict_group["overall"], 1)
    if consensus == "no-consensus":
        if tied_top:
            leaders = [gd for gd in ranked if _round9(gd["overall"]) == _round9(top["overall"])]
            leader_names = ", ".join(gd["group"]["label"] or gd["group"]["panelist_id"]
                                     for gd in leaders)
            summary = (
                f"No strategy is recommended. The top is tied at an overall score of "
                f"{top['overall']:.1f} between {leader_names}, so no group leads and no "
                "recommendation is forced. See the per-panelist scores and dissent for "
                "the closest options.")
        else:
            leader_name = leading_group["label"] or leading_group["panelist_id"]
            summary = (
                f"No strategy is recommended. The leading strategy group is {leader_name} "
                f"with a leading overall score of {top['overall']:.1f}"
                + (f" and a margin of {margin:.1f} over the next group" if margin is not None else "")
                + ", which does not lead clearly on the rubric, so no recommendation is "
                "forced. The rubric scores shown are the leading, not recommended, scores. "
                "See the per-panelist scores and dissent for the closest options.")
    else:
        best_member = max(verdict_group["group"]["members"],
                          key=lambda pid: _weighted_mean(per_panelist[pid]["scores"], rubric))
        best_answer = by_id[best_member]["answer"]
        if recommended_label is not None:
            who = (f"Recommended strategy ({recommended_label}, supported by "
                   f"{len(verdict_group['group']['members'])} of {len(panel_ids)} panelists: "
                   f"{', '.join(verdict_group['group']['members'])}): ")
        else:
            who = (f"Recommended strategy (panelist {verdict_group['group']['panelist_id']}, "
                   "the sole panelist in this unlabeled group): ")
        summary = (who + best_answer +
                   " Shared supporting elements across the group include the highest-scoring "
                   "evidence, efficiency, regulatory, differentiation and risk points named "
                   "in the per-panelist reasons.")

    # --- Per-panelist entries (DESIGN G9) --------------------------------
    per_out = []
    for p in panel["panelists"]:
        entry = per_panelist[p["id"]]
        scores = entry["scores"]
        reasons = {}
        for c in rubric:
            cid = c["id"]
            reasons[cid] = entry.get("reasons", {}).get(cid, f"No reason supplied for {cid}.")
        strengths = entry.get("strengths") or [cid for cid, s in scores.items() if s >= 4]
        weaknesses = entry.get("weaknesses") or [cid for cid, s in scores.items() if s <= 2]
        # G19: strategy labels in the verdict are the STRIPPED form (the
        # grouping form); panelist IDs are copied verbatim, never stripped.
        raw_label = p.get("strategy_label")
        stripped_label = raw_label.strip() if isinstance(raw_label, str) and raw_label.strip() else None
        per_out.append({
            "panelist_id": p["id"],
            "strategy_label": stripped_label,
            "answer_kind": p.get("answer_kind") or "summary",
            "scores": {cid: int(scores.get(cid, 0)) for cid in rubric_ids if cid in scores},
            "reasons": reasons,
            "overall_score": round(_weighted_mean(scores, rubric), 1),
            "strengths": strengths,
            "weaknesses": weaknesses,
        })

    # --- Dissent (DESIGN A5 + A6) ----------------------------------------
    baseline_group = top["group"] if consensus == "no-consensus" else verdict_group["group"]
    baseline_markets = set()
    for pid in baseline_group["members"]:
        baseline_markets |= _market_set(by_id[pid]["answer"])
    second_key = ranked[1]["group"]["key"] if len(ranked) > 1 else None
    dissent, dissent_ids, excluded_flagged = [], set(), []
    for p in panel["panelists"]:
        grp = next(g for g in groups if p["id"] in g["members"])
        is_winner_member = consensus == "minority" and grp["key"] == recommended_key
        if consensus == "no-consensus":
            if tied_top:
                include, basis = True, "tied top: no group leads"
            elif grp["key"] == baseline_group["key"]:
                others_markets = set()
                for pid in grp["members"]:
                    if pid != p["id"]:
                        others_markets |= _market_set(by_id[pid]["answer"])
                my_markets = _market_set(p["answer"])
                if my_markets and others_markets and not (my_markets & others_markets):
                    include = True
                    basis = (f"same strategy_label but materially different target market "
                             f"({sorted(my_markets)} vs {sorted(others_markets)})")
                else:
                    include = False
                    basis = ""
            else:
                include = True
                basis = (f"different strategy_label ({grp['label'] or 'none'} vs "
                         f"{baseline_group['label'] or 'none'})")
                my_markets = _market_set(p["answer"])
                if my_markets and baseline_markets and not (my_markets & baseline_markets):
                    basis += f"; different target market ({sorted(my_markets)} vs {sorted(baseline_markets)})"
        else:
            if grp["key"] == recommended_key:
                if is_winner_member:
                    include = True
                    basis = ("minority strategy that became the recommendation; its panelists "
                             "remain listed as dissent so the win stays traceable")
                else:
                    others_markets = set()
                    for pid in grp["members"]:
                        if pid != p["id"]:
                            others_markets |= _market_set(by_id[pid]["answer"])
                    my_markets = _market_set(p["answer"])
                    if my_markets and others_markets and not (my_markets & others_markets):
                        include = True
                        basis = (f"same strategy_label but materially different target market "
                                 f"({sorted(my_markets)} vs {sorted(others_markets)})")
                    else:
                        include = False
                        basis = ""
            else:
                include = True
                basis = (f"different strategy_label ({grp['label'] or 'none'} vs "
                         f"{recommended_label or 'none'})")
                my_markets = _market_set(p["answer"])
                if my_markets and baseline_markets and not (my_markets & baseline_markets):
                    basis += f"; different target market ({sorted(my_markets)} vs {sorted(baseline_markets)})"
        if not include:
            # Flag vs finding (A5): a flagged panelist with no material
            # difference is excluded, and listed in the report footer.
            if p.get("is_dissent"):
                excluded_flagged.append({
                    "panelist_id": p["id"],
                    "reason": ("Flagged as dissent in the panel file but no material difference "
                               "from the recommendation/leading group was found in the answer text."),
                })
            continue
        mine = per_panelist[p["id"]]["scores"]
        if is_winner_member:
            response = (f"adopted: the dissenting strategy became the recommendation "
                        f"(consensus minority, overall {top['overall']:.1f}).")
        elif consensus == "no-consensus":
            response = (f"unresolved: no recommendation was made (leading overall "
                        f"{top['overall']:.1f}"
                        + (f", margin {margin:.1f}" if margin is not None else ", top is tied")
                        + "), so this strategy did not become a recommendation.")
        elif grp["key"] == second_key:
            # review P1-12: the old text named the dissenter's own
            # weakest criterion and claimed it was below the
            # recommendation -- false when that criterion ties. Name
            # the criterion with the LARGEST positive gap to the
            # recommendation (rubric order breaks ties), or say
            # neutrally that no single criterion is lower when there
            # is no positive gap.
            gaps = {cid: int(verdict_scores[cid]) - int(mine[cid]) for cid in rubric_ids}
            best_gap_cid = max(rubric_ids, key=lambda cid: gaps[cid]) if rubric_ids else None
            if best_gap_cid is not None and gaps[best_gap_cid] > 0:
                gap_text = (f"it scores lower on {best_gap_cid} "
                            f"({mine[best_gap_cid]} vs {verdict_scores[best_gap_cid]} for the recommendation)")
            else:
                gap_text = ("it scores no lower than the recommendation on any single criterion; "
                            "its lower overall reflects the group scores as a whole")
            response = (
                f"partially adopted as fallback: this is the second-ranked strategy "
                f"(overall {ranked[1]['overall']:.1f} vs {top['overall']:.1f} for the recommendation). "
                "It is kept as a fallback if the recommended strategy's demand proof fails, but its "
                f"niche is not recommended: {gap_text}.")
        else:
            gaps = {cid: int(verdict_scores[cid]) - int(mine[cid]) for cid in rubric_ids}
            best_gap_cid = max(rubric_ids, key=lambda cid: gaps[cid]) if rubric_ids else None
            if best_gap_cid is not None and gaps[best_gap_cid] > 0:
                response = (f"rejected on {best_gap_cid}: the dissenting answer scores {mine[best_gap_cid]} there, "
                            f"below the recommended strategy's {verdict_scores[best_gap_cid]}.")
            else:
                response = ("rejected: the dissenting answer scores no lower than the recommended "
                            "strategy on any single criterion; it is not recommended because its "
                            "overall group score is lower.")
        dissent.append({
            "panelist_id": p["id"],
            "strategy_label": grp["label"],
            "summary": p["answer"],
            "material_difference_basis": basis,
            "verdict_response": response,
        })
        dissent_ids.add(p["id"])

    # --- State (DESIGN G4): verdict.state directly, never double-nested --
    # Self-judging is disclosed ONLY via assessor_panelist_id matching a
    # panelist id (G8-partial). Name/substring matching was removed in v0.2.
    self_judging = assessor_panelist_id is not None and assessor_panelist_id in panel_ids
    # review P1-9: declared-id detection is not the only signal.
    # If the assessor model string exactly equals a panelist's model
    # string, that is a warning (not a self_judging=True finding: a
    # model name match alone does not prove the same model judged
    # itself, but it must not pass silently).
    _assessor_model = (envelope or {}).get("assessor_model")
    model_match_ids = [p["id"] for p in panel["panelists"]
                       if _assessor_model is not None and p.get("model") == _assessor_model]
    if model_match_ids:
        if warnings is None:
            warnings = []
        warnings.append(
            f"assessor_model {_assessor_model!r} matches panelist model(s) {', '.join(model_match_ids)}: "
            "possible self-judging signal; check assessor_panelist_id")
    # Re-review N-2: the assessor's rubric is bound, not assumed.
    # If the assessments file declares a rubric_version different
    # from the rubric used at judging time, the scores were made
    # against different anchor text. That is a warning, not a
    # failure: the normal helper workflow (helper
    # "ollama-builtin-v1" anchors judged under core "default-v1")
    # would otherwise be unusable, and old files carry no
    # rubric_version at all (unknown, not claimed). The mismatch is
    # recorded in state.assessor_rubric_version and surfaced here
    # and in the report, so it can never be read as "same rubric".
    _assessor_rubric = (envelope or {}).get("rubric_version")
    if _assessor_rubric is not None and _assessor_rubric != rubric_version:
        if warnings is None:
            warnings = []
        warnings.append(
            f"assessor rubric_version {_assessor_rubric!r} differs from judging rubric "
            f"{rubric_version!r}: scores were made against the assessor's anchors, "
            "not the judging rubric text")
    panel_chars = len(panel_raw.decode("utf-8", errors="replace"))
    if truncated_info.get("truncated"):
        after = panel_chars - sum(
            truncated_info["original_answer_chars"].get(pid, 0)
            - truncated_info["truncated_answer_chars"].get(pid, 0)
            for pid in truncated_info["truncated_panelist_ids"])
    else:
        after = None
    state = {
        "panel_sha256": _sha256_bytes(panel_raw),
        "panel_file": str(panel_path),
        "rubric_version": rubric_version,
        "rubric_file": str(rubric_file) if rubric_file is not None else None,
        "assessments_sha256": assessments_sha256,
        "assessments_file": str(assessments_file) if assessments_file is not None else None,
        "assessor": assessor,
        "assessor_panelist_id": assessor_panelist_id,
        "judging_route": judging_route,
        "judge_model": judge_model,
        "ollama_endpoint": ollama_endpoint,
        "prompt_version": prompt_version,
        "max_panel_chars": budget_chars,
        "max_answer_chars": max_answer_chars,
        "max_assessments_chars": max_assessments_chars,
        "input_budget_chars": budget_chars,
        "panel_chars": panel_chars,
        "panel_chars_after_truncation": after,
        "assessments_chars": assessments_chars,
        "truncated": bool(truncated_info.get("truncated")),
        "truncated_panelist_ids": list(truncated_info.get("truncated_panelist_ids", [])),
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tool_version": TOOL_VERSION,
        "self_judging": self_judging,
        "assessor_model_matches_panelist_model": model_match_ids,
        # v1.0 additive state fields (null where the route does not apply).
        "assessor_model": (envelope or {}).get("assessor_model"),
        "assessor_provider": (envelope or {}).get("assessor_provider"),
        "prompt_sha256": (envelope or {}).get("prompt_sha256"),
        "temperature": (envelope or {}).get("temperature"),
        "usage": (envelope or {}).get("usage"),
        "heuristic_mapping_version": HEURISTIC_MAPPING_VERSION if judging_route == "heuristic" else None,
        "alias_table_version": ALIAS_TABLE_VERSION if judging_route == "supplied-assessments" else None,
        "rubric_criteria": rubric_ids,
        "checkpoint_mapping": dict(checkpoint_mapping),
        "verification_sha256": verification_sha256,
        "verification_file": verification_file,
        "verification_outcome": (verification_record["verification_outcome"]
                                  if verification_record is not None else "not-run"),
        # Re-review P1-10: coverage travels in state so sign-off can
        # enforce it (see cli.cmd_sign_off); None means no
        # verification was run, not "fully covered".
        "verification_coverage": (verification_record["verification_coverage"]
                                  if verification_record is not None else None),
        # Re-review N-2: the assessor's rubric, as declared in the
        # assessments envelope; None means the file did not declare
        # one (all pre-v4 files). Distinct from rubric_version above,
        # which is the rubric used at judging time.
        "assessor_rubric_version": _assessor_rubric,
        "sign_off": None,
    }
    if truncated_info.get("truncated"):
        state["original_answer_chars"] = truncated_info["original_answer_chars"]
        state["truncated_answer_chars"] = truncated_info["truncated_answer_chars"]
    if assessor_role:
        state["assessor_role"] = assessor_role
    verdict = {
        "panel_id": panel["panel_id"],
        "question": panel["question"],
        "verdict": {
            "recommended_strategy_label": recommended_label,
            "leading_strategy_label": leading_label,
            "recommended_strategy_summary": summary,
            "consensus": consensus,
            "overall_score": overall,
        },
        "rubric": rubric_out,
        "checkpoints": checkpoints,
        "checkpoint_disagreements": disagreements,
        "per_panelist": per_out,
        "dissent": dissent,
        "disclaimer": DISCLAIMER,
        "state": state,
        # G22: v0.2 key order is a prefix; verification is appended after state.
        "verification": verification_record,
    }
    # Output validation (DESIGN §10 rules 14-16, trace per G3): every
    # verdict rubric score must equal the verdict group's checkpoint score
    # for that criterion, OR its reason must name the checkpoint, the old
    # score, and why it changed (a departure). v1.0 produces no departures.
    cp_by_group = {}
    for cp in checkpoints:
        key = (cp["checkpoint_id"], cp["strategy_label"], cp.get("panelist_id"))
        cp_by_group[key] = cp
    verdict_group_key = verdict_group["group"]
    for row in rubric_out:
        if not isinstance(row["score"], int) or not 0 <= row["score"] <= 5 or not row["reason"]:
            raise JudgeError("output validation failed: verdict rubric score invalid")
        traced = False
        cp_id = checkpoint_mapping[row["criterion_id"]]
        cp = cp_by_group.get((cp_id, verdict_group_key["label"],
                              verdict_group_key["panelist_id"] if verdict_group_key["label"] is None else None))
        if cp is not None and row["criterion_id"] in cp["scores"]:
            if cp["scores"][row["criterion_id"]] == row["score"]:
                traced = True
            elif cp_id in row["reason"] and str(cp["scores"][row["criterion_id"]]) in row["reason"]:
                traced = True  # departure naming checkpoint + old score, reason is the why
        if not traced:
            raise JudgeError(f"output validation failed: rubric score for {row['criterion_id']} traces to no checkpoint")
    if verdict["disclaimer"] != DISCLAIMER:
        raise JudgeError("output validation failed: disclaimer missing or altered")
    return verdict, warnings or [], excluded_flagged


def render_report(verdict, judging_route, judge_model, warnings, heuristic_used,
                  excluded_flagged=None):
    """Render report.md from verdict.json (v1.0 rules).

    P0-5: dissent summaries are blockquoted panelist text, labelled with
    panelist ID + answer_kind; the recommended summary stays verdict prose
    with a lead-in naming the best panelist(s) + answer_kind, and is
    NEVER split by string search. P0-6: the sign-off sentence is
    as-of-judging only. G13: rubric rows == len(rubric) (zero on a tied
    no-consensus verdict, with a sentence instead of an empty table);
    per-panelist notes == len(per_panelist); empty strengths/weaknesses
    render as "none noted", never invented.
    """
    v = verdict["verdict"]
    st = verdict["state"]
    kinds = {pp["panelist_id"]: pp["answer_kind"] for pp in verdict["per_panelist"]}
    kind_set = set(kinds.values())
    if kind_set == {"summary"}:
        answer_kind_sentence = ("Judged on summary answers \u2014 a verdict over summaries is a verdict "
                                "over summaries: it must not claim to have judged the panelists' full reasoning.")
    elif kind_set == {"full"}:
        answer_kind_sentence = "Judged on full answers."
    else:
        answer_kind_sentence = ("Judged on mixed summary and full answers \u2014 a verdict over summaries is a verdict "
                                "over summaries: it must not claim to have judged the panelists' full reasoning.")
    lines = [f"# Panel verdict: {verdict['panel_id']}", "",
             f"**Question:** {verdict['question']}", "",
             answer_kind_sentence, "",
             "Human sign-off: none at judging time \u2014 check sign-off.json next to verdict.json "
             "for current approval status; a verdict without a matching sign-off record must not "
             "be presented as approved.", ""]
    if v["recommended_strategy_label"] is None and v["consensus"] == "no-consensus":
        lines.append("**Recommendation: no consensus.** No strategy leads clearly on the rubric, so no winner is forced.")
    elif v["recommended_strategy_label"] is None:
        lines.append(f"**Recommended strategy:** panelist {v['leading_strategy_label'] or 'unlabeled group'} "
                     f"(consensus: {v['consensus']}, overall score: {v['overall_score']:.1f} / 5)")
    else:
        lines.append(f"**Recommended strategy:** {v['recommended_strategy_label']} "
                     f"(consensus: {v['consensus']}, overall score: {v['overall_score']:.1f} / 5)")
    lines.append("")
    # Recommended-summary lead-in (P0-5): best-supported panelist(s) in
    # the recommended group + answer_kind, from per_panelist only. The
    # summary string itself is never split.
    if v["consensus"] != "no-consensus" and v["recommended_strategy_label"] is not None:
        members = [pp for pp in verdict["per_panelist"]
                   if pp["strategy_label"] == v["recommended_strategy_label"]]
        if members:
            best_score = max(pp["overall_score"] for pp in members)
            best = [pp for pp in members if pp["overall_score"] == best_score]
            best_desc = ", ".join(f"{pp['panelist_id']} ({pp['answer_kind']})" for pp in best)
            lines.append(f"Best-supported panelist(s) in the recommended strategy: {best_desc}.")
            lines.append("")
    lines.append(v["recommended_strategy_summary"])
    lines.append("")
    lines.append("## Rubric")
    lines.append("")
    is_extended = bool(st.get("rubric_version", "").startswith("custom-extended:"))
    if verdict["rubric"]:
        if is_extended:
            lines.append("| Criterion | Checkpoint | Weight | Score (0-5) | Reason |")
            lines.append("|---|---|---|---|---|")
            mapping = st.get("checkpoint_mapping") or {}
            for row in verdict["rubric"]:
                lines.append(f"| {row['criterion_id']} | {mapping.get(row['criterion_id'], '')} | {row['weight']} | {row['score']} | {row['reason']} |")
        else:
            lines.append("| Criterion | Weight | Score (0-5) | Reason |")
            lines.append("|---|---|---|---|")
            for row in verdict["rubric"]:
                lines.append(f"| {row['criterion_id']} | {row['weight']} | {row['score']} | {row['reason']} |")
    else:
        lines.append("No rubric table: the top is tied, so no leading scores are shown for a tied no-consensus verdict.")
    lines.append("")
    lines.append("## Checkpoints (independent, before the verdict)")
    lines.append("")
    for cp in verdict["checkpoints"]:
        label = cp["strategy_label"] or cp.get("panelist_id") or "(unlabeled)"
        lines.append(f"### {cp['checkpoint_id']} \u2014 {label}")
        lines.append("")
        lines.append(f"Scores: {cp['scores']}")
        lines.append("")
        lines.append(cp["notes"])
        lines.append("")
        if cp["checkpoint_id"] == "risk":
            lines.append("Risk register:")
            lines.append("")
            if cp.get("risk_register"):
                for r in cp["risk_register"]:
                    lines.append(f"- {r['risk']} (named by: {', '.join(r['named_by'])}; "
                                 f"fatal: {r['fatal']}; response in answers: "
                                 f"{r['response_in_answers'] or 'none stated'})")
            else:
                lines.append("- No risk explicitly named.")
            lines.append("")
    lines.append("## Checkpoint disagreements")
    lines.append("")
    if not verdict["checkpoint_disagreements"]:
        lines.append("There were no material checkpoint disagreements.")
    for d in verdict["checkpoint_disagreements"]:
        lines.append(f"- Criterion: {d['criterion_id']}; strategy: {d['strategy_label']}; "
                     f"checkpoint scores: {d['checkpoint_scores']}; resolution: {d['resolution']}; "
                     f"reason: {d['resolution_reason']}")
    lines.append("")
    lines.append("## Per-panelist notes")
    lines.append("")
    for pp in verdict["per_panelist"]:
        lines.append(f"- **{pp['panelist_id']}** ({pp['strategy_label']}, {pp['answer_kind']}, "
                     f"overall {pp['overall_score']:.1f}): scores {pp['scores']}; "
                     f"strengths: {', '.join(str(x) for x in pp['strengths']) if pp['strengths'] else 'none noted'}; "
                     f"weaknesses: {', '.join(str(x) for x in pp['weaknesses']) if pp['weaknesses'] else 'none noted'}.")
    lines.append("")
    lines.append("## Dissent")
    lines.append("")
    if not verdict["dissent"]:
        lines.append("No material dissent.")
    for d in verdict["dissent"]:
        lines.append(f"### {d['panelist_id']} ({d['strategy_label']})")
        lines.append("")
        kind = kinds.get(d["panelist_id"], "summary")
        kind_label = "summary paraphrase" if kind == "summary" else "full answer"
        for para in str(d["summary"]).splitlines() or [""]:
            lines.append(f"> [{d['panelist_id']}, {kind_label}]: {para}" if para else ">")
        lines.append("")
        lines.append(f"Material difference basis: {d['material_difference_basis']}")
        lines.append("")
        lines.append(f"Verdict response: {d['verdict_response']}")
        lines.append("")
    lines.append("## Disclaimer")
    lines.append("")
    lines.append(verdict["disclaimer"])
    lines.append("")
    lines.append("## Reproducibility")
    lines.append("")
    lines.append(f"- Panel file: {st['panel_file']} (SHA-256: {st['panel_sha256']})")
    lines.append(f"- Rubric version: {st['rubric_version']}" + (f" (file: {st['rubric_file']})" if st.get("rubric_file") else ""))
    if st.get("assessor_rubric_version") is not None and st["assessor_rubric_version"] != st["rubric_version"]:
        lines.append(f"- Assessor rubric version: {st['assessor_rubric_version']} (differs from the judging rubric above; "
                     "scores were made against the assessor's anchors, not the judging rubric text)")
    if st.get("rubric_criteria"):
        lines.append(f"- Rubric criteria: {', '.join(st['rubric_criteria'])}")
    lines.append(f"- Judging route: {judging_route}" + (f" (model: {judge_model})" if judge_model else ""))
    if st.get("assessor_model"):
        lines.append(f"- Assessor model: {st['assessor_model']} (provider: {st.get('assessor_provider')})")
    if st.get("assessor"):
        lines.append(f"- Assessor: {st['assessor']} (supplied scores are the assessor's responsibility; the library does not check whether supplied reasons are true)")
    if st.get("self_judging"):
        name = st.get("assessor") or st.get("assessor_panelist_id")
        lines.append(f"- SELF-JUDGING DISCLOSURE: The assessor {name} also sat on this panel as {st.get('assessor_panelist_id')}.")
    if st.get("prompt_version"):
        lines.append(f"- Prompt version: {st['prompt_version']}")
    if st.get("prompt_sha256"):
        lines.append(f"- Prompt SHA-256 (template, before panel answers inserted): {st['prompt_sha256']}")
    if st.get("temperature") is not None:
        lines.append(f"- Temperature: {st['temperature']}")
    if judging_route == "supplied-assessments":
        usage = st.get("usage") or {}
        if usage.get("cost_amount") is not None:
            lines.append(f"- Cost (provider-reported): {usage['cost_amount']} {usage.get('cost_currency')} "
                         f"(pricing source: {usage.get('pricing_source')})")
        else:
            lines.append("- Cost: cost not reported by the provider")
    if st.get("heuristic_mapping_version"):
        lines.append(f"- Heuristic mapping version: {st['heuristic_mapping_version']}")
    lines.append(f"- Input budget (chars): max panel {st['max_panel_chars']}, max answer {st['max_answer_chars']}, "
                 f"max assessments {st['max_assessments_chars']}; panel chars: {st['panel_chars']}"
                 + (f" (after truncation: {st['panel_chars_after_truncation']})" if st.get("panel_chars_after_truncation") is not None else "")
                 + f"; truncated: {st['truncated']}"
                 + (f"; truncated panelists: {', '.join(st['truncated_panelist_ids'])}" if st.get("truncated_panelist_ids") else ""))
    lines.append(f"- Created: {st['created']}; tool: {st['tool_version']}")
    if is_extended:
        added = [cid for cid in (st.get("rubric_criteria") or []) if cid not in CRITERIA]
        lines.append(f"- Extended rubric: {len(st.get('rubric_criteria') or [])} criteria including {', '.join(added)}. "
                     "Overall scores are not comparable with canonical-rubric verdicts.")
    if heuristic_used:
        lines.append("- Heuristic route limits: the heuristic scorer rewards specificity and structure; "
                     "it cannot tell whether a cited precedent is real or a regulation actually applies. "
                     "It is a reproducible baseline, not a substitute for human judgement on a real decision.")
    if judging_route == "supplied-assessments":
        lines.append("- Supplied-assessments route limits: the library does not check whether the "
                     "supplied reasons are true; that is the assessor's responsibility, and scores "
                     "inherit the assessor's own accuracy limits.")
    if warnings:
        lines.append("- Warnings:")
        for w in warnings:
            lines.append(f"  - {w}")
    if excluded_flagged:
        lines.append("- Excluded flagged panelists (flagged as dissent in the panel file, but no material "
                     "difference found):")
        for e in excluded_flagged:
            lines.append(f"  - {e['panelist_id']}: {e['reason']}")
    lines.append("")
    # Verification section AFTER the reproducibility footer, only when run.
    verification = verdict.get("verification")
    if verification is not None:
        lines.append("## Verification (not scoring)")
        lines.append("")
        lines.append(f"Verifier: {verification['verifier']}; verified_at: {verification.get('verified_at') or 'not recorded'}.")
        lines.append("")
        for chk in verification["checks"]:
            kind_label = "verbatim" if chk["claim_is_verbatim"] else "paraphrase"
            lines.append(f"- Panelist {chk['panelist_id']} [{kind_label} excerpt]: {chk['claim_excerpt']}")
            lines.append(f"  Source: {chk['source_ref']}; status: {chk['status']}; note: {chk['evidence_note']}")
        lines.append("")
        counts = verification["summary_counts"]
        lines.append(f"Summary counts: verified {counts['verified']}, contradicted {counts['contradicted']}, "
                     f"unverifiable {counts['unverifiable']}, not_checked {counts['not_checked']}.")
        cov = verification.get("verification_coverage")
        if cov is not None:
            lines.append(f"Coverage: {len(cov['panelists_checked'])} of {cov['panelists_total']} panelists checked"
                         + (f"; unchecked: {', '.join(cov['panelists_unchecked'])}" if cov["panelists_unchecked"] else "; all panelists checked")
                         + ". A 'passed' outcome with partial coverage is passed only for the checked claims.")
        outcome = verification["verification_outcome"]
        if outcome == "passed":
            when = verification.get("verified_at") or st["created"][:10]
            # Re-review P1-10 full: the passed sentence itself must
            # not read as panel-wide when coverage is incomplete.
            if cov is not None and not cov["coverage_complete"]:
                lines.append(f"The claims listed in the verification section were checked against the sources listed there on {when}; "
                             f"verification passed ONLY for those checked claims ({len(cov['panelists_checked'])} of "
                             f"{cov['panelists_total']} panelists checked) -- it is NOT panel-wide verification. "
                             "The verdict's scores were not changed by verification.")
            else:
                lines.append(f"The claims listed in the verification section were checked against the sources listed there on {when}; "
                             "the verdict's scores were not changed by verification.")
        else:
            unv = counts["unverifiable"] + counts["not_checked"]
            lines.append(f"Verification ran with outcome {outcome}; see the verification section \u2014 "
                         f"{counts['contradicted']} claim(s) contradicted, {unv} claim(s) unverifiable or not checked.")
        lines.append("")
    return "\n".join(lines)
