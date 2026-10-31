#!/usr/bin/env python3
"""Ollama route for the generic panel-judge library.

This module lets a local Ollama model produce the assessments JSON that
the panel-judge core CLI consumes. It contains no panel question, no
verdict, no API key and no other secret: it only turns an input panel
JSON into a scoring prompt, sends that prompt to Ollama, and validates
the returned JSON. On any failure it exits non-zero with a clear error
and never emits a fake assessment.

Panel input format
------------------
A JSON object with a question and a list of panelists::

    {
      "question": "What should we do about ...?",
      "panelists": [
        {"id": "panelist-a", "answer": "..."},
        {"id": "panelist-b", "answer": "..."}
      ]
    }

A panelist's identifier is taken from ``id`` if present, otherwise
``name`` or ``panelist``; the answer text is taken from ``answer`` if
present, otherwise ``text`` or ``response``. Panelists without an
identifier fall back to ``panelist-1``, ``panelist-2``, ... in order.

Assessments output format
-------------------------
A JSON object with one entry per panelist, each scored 0-5 on every
criterion with a short reason per criterion::

    {
      "assessments": [
        {
          "panelist": "panelist-a",
          "scores": {
            "evidence": 4,
            "efficiency": 3,
            "regulatory": 5,
            "differentiation": 2,
            "risk_mitigation": 4
          },
          "reasons": {
            "evidence": "Cites specific data.",
            "efficiency": "One extra step is unnecessary.",
            "regulatory": "Addresses the relevant rule.",
            "differentiation": "Similar to the standard approach.",
            "risk_mitigation": "Names a fallback."
          }
        }
      ]
    }

An assessment entry's identifier is ``panelist_id`` (preferred, and
the key the prompt skeleton uses); ``panelist``, ``id`` and ``name``
are accepted as legacy aliases, and differing identifier keys in one
entry are rejected.

Host / connectivity notes
-------------------------
Default host ``http://172.22.112.1:11434`` is the Ollama address when
this script runs on AIBeast WSL. Pass ``--host`` to override it.

From the Muse VM, the Tailscale Serve address ``http://aibeast:11434``
is the alternative host.

SSH to AIBeast is as ``muse@aibeast-1`` via the proxy script
``~/workspace/tools/ssh-tailscale-proxy.py`` on port 3130 -- never as
``sivab``, and never via ``aperture`` (a different tailnet device, not
AIBeast).

Usage
-----
    python3 ollama_judge.py --panel panel.json --model qwen3:8b \\
        --host http://172.22.112.1:11434 --out assessments.json
    python3 ollama_judge.py --panel panel.json --model qwen3:8b --dry-run

Pure Python standard library only (urllib, json, argparse, sys).
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request

# review 2026-10-31 review fixes (P1-4, P1-6, re-review P1-6 full /
# N-2): the prompt version bumps to v4 because prompt_sha256 is now
# the SHA-256 of the FULLY RENDERED prompt for a fixed dummy panel
# (template + criteria block + panel framing + skeleton rendering +
# version line) plus the repair suffix -- not a hand-assembled list
# of parts that could drift from what build_prompt actually sends.
# The integrated CLI route remains OUT of scope; this helper is
# standalone only.
PROMPT_VERSION = "ollama-judge-v4"

# The helper's rubric: its own canonical anchors (CRITERION_GUIDANCE),
# recorded in every v4 record/envelope as rubric_version. This is
# deliberately NOT "default-v1": the core default rubric's criterion
# descriptions differ in wording from the helper guidance, so the two
# must never be mistaken for the same rubric text. Core records the
# assessor's rubric_version separately (state.assessor_rubric_version)
# and warns when it differs from the rubric used at judging time;
# custom core rubrics hash as "custom:<sha>" / "custom-extended:<sha>"
# and are therefore distinguishable from this value by construction.
HELPER_RUBRIC_VERSION = "ollama-builtin-v1"

CRITERIA = (
    "evidence",
    "efficiency",
    "regulatory",
    "differentiation",
    "risk_mitigation",
)

# Rubric descriptions and 0-5 anchors, matching DESIGN.md §5, so Ollama
# scores are anchored to the library rubric, not to bare criterion names.
CRITERION_GUIDANCE = {
    "evidence": "Evidence strength: specific, checkable evidence (named precedents, numbers, markets, mechanisms). 0 absent/contradicted, 1 mentioned unsupported, 2 thin, 3 adequate with a gap, 4 strong with minor gaps, 5 specific evidence and a concrete mechanism with no material gap.",
    "efficiency": "Capital/resource efficiency: little upfront capital, bought-in components, build-to-order, low volumes, deposits. 0 needs proprietary development/large plant, 5 needs almost none.",
    "regulatory": "Regulatory/legal runway: a named, currently available legal route to market with a plausible life. 0 route already closing or unnamed, 5 named route with long life.",
    "differentiation": "Differentiation: distinct from incumbents in a way buyers would pay for. 0 generic/copyable, 5 uniquely distinctive.",
    "risk_mitigation": "Risk mitigation: biggest risks named, with a concrete response to each (pivot, cap, deposit model, second market). 0 risks unnamed/dismissed, 5 every major risk answered concretely.",
}

# Prompt template. {question}, {panel_answers}, {criteria_block},
# {skeleton} and {prompt_version} are instantiation placeholders.
# The literal version line is rendered from PROMPT_VERSION, so the
# template text and the version constant cannot drift apart (review
# P1-6). prompt_sha256 is the SHA-256 of the FULLY RENDERED prompt
# for a fixed dummy panel (see _rendered_prompt_material) plus the
# repair suffix -- i.e. exactly what build_prompt produces, not a
# hand-assembled list of parts. Anything that shapes the scores
# (template text, guidance wording, criteria-block join format, panel
# framing, skeleton rendering, version line, repair suffix) changes
# the hash, except the panel question/answers themselves (which vary
# per run and are bound separately by panel_sha256).
PROMPT_TEMPLATE = """You are a strict, impartial judge scoring panel answers.

Question:
{question}

Panel answers:
{panel_answers}

Score EACH panelist from 0 to 5 (integer, 0 = worst, 5 = best) on EACH of these criteria, with a short reason for each score:
{criteria_block}

Prompt version: {prompt_version}

Return ONLY a JSON object matching this exact format, with one entry per panelist and every criterion present in both 'scores' and 'reasons'. Do not add any other text, explanation or markdown outside the JSON object.
{skeleton}
"""

# Repair suffix, extracted as a constant so it is part of the hashed
# prompt material (review P1-6) instead of a free-floating string in
# _repair_prompt that could change without a version/SHA change.
_REPAIR_SUFFIX = (
    "\nCORRECTION REQUIRED: your previous output was REJECTED for this reason:\n"
    "{error}\nYour previous output was:\n{raw}\n"
    "Return ONLY a corrected JSON object with one entry for this panelist, "
    "with EVERY criterion (evidence, efficiency, regulatory, "
    "differentiation, risk_mitigation) present in both 'scores' and "
    "'reasons', each score an integer 0-5 and each reason non-empty. "
    "Do not add any other text.\n"
)

# Fixed dummy panel used ONLY to hash the rendered prompt. Its
# question/answer are constants, so the hash covers the rendering
# code path (framing, joins, skeleton dump, trailing newline) without
# depending on any real panel. build_prompt is defined below; this
# function is resolved at call time, not at module import.
_DUMMY_QUESTION = "__prompt_hash_dummy_question__"
_DUMMY_PANELISTS = [{"id": "__dummy_panelist__", "answer": "__dummy_answer__"}]


def _rendered_prompt_material():
    """The exact prompt build_prompt renders, for a fixed dummy panel,
    plus the repair suffix template. Hashing this -- rather than a
    hand-assembled list of parts -- closes the re-review P1-6 gap:
    skeleton/panel framing built in code cannot drift from the hash,
    because the hash IS the code's output."""
    return (build_prompt(_DUMMY_QUESTION, _DUMMY_PANELISTS)
            + "\n--- repair suffix ---\n" + _REPAIR_SUFFIX)


def prompt_sha256():
    return hashlib.sha256(_rendered_prompt_material().encode("utf-8")).hexdigest()


def panel_sha256(question, panelists):
    """Canonical panel binding (review P0-2): question + panelist
    id/answer pairs in panel order, JSON with sorted keys. This is
    the SAME definition as core.canonical_panel_sha256, so a helper
    panel and the core panel for the same question/answers hash
    identically. panelists are the helper's normalised dicts with
    'id' and 'answer' keys.
    """
    canon = {
        "question": question,
        "panelists": [{"id": item["id"], "answer": item["answer"]}
                      for item in panelists],
    }
    return hashlib.sha256(
        json.dumps(canon, sort_keys=True, ensure_ascii=False,
                   separators=(",", ":")).encode("utf-8")).hexdigest()


def _loads_no_duplicates(text, what):
    """json.loads that rejects duplicate object keys (review P1-5)."""
    def _no_dup(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise JudgeError("%s has a duplicate key: %r" % (what, key))
            out[key] = value
        return out
    try:
        return json.loads(text, object_pairs_hook=_no_dup)
    except JudgeError:
        raise
    except json.JSONDecodeError as exc:
        raise JudgeError("%s is not valid JSON: %s" % (what, exc)) from exc


DEFAULT_HOST = "http://172.22.112.1:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_TIMEOUT = 120
# review P1-6 / re-review N-1: the context window is set explicitly
# instead of left to Ollama's default, and the served prompt token
# count is recorded per panelist. The guard FAILS CLOSED: a missing
# prompt_eval_count is a refusal, not a skipped check (Ollama can
# omit it, e.g. on cache hits), and the limit is NUM_CTX minus an
# output margin, because a prompt that leaves no room for the JSON
# answer cannot produce a complete assessment. Residual limit,
# stated not hidden: if Ollama silently truncates AND reports a
# smaller post-truncation count, the count alone cannot detect it;
# the margin only reduces that risk.
NUM_CTX = 16384
OUTPUT_MARGIN = 512
EFFECTIVE_WINDOW_LIMIT = NUM_CTX - OUTPUT_MARGIN


class JudgeError(Exception):
    """Raised for any input, model-output or Ollama failure."""


def load_panel(path):
    """Read and minimally validate a panel JSON file."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            panel = _loads_no_duplicates(handle.read(), "panel file")
    except OSError as exc:
        raise JudgeError("cannot read panel file %r: %s" % (path, exc))
    if not isinstance(panel, dict):
        raise JudgeError("panel file %r must contain a JSON object" % path)
    panelists = panel.get("panelists")
    if not isinstance(panelists, list) or not panelists:
        raise JudgeError("panel file %r needs a non-empty 'panelists' list" % path)
    question = panel.get("question", "")
    if not isinstance(question, str):
        raise JudgeError("panel 'question' must be a string")
    normalised = []
    for index, raw in enumerate(panelists, start=1):
        if not isinstance(raw, dict):
            raise JudgeError("panelist %d is not a JSON object" % index)
        panelist_id = raw.get("id") or raw.get("name") or raw.get("panelist")
        if panelist_id is not None and not isinstance(panelist_id, str):
            raise JudgeError("panelist %d identifier must be a string" % index)
        answer = raw.get("answer")
        if answer is None:
            answer = raw.get("text")
        if answer is None:
            answer = raw.get("response")
        if not isinstance(answer, str):
            raise JudgeError(
                "panelist %r has no answer text (expected 'answer', "
                "'text' or 'response')" % (panelist_id or index)
            )
        normalised.append(
            {
                "id": panelist_id or "panelist-%d" % index,
                "answer": answer,
            }
        )
    ids = [item["id"] for item in normalised]
    if len(set(ids)) != len(ids):
        raise JudgeError("panelist identifiers must be unique, got: %s" % ids)
    return question, normalised


def build_prompt(question, panelists):
    """Build the scoring prompt sent to the model (template + panel)."""
    panel_answers = "\n\n".join(
        "Panelist: %s\n%s" % (item["id"], item["answer"]) for item in panelists)
    criteria_block = "\n".join(
        "- %s: %s" % (criterion, CRITERION_GUIDANCE[criterion]) for criterion in CRITERIA)
    # review P1-4: skeleton scores are null, not 0. A model that
    # echoes the skeleton and writes only reasons must fail
    # validation on the scores it never wrote.
    skeleton = {
        "assessments": [
            {
                "panelist_id": item["id"],
                "scores": {criterion: None for criterion in CRITERIA},
                "reasons": {criterion: "" for criterion in CRITERIA},
            }
            for item in panelists
        ]
    }
    return PROMPT_TEMPLATE.format(
        question=question, panel_answers=panel_answers,
        criteria_block=criteria_block, skeleton=json.dumps(skeleton, indent=2),
        prompt_version=PROMPT_VERSION) + "\n"


def _strip_model_output(text):
    """Strip markdown fences; fall back to the outermost JSON object."""
    if not isinstance(text, str):
        raise JudgeError("model response was not text")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # ```json ... ``` or plain ``` ... ```
        first_newline = cleaned.find("\n")
        if first_newline == -1:
            raise JudgeError("model returned an unclosed markdown fence")
        cleaned = cleaned[first_newline + 1 :]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[: -len("```")].rstrip()
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]
    return cleaned.strip()


def _clamp_score(value, panelist, criterion):
    """Accept ONLY an int (not bool) in 0-5, or raise (review P1-3).

    The old version coerced and clamped (9 -> 5, -2 -> 0, "3" -> 3,
    4.5 -> 4), changing scores the model did not give, while core
    rejects all of those. A rejected value now goes through the
    retry/repair path instead of being silently rewritten. Floats,
    strings, bools, null, Infinity and NaN are all rejected here
    (floats are not int, so int(round(inf)) can no longer crash).
    """
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5:
        raise JudgeError(
            "panelist %r: score for %r must be an integer 0-5, got %r"
            % (panelist, criterion, value)
        )
    return value


def _fix_criterion_keys(mapping, panelist):
    """Fix criterion-key spelling that changes no content.

    A model that writes ``reg,ulatory`` clearly meant ``regulatory``:
    the score, the reason and the panelist are unchanged, only a comma
    in a key name is wrong. Stripping commas from keys and mapping the
    result to an exact known criterion therefore recovers the intended
    assessment without inventing or altering any score or reason.
    Anything else is left untouched, so strict validation still
    rejects unknown keys, missing criteria and duplicate keys (two
    keys mapping to the same criterion). This is a parser
    normalisation, not a hand-edit of a saved file: the model output,
    the scores and the reasons are unchanged, and the correction is
    deterministic. review P1-2: the correction is no longer only
    "logged by the caller" -- this function now RETURNS the list of
    applied fixes, and _normalise_assessment carries it into the
    per-panelist record and the combined envelope as key_fixes, so a
    corrected run is distinguishable from a clean run. review P1-1:
    the docstring's old claim that "strict validation still rejects
    unknown keys" was false for this module (it only looped over
    CRITERIA and dropped extras); _normalise_assessment now requires
    exact set equality after fixing, so unknown keys are rejected.
    Returns (fixed_mapping, fixes) where fixes are "old->new" strings.
    """
    if not isinstance(mapping, dict):
        return mapping, []
    fixed = {}
    fixes = []
    for key, value in mapping.items():
        clean = key.replace(",", "") if isinstance(key, str) else key
        if clean in CRITERIA and clean != key:
            if clean in fixed:
                raise JudgeError(
                    "assessment for %r has duplicate criterion keys mapping "
                    "to %r" % (panelist, clean)
                )
            fixed[clean] = value
            fixes.append("%s->%s" % (key, clean))
        else:
            if key in fixed:
                raise JudgeError(
                    "assessment for %r has duplicate criterion key %r"
                    % (panelist, key)
                )
            fixed[key] = value
    return fixed, fixes


def _normalise_assessment(raw, panelist):
    """Normalise one assessment entry to the canonical format.

    Returns {"panelist_id", "scores", "reasons", "key_fixes"} where
    key_fixes lists every comma-key correction applied (review P1-2).
    Callers that emit core-facing assessments must strip key_fixes
    from the entry and aggregate it at envelope level (see
    _combined_envelope), because core entry keys are closed.
    review P1-1: after fixing, scores and reasons must be EXACTLY
    the criterion set -- unknown keys (e.g. "regulatry", "foo") are
    rejected here, not dropped.
    """
    if not isinstance(raw, dict):
        raise JudgeError("assessment for %r is not a JSON object" % panelist)
    scores = raw.get("scores")
    reasons = raw.get("reasons")
    fixes = []
    if isinstance(scores, dict):
        scores, f = _fix_criterion_keys(scores, panelist)
        fixes += ["scores:%s" % x for x in f]
    if isinstance(reasons, dict):
        reasons, f = _fix_criterion_keys(reasons, panelist)
        fixes += ["reasons:%s" % x for x in f]
    if scores is None and reasons is None:
        # Accept the nested form {criterion: {"score": n, "reason": s}}.
        # Same key-spelling fix applies to the nested criterion keys.
        raw, f = _fix_criterion_keys(raw, panelist)
        fixes += ["nested:%s" % x for x in f]
        unknown_nested = [k for k in raw
                          if k not in CRITERIA
                          and k not in ("panelist_id", "panelist", "id", "name")]
        if unknown_nested:
            raise JudgeError(
                "assessment for %r has unknown criterion key(s): %s"
                % (panelist, ", ".join(sorted(unknown_nested))))
        scores, reasons = {}, {}
        for criterion in CRITERIA:
            entry = raw.get(criterion)
            if isinstance(entry, dict):
                scores[criterion] = entry.get("score")
                reasons[criterion] = entry.get("reason")
    if not isinstance(scores, dict) or not isinstance(reasons, dict):
        raise JudgeError(
            "assessment for %r needs 'scores' and 'reasons' objects" % panelist
        )
    unknown_scores = [c for c in scores if c not in CRITERIA]
    unknown_reasons = [c for c in reasons if c not in CRITERIA]
    if unknown_scores:
        raise JudgeError(
            "assessment for %r has unknown score criterion key(s): %s"
            % (panelist, ", ".join(sorted(unknown_scores))))
    if unknown_reasons:
        raise JudgeError(
            "assessment for %r has unknown reason criterion key(s): %s"
            % (panelist, ", ".join(sorted(unknown_reasons))))
    missing_scores = [c for c in CRITERIA if c not in scores]
    missing_reasons = [c for c in CRITERIA if c not in reasons]
    if missing_scores:
        raise JudgeError(
            "assessment for %r is missing scores for: %s"
            % (panelist, ", ".join(missing_scores))
        )
    if missing_reasons:
        raise JudgeError(
            "assessment for %r is missing reasons for: %s"
            % (panelist, ", ".join(missing_reasons))
        )
    out_scores = {}
    out_reasons = {}
    for criterion in CRITERIA:
        out_scores[criterion] = _clamp_score(scores[criterion], panelist, criterion)
        reason = reasons[criterion]
        if not isinstance(reason, str) or not reason.strip():
            raise JudgeError(
                "assessment for %r needs a non-empty reason for %r"
                % (panelist, criterion)
            )
        out_reasons[criterion] = reason.strip()
    return {"panelist_id": panelist, "scores": out_scores, "reasons": out_reasons,
            "key_fixes": fixes}


def parse_and_validate(text, expected_ids):
    """Parse model output and validate it against the expected panelists."""
    cleaned = _strip_model_output(text)
    data = _loads_no_duplicates(cleaned, "model output")
    if isinstance(data, list):
        assessments = data
    elif isinstance(data, dict) and isinstance(data.get("assessments"), list):
        assessments = data["assessments"]
    else:
        raise JudgeError(
            "model output must be a JSON object with an 'assessments' list"
        )
    found = {}
    for raw in assessments:
        if not isinstance(raw, dict):
            raise JudgeError("an assessment entry is not a JSON object")
        # panelist_id first (the key build_prompt()'s skeleton uses);
        # panelist, id and name are legacy aliases. Differing
        # identifier keys in one entry are rejected, as in core.
        id_values = {k: raw[k] for k in ("panelist_id", "panelist", "id", "name")
                     if k in raw}
        id_list = list(id_values.values())
        if (not id_list
                or any(not isinstance(v, str) or not v for v in id_list)
                or len(set(id_list)) > 1):
            raise JudgeError("an assessment entry has no usable panelist identifier "
                             "or has differing identifier keys")
        panelist = next(iter(id_values.values()))
        if panelist in found:
            raise JudgeError("duplicate assessment for panelist %r" % panelist)
        found[panelist] = _normalise_assessment(raw, panelist)
    missing = [pid for pid in expected_ids if pid not in found]
    if missing:
        raise JudgeError(
            "model output is missing assessments for: %s" % ", ".join(missing)
        )
    unknown = [pid for pid in found if pid not in expected_ids]
    if unknown:
        raise JudgeError(
            "model output has assessments for unknown panelists: %s"
            % ", ".join(unknown)
        )
    return {"assessments": [found[pid] for pid in expected_ids]}


def call_ollama(host, model, prompt, timeout):
    """POST the prompt to Ollama's /api/generate.

    Returns (response_text, meta) where meta carries the model
    Ollama actually served, the prompt token count, the context
    window requested and the effective window limit (review P1-6 /
    re-review N-1). If the prompt fills the effective window, or
    Ollama omits prompt_eval_count so the check cannot be performed,
    the call FAILS instead of returning scores for answers the model
    may never have seen. NOTE: the return type changed in the review
    fixes (was: bare response text); callers and test mocks must
    unpack the tuple.
    """
    url = host.rstrip("/") + "/api/generate"
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": NUM_CTX},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise JudgeError("Ollama returned HTTP %s for %s" % (exc.code, url))
    except urllib.error.URLError as exc:
        raise JudgeError("cannot reach Ollama at %s: %s" % (url, exc.reason))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise JudgeError("Ollama request failed: %s" % exc)
    if not isinstance(body, dict) or not isinstance(body.get("response"), str):
        raise JudgeError("Ollama response did not contain a 'response' string")
    meta = {"model": body.get("model"), "prompt_eval_count": body.get("prompt_eval_count"),
            "eval_count": body.get("eval_count"), "num_ctx": NUM_CTX,
            "effective_window_limit": EFFECTIVE_WINDOW_LIMIT}
    _pec = meta["prompt_eval_count"]
    # Fail closed (re-review N-1): a missing or non-integer count
    # means the window check cannot be performed, so the run must
    # not claim a clean result. bool is excluded: True == 1 in
    # Python and must never pass as a token count.
    if not isinstance(_pec, int) or isinstance(_pec, bool):
        raise JudgeError(
            "Ollama did not report prompt_eval_count; refusing to score "
            "because the context-window check cannot be performed")
    if _pec >= EFFECTIVE_WINDOW_LIMIT:
        raise JudgeError(
            "Ollama prompt filled the effective context window (%d prompt tokens, "
            "limit %d = num_ctx %d - %d output margin); "
            "refusing to score answers the model may not have seen"
            % (_pec, EFFECTIVE_WINDOW_LIMIT, NUM_CTX, OUTPUT_MARGIN))
    return body["response"], meta


def _check_resume_envelope(data, model, source, panel_sha=None):
    """Check a resumed assessments envelope matches this run.

    review P0-2 / P1-7: assessor_model and prompt_version must match;
    prompt_sha256 is now REQUIRED (not "when present") and must match;
    temperature must be present and 0 (the only temperature this
    helper ever uses); and panel_sha256, when panel_sha is given,
    must equal the current panel's canonical SHA. Re-review N-2:
    rubric_version is likewise REQUIRED for v4 records and must
    equal HELPER_RUBRIC_VERSION, so a record made against different
    anchors cannot be resumed as if it were this helper's. Host is
    deliberately NOT checked: it is transport, not content, so a partial
    run recorded via Tailscale Serve (http://aibeast:11434) remains
    resumable from AIBeast WSL (http://172.22.112.1:11434) and vice
    versa. A mismatch rejects the WHOLE source, never just the bad
    entry, so a corrupt partial can never be silently merged.
    """
    # Combined envelopes use "assessor_model"; per-panelist records
    # use "model". Either is accepted, but a present value must match.
    model_value = data.get("assessor_model", data.get("model"))
    if model_value != model:
        raise JudgeError(
            "resume source %s: assessor_model/model does not match this "
            "run's --model; refusing to merge" % source)
    if data.get("prompt_version") != PROMPT_VERSION:
        raise JudgeError(
            "resume source %s: prompt_version does not match this helper's "
            "prompt version; refusing to merge" % source)
    sha = data.get("prompt_sha256")
    if sha is None:
        raise JudgeError(
            "resume source %s: prompt_sha256 is required; refusing to merge" % source)
    if sha != prompt_sha256():
        raise JudgeError(
            "resume source %s: prompt_sha256 does not match this helper; "
            "refusing to merge" % source)
    if data.get("temperature") != 0:
        raise JudgeError(
            "resume source %s: temperature must be 0 for this helper; "
            "refusing to merge" % source)
    if panel_sha is not None:
        psha = data.get("panel_sha256")
        if psha is None:
            raise JudgeError(
                "resume source %s: panel_sha256 is required; refusing to merge" % source)
        if psha != panel_sha:
            raise JudgeError(
                "resume source %s: panel_sha256 does not match this panel; "
                "the resumed assessments were made for a different panel; "
                "refusing to merge" % source)
    if data.get("rubric_version") != HELPER_RUBRIC_VERSION:
        raise JudgeError(
            "resume source %s: rubric_version must be %r for this helper; "
            "refusing to merge" % (source, HELPER_RUBRIC_VERSION))


def _validate_resumed_assessment(raw, panelist_id, source):
    """Strictly re-validate one resumed assessment entry.

    Returns the canonical assessment dict. Any failure raises
    JudgeError naming the source, so a planted negative-control file
    (broken JSON, missing criterion, empty reason, wrong panelist) is
    detected and stops the run instead of being silently passed or
    overwritten.
    """
    if not isinstance(raw, dict):
        raise JudgeError("resume source %s: assessment is not a JSON object" % source)
    # Per-panelist record form: {"assessment": {...}, ...}
    entry = raw.get("assessment") if "assessment" in raw else raw
    if not isinstance(entry, dict):
        raise JudgeError("resume source %s: assessment entry is not a JSON object" % source)
    id_values = {k: entry[k] for k in ("panelist_id", "panelist", "id", "name")
                 if k in entry}
    if (not id_values or any(not isinstance(v, str) or not v for v in id_values.values())
            or len(set(id_values.values())) > 1):
        raise JudgeError("resume source %s: entry has no usable panelist identifier "
                         "or has differing identifier keys" % source)
    pid = next(iter(id_values.values()))
    if pid != panelist_id:
        raise JudgeError("resume source %s: assessment is for panelist %r, "
                         "expected %r" % (source, pid, panelist_id))
    normalised = _normalise_assessment(entry, pid)
    # Preserve the record-level key-fix trace (review P1-2): a resumed
    # assessment's own keys are already canonical, so re-normalising
    # would report no fixes and erase the original correction.
    if isinstance(raw.get("key_fixes"), list):
        normalised["key_fixes"] = list(raw["key_fixes"])
    return normalised


def load_resume_source(path, panelists, model, panel_sha=None):
    """Load resumed assessments from --resume-from (file or directory).

    File form: a combined envelope as this helper emits (an object with
    an 'assessments' list), or a single per-panelist record. Directory
    form: every assessment-*.json file in the directory. In all forms
    the WHOLE source is rejected on any invalid entry (mirroring
    openrouter P0-2): no entry is reused from a corrupt source.
    Returns a dict panelist_id -> canonical assessment, in panel order
    filtered to panelists in the current panel. Unknown panelists in
    the source reject the source.
    """
    panel_ids = [item["id"] for item in panelists]
    resumed = {}
    if os.path.isdir(path):
        files = sorted(f for f in os.listdir(path)
                       if f.startswith("assessment-") and f.endswith(".json"))
        if not files:
            raise JudgeError("resume directory %r contains no assessment-*.json files" % path)
        for name in files:
            fpath = os.path.join(path, name)
            try:
                with open(fpath, "r", encoding="utf-8") as handle:
                    data = _loads_no_duplicates(handle.read(), "resume file")
            except OSError as exc:
                raise JudgeError("resume file %r is not readable JSON: %s" % (fpath, exc))
            if not isinstance(data, dict):
                raise JudgeError("resume file %r must contain a JSON object" % fpath)
            _check_resume_envelope(data, model, fpath, panel_sha)
            pid = name[len("assessment-"):-len(".json")]
            if pid not in panel_ids:
                raise JudgeError("resume file %r: unknown panelist %r; whole source rejected"
                                 % (fpath, pid))
            if pid in resumed:
                raise JudgeError("resume directory %r: duplicate entry for panelist %r" % (path, pid))
            resumed[pid] = _validate_resumed_assessment(data, pid, fpath)
        return resumed
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = _loads_no_duplicates(handle.read(), "resume file")
    except OSError as exc:
        raise JudgeError("cannot read resume file %r: %s" % (path, exc))
    if not isinstance(data, dict):
        raise JudgeError("resume file %r must contain a JSON object" % path)
    _check_resume_envelope(data, model, path, panel_sha)
    if "assessment" in data:
        # Single per-panelist record.
        pid = data.get("panelist_id")
        if pid is None:
            entry = data["assessment"]
            pid = entry.get("panelist_id") if isinstance(entry, dict) else None
        if pid not in panel_ids:
            raise JudgeError("resume file %r: unknown panelist %r; whole file rejected"
                             % (path, pid))
        resumed[pid] = _validate_resumed_assessment(data, pid, path)
        return resumed
    assessments = data.get("assessments")
    if not isinstance(assessments, list):
        raise JudgeError("resume file %r must contain an 'assessments' list "
                         "or a single 'assessment' record" % path)
    for entry in assessments:
        if not isinstance(entry, dict):
            raise JudgeError("resume file %r: an assessment entry is not a JSON object" % path)
        id_values = {k: entry[k] for k in ("panelist_id", "panelist", "id", "name")
                     if k in entry}
        pid = next(iter(id_values.values())) if id_values else None
        if pid not in panel_ids:
            raise JudgeError("resume file %r: unknown panelist %r; whole file rejected"
                             % (path, pid))
        if pid in resumed:
            raise JudgeError("resume file %r: duplicate entry for panelist %r; whole file rejected"
                             % (path, pid))
        resumed[pid] = _validate_resumed_assessment(entry, pid, path)
    return resumed


def _atomic_write(path, text):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp, path)


def _per_panelist_record(panelist_id, assessment, model, host, attempts, repair_used, seconds,
                         panel_sha=None, num_ctx=None, prompt_eval_count=None,
                         window_limit=None):
    # assessment is stored WITHOUT its internal key_fixes key: the
    # trace lives at record level (review P1-2), keeping the stored
    # assessment in core-facing shape.
    clean = {k: v for k, v in assessment.items() if k != "key_fixes"}
    return {
        "panelist_id": panelist_id,
        "assessment": clean,
        "model": model,
        "host": host,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "rubric_version": HELPER_RUBRIC_VERSION,
        "temperature": 0,
        "attempts": attempts,
        "repair_used": repair_used,
        "seconds": seconds,
        "panel_sha256": panel_sha,
        "key_fixes": list(assessment.get("key_fixes") or []),
        "num_ctx": num_ctx,
        "prompt_eval_count": prompt_eval_count,
        "window_limit": window_limit,
    }


def _copy_resumed_record(out_dir, pid, resumed_assessment, model, host, panel_sha):
    """Write a resumed per-panelist record into out_dir (review P1-7).

    If a validated record already exists there with the SAME
    assessment, it is kept untouched (attempts, repair_used, seconds
    and key_fixes preserved). If it exists with a DIFFERENT
    assessment, that is a conflict and the run fails -- it is never
    silently overwritten. If no record exists, a new one is written
    (attempts 0 / repair False / seconds 0.0 mark a combined-envelope
    resume, which carries no per-panelist provenance).
    """
    per_path = os.path.join(out_dir, "assessment-%s.json" % pid)
    if os.path.exists(per_path):
        try:
            with open(per_path, "r", encoding="utf-8") as handle:
                existing = _loads_no_duplicates(handle.read(), "saved assessment")
            _check_resume_envelope(existing, model, per_path, panel_sha)
            existing_assessment = _validate_resumed_assessment(existing, pid, per_path)
        except JudgeError:
            raise
        except OSError as exc:
            raise JudgeError("saved assessment %r is not readable: %s" % (per_path, exc))
        _strip = lambda a: {k: v for k, v in a.items() if k != "key_fixes"}
        if _strip(existing_assessment) != _strip(resumed_assessment):
            raise JudgeError(
                "conflict: out-dir record %r differs from the resumed assessment for %r; "
                "refusing to overwrite it" % (per_path, pid))
        return existing_assessment
    _atomic_write(per_path, json.dumps(
        _per_panelist_record(pid, resumed_assessment, model, host, 0, False, 0.0,
                             panel_sha=panel_sha), indent=2) + "\n")
    return resumed_assessment


def _repair_prompt(question, panelist, error, raw):
    base = build_prompt(question, [panelist])
    return base + _REPAIR_SUFFIX.format(
        error=str(error), raw=(raw[:4000] if raw else "(no output)"))


def judge_one_panelist(question, panelist, host, model, timeout, max_attempts, out_dir,
                       panel_sha=None):
    """Judge one panelist with save-immediately semantics.

    Returns (assessment, log_dict). On success the validated
    assessment is written atomically to out_dir/assessment-<id>.json
    before returning, so a kill after this panelist loses nothing.
    A raw-<id>-attemptN.txt file is written per attempt for audit.
    On failure after all attempts, raises JudgeError and writes NO
    assessment file for this panelist (never a hand-filled gap).
    panel_sha, when given, binds the skipped-saved-file path to
    this panel (review P0-2); a saved file for another panel is
    rejected, not reused.
    """
    pid = panelist["id"]
    per_path = os.path.join(out_dir, "assessment-%s.json" % pid)
    if os.path.exists(per_path):
        # Saved file in the out-dir: re-validate strictly. A corrupt
        # saved file is a hard error (negative-control detection), not
        # something to silently overwrite or skip past.
        try:
            with open(per_path, "r", encoding="utf-8") as handle:
                existing = _loads_no_duplicates(handle.read(), "saved assessment")
            _check_resume_envelope(existing, model, per_path, panel_sha)
            assessment = _validate_resumed_assessment(existing, pid, per_path)
        except JudgeError:
            raise
        except OSError as exc:
            raise JudgeError("saved assessment %r is not readable: %s" % (per_path, exc))
        return assessment, {"panelist": pid, "status": "skipped-already-saved",
                            "attempts": existing.get("attempts", 0),
                            "repair_used": existing.get("repair_used", False),
                            "file": per_path}
    errors = []
    raw = None
    repair_used = False
    started = time.time()
    meta = None
    for attempt in range(1, max_attempts + 1):
        if attempt <= 2:
            prompt = build_prompt(question, [panelist])
        else:
            prompt = _repair_prompt(question, panelist,
                                    errors[-1] if errors else "unknown", raw)
            repair_used = True
        try:
            raw, meta = call_ollama(host, model, prompt, timeout)
            with open(os.path.join(out_dir, "raw-%s-attempt%d.txt" % (pid, attempt)),
                      "w", encoding="utf-8") as handle:
                handle.write(raw)
            parsed = parse_and_validate(raw, [pid])
        except JudgeError as exc:
            errors.append(str(exc))
            print("  %s attempt %d FAILED: %s" % (pid, attempt, exc), flush=True)
            if attempt < max_attempts:
                time.sleep(5)
            continue
        assessment = parsed["assessments"][0]
        seconds = round(time.time() - started, 1)
        if assessment.get("key_fixes"):
            print("  %s key_fixes applied: %s" % (pid, ", ".join(assessment["key_fixes"])), flush=True)
        _atomic_write(per_path, json.dumps(
            _per_panelist_record(pid, assessment, model, host, attempt, repair_used, seconds,
                                 panel_sha=panel_sha,
                                 num_ctx=(meta or {}).get("num_ctx"),
                                 prompt_eval_count=(meta or {}).get("prompt_eval_count"),
                                 window_limit=(meta or {}).get("effective_window_limit")),
            indent=2) + "\n")
        print("  %s OK on attempt %d in %.1fs -> %s" % (pid, attempt, seconds, per_path), flush=True)
        return assessment, {"panelist": pid, "status": "ok", "attempts": attempt,
                            "repair_used": repair_used, "seconds": seconds,
                            "file": per_path}
    raise JudgeError("%s FAILED after %d attempts: %s" % (pid, max_attempts, "; ".join(errors)))


def _combined_envelope(model, assessments, panel_sha=None, num_ctx=None, prompt_eval_count=None,
                       window_limit=None):
    """Combined envelope. Assessment entries are stripped of their
    internal key_fixes key (core entry keys are closed); the trace
    is aggregated at envelope level as key_fixes {panelist: [fixes]}
    (review P1-2), and panel_sha256 binds the envelope to the panel
    it judged (review P0-2)."""
    clean = []
    key_fixes = {}
    for entry in assessments:
        clean.append({k: v for k, v in entry.items() if k != "key_fixes"})
        if entry.get("key_fixes"):
            key_fixes[entry["panelist_id"]] = list(entry["key_fixes"])
    env = {
        "assessor": model,
        "assessor_role": "judge-model",
        "assessor_model": model,
        "assessor_provider": "ollama",
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": prompt_sha256(),
        "rubric_version": HELPER_RUBRIC_VERSION,
        "temperature": 0,
        "panel_sha256": panel_sha,
        "assessments": clean,
    }
    if key_fixes:
        env["key_fixes"] = key_fixes
    if num_ctx is not None:
        env["num_ctx"] = num_ctx
    if prompt_eval_count is not None:
        env["prompt_eval_count"] = prompt_eval_count
    if window_limit is not None:
        env["window_limit"] = window_limit
    return env



def parse_args(argv=None):
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Ask a local Ollama model to score panel answers 0-5 on "
            "evidence, efficiency, regulatory, differentiation and "
            "risk_mitigation, and write the validated assessments JSON."
        )
    )
    parser.add_argument("--panel", required=True, help="input panel JSON file")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Ollama model (default: %(default)s)"
    )
    parser.add_argument(
        "--host", default=DEFAULT_HOST, help="Ollama base URL (default: %(default)s)"
    )
    parser.add_argument(
        "--out", default=None, help="output assessments JSON file (required unless --dry-run)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the prompt that would be sent and exit 0 without calling Ollama",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help="Ollama request timeout in seconds (default: %(default)s)",
    )
    parser.add_argument(
        "--resume-from", default=None,
        help="partial assessments file (combined envelope or single "
             "per-panelist record) or directory of assessment-*.json "
             "files to resume from; the whole source is rejected on any "
             "invalid entry",
    )
    parser.add_argument(
        "--per-panelist", action="store_true",
        help="judge one panelist per Ollama call and save each validated "
             "assessment immediately to --out-dir/assessment-<id>.json",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="output directory for --per-panelist (required with it)",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=3,
        help="attempts per panelist in --per-panelist mode (default: %(default)s; "
             "attempt 3+ uses a repair prompt)",
    )
    parser.add_argument(
        "--only", nargs="*", default=None,
        help="in --per-panelist mode, judge only these panelist ids",
    )
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be a positive number of seconds")
    if args.max_attempts < 1:
        parser.error("--max-attempts must be at least 1")
    if args.per_panelist and not args.out_dir:
        parser.error("--out-dir is required with --per-panelist")
    if not args.per_panelist and not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is given")
    return args


def main(argv=None):
    """CLI entry point; returns an exit code."""
    args = parse_args(argv)
    try:
        question, panelists = load_panel(args.panel)
        if args.only and not args.per_panelist:
            raise JudgeError("--only is only valid with --per-panelist")
        target = panelists
        if args.only:
            target = [item for item in panelists if item["id"] in args.only]
            if not target:
                raise JudgeError("--only matched no panelists")
        panel_sha = panel_sha256(question, panelists)
        resumed = {}
        if args.resume_from:
            resumed = load_resume_source(args.resume_from, panelists, args.model, panel_sha)
        resumed_ids = set(resumed)
        # Resume applies only to targeted panelists; resumed entries for
        # non-targeted panelists are ignored (not judged, not written).
        missing = [item for item in target if item["id"] not in resumed_ids]
        if args.dry_run:
            if args.per_panelist:
                for item in target:
                    print("%s: %s" % (item["id"],
                                      "resumed" if item["id"] in resumed_ids else "would-judge"))
            else:
                if not missing:
                    print("(resume source already complete; no request would be made)")
                else:
                    print(build_prompt(question, missing), end="")
            return 0
        if args.per_panelist:
            os.makedirs(args.out_dir, exist_ok=True)
            ordered = {}
            # Resumed assessments are copied into the out-dir as
            # per-panelist records (atomic), so the out-dir is complete
            # and auditable even when nothing was re-judged. An
            # existing out-dir record is kept on a match and raises
            # on a conflict (review P1-7) -- never silently replaced.
            for item in target:
                pid = item["id"]
                if pid in resumed_ids:
                    _copy_resumed_record(args.out_dir, pid, resumed[pid],
                                         args.model, args.host, panel_sha)
                    ordered[pid] = resumed[pid]
                    print("%s: resumed" % pid, flush=True)
            failed = []
            for item in missing:
                print("Panelist %s ..." % item["id"], flush=True)
                try:
                    assessment, _log = judge_one_panelist(
                        question, item, args.host, args.model, args.timeout,
                        args.max_attempts, args.out_dir, panel_sha)
                except JudgeError as exc:
                    failed.append(str(exc))
                    print("  %s FAILED: %s" % (item["id"], exc), flush=True)
                    continue
                ordered[item["id"]] = assessment
            if failed:
                print("error: incomplete: %s; no combined file written"
                      % "; ".join(failed), file=sys.stderr)
                return 1
            if args.out:
                _atomic_write(args.out, json.dumps(
                    _combined_envelope(args.model,
                                       [ordered[item["id"]] for item in target],
                                       panel_sha=panel_sha),
                    indent=2) + "\n")
                print("Assessments: %s" % args.out)
            return 0
        # Whole-panel mode (unchanged behaviour, now resume-aware).
        whole_meta = None
        if not missing:
            parsed_assessments = [resumed[item["id"]] for item in panelists]
        else:
            prompt = build_prompt(question, missing)
            response, whole_meta = call_ollama(args.host, args.model, prompt, args.timeout)
            parsed = parse_and_validate(
                response, [item["id"] for item in missing]
            )
            merged = dict(resumed)
            for entry in parsed["assessments"]:
                merged[entry["panelist_id"]] = entry
            if any(item["id"] not in merged for item in panelists):
                raise JudgeError("merged assessments are incomplete")
            parsed_assessments = [merged[item["id"]] for item in panelists]
        try:
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(_combined_envelope(
                    args.model, parsed_assessments, panel_sha=panel_sha,
                    num_ctx=(whole_meta or {}).get("num_ctx"),
                    prompt_eval_count=(whole_meta or {}).get("prompt_eval_count"),
                    window_limit=(whole_meta or {}).get("effective_window_limit")),
                          handle, indent=2)
                handle.write("\n")
        except OSError as exc:
            raise JudgeError("cannot write output file %r: %s" % (args.out, exc))
    except JudgeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
