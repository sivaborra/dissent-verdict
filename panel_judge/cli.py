#!/usr/bin/env python3
"""CLI for the generic panel-judge library. See DESIGN.md (v1.0).

Commands: judge, validate, sign-off. No network, no subprocess, no
helper invocation: model judges enter only via --assessments files.

Exit codes (DESIGN G14): 0 success (warnings allowed), 1 validation /
judging / output / I-O failure, 2 command-line usage error. Named-error
catalogue for gate checks (P0-11): only `ollama-route-out-of-scope`
(exit 2) and `heuristic-cannot-score-extended-rubric` (exit 1) have
gate-checked code strings; every other failure prints a single line
`error: <descriptive message>` (argparse keeps its own multi-line
format, exit 2; internal errors print `error: internal error` plus a
traceback).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
import tempfile
import traceback
from pathlib import Path

from .core import (
    DEFAULT_MAX_ANSWER_CHARS,
    DEFAULT_MAX_ASSESSMENTS_CHARS,
    DEFAULT_MAX_PANEL_CHARS,
    TOOL_VERSION,
    JudgeError,
    heuristic_scores,
    load_assessments,
    load_panel,
    load_rubric,
    load_verification,
    render_report,
    synthesise,
)

OLLAMA_UNPROVEN = (
    "Full-panel Ollama generation is UNPROVEN: a qwen3:8b run completed "
    "but was correctly rejected for a missing `regulatory` score, and a "
    "qwen3:14b run failed after ~2.3 hours (SSH broken pipe, exit 255, no "
    "output). Do not rely on an Ollama end-to-end route until a full-panel "
    "run completes and passes output validation."
)
OLLAMA_TRAP_MESSAGE = (
    "ollama-route-out-of-scope: the integrated Ollama route is out of scope "
    "in v1.0. Run panel_judge/ollama_judge.py standalone to produce an "
    "assessments file, then run `panel-judge judge --assessments <file>`; "
    "the assessments file is subject to full validation. " + OLLAMA_UNPROVEN
)

# Sign-off notes: banned fact-checking claims (case-insensitive substring
# match, checked only when verification_outcome is not exactly "passed").
# A note saying "not verified" still contains "verified" and is therefore
# rejected conservatively (deliberate over-blocking, stated in DESIGN).
_SIGNOFF_BANNED_SUBSTRINGS = (
    "fact-check", "fact checked", "verified", "checked against sources",
    "verification passed",
)


class UsageError(Exception):
    """A command-line usage error: exit 2, never a judging failure."""


def _check_budgets(args):
    for name, value in (("--max-panel-chars", args.max_panel_chars),
                        ("--max-answer-chars", args.max_answer_chars),
                        ("--max-assessments-chars", args.max_assessments_chars)):
        if value is None or value <= 0:
            raise UsageError(f"{name} must be a positive integer, got {value!r}")


def _stale_warning(out_dir):
    """G14 stale-file warning for a failed run into a pre-existing out-dir.

    The existing files are never touched (no delete, no partial
    overwrite). Returns the warning string, or None if no verdict exists.
    """
    stale = Path(out_dir) / "verdict.json"
    if not stale.exists():
        return None
    created = "unknown, file unreadable"
    try:
        data = json.loads(stale.read_text(encoding="utf-8"))
        created = data.get("state", {}).get("created") or created
    except Exception:
        pass
    return (f"warning: existing verdict.json in {out_dir} is from a previous run "
            f"(created {created}) and was not overwritten by this failed run; "
            f"use a fresh --out-dir or re-run successfully to replace it")


def _write_pair(out_dir, verdict_text, report_text):
    """Best-effort pair atomicity (G14): render fully in memory first
    (done by the caller), write each file to a temp file in the same
    directory and os.replace into place, verdict.json first, then
    report.md. A kill between renames can leave a mismatched pair,
    detectable via state.created; verdict.json wins on disagreement.
    Temp files are removed on any failure before rename."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    temps = []
    try:
        for name, text in (("verdict.json", verdict_text), ("report.md", report_text)):
            fd, tmp = tempfile.mkstemp(prefix=f".{name}.", dir=str(out_dir))
            temps.append(tmp)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(text)
            os.replace(tmp, out_dir / name)
            temps.remove(tmp)
    finally:
        for tmp in temps:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def _add_common(parser):
    parser.add_argument("--panel", required=True, help="input panel JSON file")
    parser.add_argument("--rubric", default=None, help="optional custom rubric JSON file")
    parser.add_argument("--max-panel-chars", type=int, default=DEFAULT_MAX_PANEL_CHARS)
    parser.add_argument("--max-answer-chars", type=int, default=DEFAULT_MAX_ANSWER_CHARS)
    parser.add_argument("--max-assessments-chars", type=int, default=DEFAULT_MAX_ASSESSMENTS_CHARS)
    parser.add_argument("--allow-truncate", action="store_true",
                        help="truncate over-long answers instead of failing (marked in state)")


def cmd_judge(args):
    # P0-4 trap: --ollama-model is a hidden trap argument. It must not
    # invoke any helper, judge, or write anything.
    if getattr(args, "ollama_model", None) is not None:
        raise UsageError(OLLAMA_TRAP_MESSAGE)
    _check_budgets(args)
    if args.assessments and args.heuristic:
        raise UsageError("exactly one judging route: --assessments selects the "
                         "supplied-assessments route and excludes --heuristic")
    panel, raw, warnings, truncated = load_panel(
        args.panel, args.max_panel_chars, args.max_answer_chars, args.allow_truncate)
    rubric, rubric_version, checkpoint_mapping, is_extended = load_rubric(args.rubric)
    verification_record = verification_sha = verification_file = None
    if getattr(args, "verification", None):
        verification_record, verification_sha, verification_file = load_verification(
            args.verification, panel, rubric)
    if args.assessments:
        (per, assessments_sha, assessor, assessor_role, supplied, prompt_version,
         assessor_panelist_id, assessments_chars, envelope) = load_assessments(
            args.assessments, panel, rubric, args.max_assessments_chars,
            checkpoint_mapping=checkpoint_mapping)
        assessments_file = args.assessments
        route, judge_model = "supplied-assessments", None
        per_panelist = {pid: {"scores": e["scores"], "reasons": e["reasons"],
                              "strengths": e["strengths"], "weaknesses": e["weaknesses"]}
                        for pid, e in per.items()}
    else:
        # Default when --assessments is absent is --heuristic (G20).
        if is_extended:
            raise JudgeError(
                "heuristic-cannot-score-extended-rubric: the heuristic scorer knows only "
                "the canonical five criteria and cannot score an extended rubric; "
                "use the supplied-assessments route instead")
        route, judge_model, assessments_sha, supplied, assessor = "heuristic", None, None, None, None
        assessor_role, prompt_version, assessor_panelist_id, assessments_chars = None, None, None, None
        assessments_file, envelope = None, None
        per_panelist = {}
        for p in panel["panelists"]:
            scores, reasons, _counts = heuristic_scores(p["answer"])
            per_panelist[p["id"]] = {"scores": scores, "reasons": reasons,
                                     "strengths": [], "weaknesses": []}
    verdict, warnings, excluded_flagged = synthesise(
        panel, rubric, per_panelist, route, judge_model, raw, args.panel,
        rubric_version, assessments_sha, args.max_panel_chars, truncated,
        supplied_checkpoints=supplied, assessor=assessor, warnings=warnings,
        assessor_role=assessor_role, prompt_version=prompt_version,
        max_answer_chars=args.max_answer_chars,
        max_assessments_chars=args.max_assessments_chars,
        rubric_file=args.rubric, assessments_file=assessments_file,
        assessor_panelist_id=assessor_panelist_id,
        assessments_chars=assessments_chars,
        checkpoint_mapping=checkpoint_mapping, envelope=envelope,
        verification_record=verification_record,
        verification_sha256=verification_sha, verification_file=verification_file)
    verdict_text = json.dumps(verdict, indent=2) + "\n"
    report_text = render_report(verdict, route, judge_model, warnings,
                                heuristic_used=(route == "heuristic"),
                                excluded_flagged=excluded_flagged)
    _write_pair(args.out_dir, verdict_text, report_text)
    v = verdict["verdict"]
    _overall = f"{v['overall_score']:.1f}" if v['overall_score'] is not None else "null"
    print(f"Recommended strategy: {v['recommended_strategy_label'] or 'no-consensus'} "
          f"(consensus: {v['consensus']}, overall score: {_overall})")
    print(f"Verdict: {Path(args.out_dir) / 'verdict.json'}")
    print(f"Report: {Path(args.out_dir) / 'report.md'}")
    for w in warnings:
        print(f"Warning: {w}")
    return 0


def cmd_validate(args):
    # P0-9: validate takes no --heuristic and no --ollama-model; passing
    # either is a usage error. --assessments is a validation target, not
    # route selection. --verification is accepted and validated (P1-6).
    if getattr(args, "ollama_model", None) is not None:
        raise UsageError(OLLAMA_TRAP_MESSAGE)
    if getattr(args, "heuristic", False):
        raise UsageError("validate takes no --heuristic flag: it validates inputs, it does not judge")
    _check_budgets(args)
    panel, _raw, warnings, trunc = load_panel(
        args.panel, args.max_panel_chars, args.max_answer_chars, args.allow_truncate)
    rubric, _ver, _mapping, _ext = load_rubric(args.rubric)
    if args.assessments:
        load_assessments(args.assessments, panel, rubric, args.max_assessments_chars,
                         checkpoint_mapping=_mapping)
    if getattr(args, "verification", None):
        load_verification(args.verification, panel, rubric)
    print(f"Valid: panel {panel['panel_id']} with {len(panel['panelists'])} panelists.")
    if trunc.get("truncated"):
        for pid in trunc["truncated_panelist_ids"]:
            print(f"Warning: with --allow-truncate, answer for panelist {pid!r} would be "
                  f"truncated from {trunc['original_answer_chars'][pid]} to "
                  f"{trunc['truncated_answer_chars'][pid]} characters")
    for w in warnings:
        print(f"Warning: {w}")
    return 0


def _sign_off_completeness(verdict):
    """Completeness checklist (P1-6a) for sign-off. Raises JudgeError."""
    if not isinstance(verdict, dict):
        raise JudgeError("sign-off: verdict file must contain a JSON object")
    required_top = {"panel_id", "question", "verdict", "rubric", "checkpoints",
                    "checkpoint_disagreements", "per_panelist", "dissent",
                    "disclaimer", "state", "verification"}
    unknown = sorted(set(verdict) - required_top)
    if unknown:
        raise JudgeError(f"sign-off: verdict file has unknown top-level key(s): {', '.join(unknown)}")
    missing_top = sorted(required_top - set(verdict))
    if missing_top:
        raise JudgeError(f"sign-off: verdict file is missing top-level key(s): {', '.join(missing_top)}")
    from .core import DISCLAIMER
    if verdict["disclaimer"] != DISCLAIMER:
        raise JudgeError("sign-off: verdict disclaimer is not verbatim")
    st = verdict["state"]
    if not isinstance(st, dict):
        raise JudgeError("sign-off: verdict state record is missing or not an object")
    for key in ("panel_sha256", "panel_file", "rubric_version", "rubric_criteria",
                "checkpoint_mapping", "verification_outcome"):
        if key not in st:
            raise JudgeError(f"sign-off: state record is missing {key!r}")
    route = st.get("judging_route")
    if route == "heuristic" and not st.get("heuristic_mapping_version"):
        raise JudgeError("sign-off: heuristic run requires a heuristic_mapping_version in state")
    if route == "supplied-assessments":
        if st.get("assessor_role") == "judge-model":
            for key in ("assessor_model", "assessor_provider", "prompt_version"):
                if not st.get(key):
                    raise JudgeError(f"sign-off: supplied model run requires {key!r} in state")
            # review P1-8 at sign-off: a judge-model verdict with a
            # null prompt SHA or a missing temperature must not be
            # signed off, even if it slipped past an older loader.
            _sha = st.get("prompt_sha256")
            if not isinstance(_sha, str) or len(_sha) != 64 or any(c not in "0123456789abcdef" for c in _sha):
                raise JudgeError("sign-off: supplied model run requires a 64-hex prompt_sha256 in state")
            if st.get("temperature") is None:
                raise JudgeError("sign-off: supplied model run requires a temperature in state")
        elif st.get("assessor_role") == "human":
            if st.get("assessor_model") is not None:
                raise JudgeError("sign-off: human supplied run must not carry an assessor_model in state")
        else:
            raise JudgeError("sign-off: supplied run requires an assessor_role in state")
    outcome = st.get("verification_outcome")
    top_ver = verdict.get("verification")
    if top_ver is None:
        if outcome != "not-run":
            raise JudgeError("sign-off: state verification_outcome disagrees with the top-level verification record")
        if st.get("verification_sha256") is not None or st.get("verification_file") is not None:
            raise JudgeError("sign-off: not-run verification must have null verification_sha256/file in state")
    else:
        if not isinstance(top_ver, dict) or top_ver.get("verification_outcome") != outcome:
            raise JudgeError("sign-off: state verification_outcome disagrees with the top-level verification record")
    return st


def _first_diff(a, b, path=""):
    """First differing path between two JSON-shaped values, or None.

    Used by sign-off re-derivation: the error names the exact field
    that was edited, instead of a bare 'does not match'. bool and
    int are distinct types here (type() check), so a True/1 swap is
    a difference, not an equality accident."""
    if type(a) is not type(b):
        return (path or "/", a, b)
    if isinstance(a, dict):
        for key in sorted(set(a) | set(b)):
            if key not in a or key not in b:
                return (path + "/" + str(key),
                        a.get(key, "<missing>"), b.get(key, "<missing>"))
            diff = _first_diff(a[key], b[key], path + "/" + str(key))
            if diff is not None:
                return diff
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return (path, f"list of length {len(a)}", f"list of length {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff = _first_diff(x, y, f"{path}[{i}]")
            if diff is not None:
                return diff
        return None
    if a != b:
        return (path or "/", a, b)
    return None


def _rederive_verdict(verdict, st):
    """Re-run the judging pipeline from the verdict's recorded inputs
    and return the re-derived verdict (review re-review P1-11 full).

    The previous sign-off only re-hashed the panel and assessments
    files, which cannot catch a hand-edited verdict.json: its inputs
    are unchanged, so the hashes match and the approval then binds
    to the edited bytes. Re-derivation catches exactly that case:
    the re-derived verdict is compared to the file, ignoring only
    state.created (a timestamp). Any other difference -- a changed
    score, reason, consensus, or state field -- is a refusal.
    Consequence, stated: sign-off now requires a verdict
    reproducible by THIS library version from its recorded inputs.
    A verdict produced by an older version (missing newer state
    fields) will not re-derive identically and must be re-judged.
    Missing/unreadable inputs are JudgeErrors, never internal
    errors, and an unknown judging_route cannot be re-derived.
    """
    try:
        panel_path = st.get("panel_file")
        if not panel_path:
            raise JudgeError("sign-off: verdict state has no panel_file to re-derive from")
        max_panel = st.get("max_panel_chars") or DEFAULT_MAX_PANEL_CHARS
        max_answer = st.get("max_answer_chars") or DEFAULT_MAX_ANSWER_CHARS
        max_assess = st.get("max_assessments_chars") or DEFAULT_MAX_ASSESSMENTS_CHARS
        panel, raw, _w, truncated = load_panel(
            panel_path, max_panel, max_answer, bool(st.get("truncated")))
        rubric, rubric_version, mapping, is_extended = load_rubric(st.get("rubric_file"))
        if rubric_version != st.get("rubric_version"):
            raise JudgeError(
                "sign-off: rubric at the recorded path does not match state.rubric_version; "
                "the verdict's inputs have changed since judging -- refusing to sign off")
        verification_record = verification_sha = verification_file = None
        if st.get("verification_file") is not None:
            verification_record, verification_sha, verification_file = load_verification(
                st["verification_file"], panel, rubric)
            if verification_sha != st.get("verification_sha256"):
                raise JudgeError(
                    "sign-off: verification file at the recorded path does not match "
                    "state.verification_sha256 -- refusing to sign off")
        elif (st.get("verification_outcome") or "not-run") != "not-run":
            raise JudgeError(
                "sign-off: verdict claims verification but state has no verification_file; "
                "cannot re-derive -- refusing to sign off")
        route = st.get("judging_route")
        if route == "heuristic":
            if is_extended:
                raise JudgeError(
                    "sign-off: heuristic route cannot score an extended rubric; cannot re-derive")
            per_panelist = {}
            for p in panel["panelists"]:
                scores, reasons, _counts = heuristic_scores(p["answer"])
                per_panelist[p["id"]] = {"scores": scores, "reasons": reasons,
                                         "strengths": [], "weaknesses": []}
            assessments_sha = supplied = assessor = None
            assessor_role = prompt_version = assessor_panelist_id = None
            assessments_chars = envelope = assessments_file = None
        elif route == "supplied-assessments":
            (per, assessments_sha, assessor, assessor_role, supplied, prompt_version,
             assessor_panelist_id, assessments_chars, envelope) = load_assessments(
                st.get("assessments_file"), panel, rubric, max_assess,
                checkpoint_mapping=mapping)
            assessments_file = st.get("assessments_file")
            per_panelist = {pid: {"scores": e["scores"], "reasons": e["reasons"],
                                  "strengths": e["strengths"], "weaknesses": e["weaknesses"]}
                            for pid, e in per.items()}
            if assessments_sha != st.get("assessments_sha256"):
                raise JudgeError(
                    "sign-off: re-derived assessments hash does not match "
                    "state.assessments_sha256 -- refusing to sign off")
        else:
            raise JudgeError(
                f"sign-off: unknown judging_route {route!r}; cannot re-derive -- refusing to sign off")
        verdict2, _w2, _ex2 = synthesise(
            panel, rubric, per_panelist, route, st.get("judge_model"), raw, panel_path,
            rubric_version, assessments_sha, max_panel, truncated,
            supplied_checkpoints=supplied, assessor=assessor, warnings=[],
            assessor_role=assessor_role, prompt_version=prompt_version,
            max_answer_chars=max_answer, max_assessments_chars=max_assess,
            rubric_file=st.get("rubric_file"), assessments_file=assessments_file,
            ollama_endpoint=st.get("ollama_endpoint"),
            assessor_panelist_id=assessor_panelist_id,
            assessments_chars=assessments_chars, checkpoint_mapping=mapping,
            envelope=envelope, verification_record=verification_record,
            verification_sha256=verification_sha, verification_file=verification_file)
        return verdict2
    except JudgeError:
        raise
    except OSError as exc:
        raise JudgeError(f"sign-off: cannot re-read a recorded input: {exc}")


def cmd_sign_off(args):
    if not isinstance(args.approver, str) or not args.approver.strip():
        raise UsageError("--approver must be a non-empty human name")
    verdict_path = Path(args.verdict)
    try:
        raw = verdict_path.read_bytes()
    except OSError as exc:
        raise JudgeError(f"sign-off: cannot read verdict file {args.verdict!r}: {exc}")
    try:
        verdict = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise JudgeError(f"sign-off: verdict file is not valid JSON: {exc}")
    st = _sign_off_completeness(verdict)
    # review P1-11: re-verify the verdict's inputs on disk before
    # signing off. A hand-edited verdict.json (changed scores) is
    # caught only if its recorded input hashes no longer match the
    # files at their recorded paths; a missing input file is a
    # refusal, not a silent pass. This is the minimum re-verification
    # the review asked for (SHA check, not a full re-derivation,
    # which would also need the original rubric and route flags).
    _panel_path = st.get("panel_file")
    if not _panel_path:
        raise JudgeError("sign-off: verdict state has no panel_file to re-verify")
    try:
        _panel_bytes = Path(_panel_path).read_bytes()
    except OSError as exc:
        raise JudgeError(f"sign-off: cannot re-read panel file {_panel_path!r}: {exc}")
    if hashlib.sha256(_panel_bytes).hexdigest() != st.get("panel_sha256"):
        raise JudgeError(
            "sign-off: panel file at the recorded path does not match state.panel_sha256; "
            "the verdict's inputs have changed since judging -- refusing to sign off")
    _assess_path = st.get("assessments_file")
    _assess_sha = st.get("assessments_sha256")
    if _assess_path is not None or _assess_sha is not None:
        if not _assess_path or not _assess_sha:
            raise JudgeError("sign-off: verdict state has an incomplete assessments file record")
        try:
            _assess_bytes = Path(_assess_path).read_bytes()
        except OSError as exc:
            raise JudgeError(f"sign-off: cannot re-read assessments file {_assess_path!r}: {exc}")
        if hashlib.sha256(_assess_bytes).hexdigest() != _assess_sha:
            raise JudgeError(
                "sign-off: assessments file at the recorded path does not match "
                "state.assessments_sha256; the verdict's inputs have changed since judging "
                "-- refusing to sign off")
    # review re-review P1-11 full: re-derive the verdict from its
    # recorded inputs and compare, ignoring only state.created.
    # SHA checks alone cannot catch a hand-edited verdict.json --
    # its inputs are unchanged -- but re-derivation can: the edited
    # score/reason/consensus will differ from the re-derived one.
    _rederived = _rederive_verdict(verdict, st)
    _orig = json.loads(json.dumps(verdict))
    _re = json.loads(json.dumps(_rederived))
    _orig["state"]["created"] = None
    _re["state"]["created"] = None
    _diff = _first_diff(_orig, _re)
    if _diff is not None:
        raise JudgeError(
            f"sign-off: re-derived verdict does not match the verdict file at {_diff[0]} "
            f"(file: {_diff[1]!r}, re-derived: {_diff[2]!r}); the verdict has been edited "
            "since judging -- refusing to sign off")
    outcome = st.get("verification_outcome") or "not-run"
    # review re-review P1-10 full: a fact-checking note is refused
    # unless verification passed AND coverage is complete. A
    # single-claim "passed" must never unlock panel-wide
    # fact-check wording. Coverage comes from the top-level
    # verification record, falling back to state (new verdicts carry
    # it there); a missing coverage record is treated as unknown,
    # i.e. the note is refused -- fail closed.
    _cov = None
    if isinstance(verdict.get("verification"), dict):
        _cov = verdict["verification"].get("verification_coverage")
    if _cov is None:
        _cov = st.get("verification_coverage")
    coverage_complete = _cov.get("coverage_complete") if isinstance(_cov, dict) else None
    note = args.note
    if note is not None:
        lowered = note.lower()
        if any(bad in lowered for bad in _SIGNOFF_BANNED_SUBSTRINGS):
            if not (outcome == "passed" and coverage_complete is True):
                raise JudgeError(
                    "sign-off: note claims fact-checking but verification is not 'passed' with "
                    "complete coverage (outcome: %s, coverage_complete: %s); a fact-checking "
                    "note is refused unless verification passed for all panelists"
                    % (outcome, coverage_complete))
    out_path = verdict_path.parent / "sign-off.json"
    if out_path.exists() and not args.force:
        raise JudgeError(
            f"sign-off: {out_path} already exists; re-sign with --force (the old record "
            "is yours to preserve via version control or a copy)")
    record = {
        "verdict_sha256": hashlib.sha256(raw).hexdigest(),
        "panel_id": verdict["panel_id"],
        "panel_sha256": st["panel_sha256"],
        "rubric_version": st["rubric_version"],
        "rubric_criteria": st.get("rubric_criteria"),
        "assessments_sha256": st.get("assessments_sha256"),
        "assessor_model": st.get("assessor_model"),
        "prompt_version": st.get("prompt_version"),
        "verification_outcome": outcome,
        "verification_coverage_complete": coverage_complete,
        "verdict_rederived": True,
        "approver": args.approver,
        "decision": args.decision,
        "note": note,
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "tool_version": TOOL_VERSION,
    }
    # Forgery limit (P1-6b), stated in DESIGN: sign-off.json carries no
    # signature; approval trust comes from custody of the directory, not
    # from the file itself. Non-repudiation must be added outside the library.
    _write_pair_signoff(out_path, json.dumps(record, indent=2) + "\n")
    print(f"Sign-off recorded: {out_path} (decision: {args.decision}, "
          f"verdict_sha256: {record['verdict_sha256']})")
    return 0


def _write_pair_signoff(out_path, text):
    fd, tmp = tempfile.mkstemp(prefix=".sign-off.", dir=str(out_path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def build_parser():
    parser = argparse.ArgumentParser(prog="panel-judge", description="Generic, file-based AI-panel judge.")
    sub = parser.add_subparsers(dest="command", required=True)
    pj = sub.add_parser("judge", help="judge a panel file and write verdict.json + report.md")
    _add_common(pj)
    pj.add_argument("--assessments", default=None,
                    help="supplied assessments JSON: selects the supplied-assessments route; excludes --heuristic")
    pj.add_argument("--heuristic", action="store_true",
                    help="deterministic heuristic scorer: selects the heuristic route (default when --assessments is absent); excludes --assessments")
    pj.add_argument("--ollama-model", default=None, help=argparse.SUPPRESS)
    pj.add_argument("--verification", default=None, help="optional verification JSON file (annotation only; never changes scores)")
    pj.add_argument("--out-dir", required=True, help="output directory")
    pj.set_defaults(func=cmd_judge)
    pv = sub.add_parser("validate", help="validate inputs without judging")
    _add_common(pv)
    pv.add_argument("--assessments", default=None, help="optional supplied assessments JSON to validate")
    pv.add_argument("--verification", default=None, help="optional verification JSON file to validate")
    pv.add_argument("--heuristic", action="store_true", help=argparse.SUPPRESS)
    pv.add_argument("--ollama-model", default=None, help=argparse.SUPPRESS)
    pv.set_defaults(func=cmd_validate)
    ps = sub.add_parser("sign-off", help="record a human approval sidecar (sign-off.json) next to a verdict")
    ps.add_argument("--verdict", required=True, help="verdict.json file to approve")
    ps.add_argument("--approver", required=True, help="human name of the approver (recorded verbatim; identity is not verified)")
    ps.add_argument("--decision", required=True, choices=["approved", "not-approved"],
                    help="approval decision; conditions go in --note, and a decision with material conditions should be not-approved")
    ps.add_argument("--note", default=None, help="one-line condition or reason (fact-checking claims refused unless verification passed with complete coverage)")
    ps.add_argument("--force", action="store_true", help="overwrite an existing sign-off.json")
    ps.set_defaults(func=cmd_sign_off)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except UsageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except JudgeError as exc:
        out_dir = getattr(args, "out_dir", None)
        if out_dir:
            warning = _stale_warning(out_dir)
            if warning:
                print(warning, file=sys.stderr)
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except Exception:
        print("error: internal error", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
