"""Tests for the review 5.5 re-review fixes (2026-10-31): the
PARTIAL and NEW P1 items only -- P1-6 full + N-2 (rendered-prompt
hash + rubric binding), N-1 (fail-closed window guard), P1-10 full
(coverage-enforced sign-off), P1-11 full (sign-off re-derivation).
No network, no model: Ollama calls are stubbed where needed.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from panel_judge import core, ollama_judge as oj
from panel_judge.core import JudgeError, canonical_panel_sha256, load_assessments, load_panel, load_verification, render_report, synthesise

CANON = ["evidence_strength", "capital_efficiency", "regulatory_runway",
         "differentiation", "risk_mitigation"]


def _write(tmp, name, obj):
    p = os.path.join(tmp, name)
    with open(p, "w") as h:
        json.dump(obj, h)
    return p


def _panel(panelists, panel_id="t"):
    return {"panel_id": panel_id, "question": "Which strategy is best?",
            "panelists": panelists}


def _panelist(pid, answer, label=None, model="some-model"):
    p = {"id": pid, "model": model, "source": "model", "answer": answer}
    if label:
        p["strategy_label"] = label
    return p


def _scores(v):
    return {c: v for c in CANON}


def _reasons():
    return {c: "reason for %s" % c for c in CANON}


def _assessments_obj(scores_map, panel, extra=None):
    entries = [{"panelist_id": pid, "scores": dict(s), "reasons": _reasons()}
               for pid, s in scores_map.items()]
    obj = {"assessor": "m", "assessor_role": "judge-model", "assessor_model": "m",
           "assessor_provider": "ollama", "prompt_version": "x",
           "prompt_sha256": "a" * 64, "temperature": 0,
           "panel_sha256": canonical_panel_sha256(panel),
           "assessments": entries}
    if extra:
        obj.update(extra)
    return obj


def _judge_heuristic(tmp, panel_obj, verification_obj=None):
    pp = _write(tmp, "panel.json", panel_obj)
    cmd = [sys.executable, "-m", "panel_judge", "judge",
           "--panel", pp, "--heuristic", "--out-dir", os.path.join(tmp, "out")]
    if verification_obj is not None:
        vp = _write(tmp, "ver.json", verification_obj)
        cmd += ["--verification", vp]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return os.path.join(tmp, "out", "verdict.json")


def _signoff(verdict_path, note=None, force=False):
    cmd = [sys.executable, "-m", "panel_judge", "sign-off",
           "--verdict", verdict_path, "--approver", "Tester",
           "--decision", "approved"]
    if note is not None:
        cmd += ["--note", note]
    if force:
        cmd += ["--force"]
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


class TestP1_6FullAndN2(unittest.TestCase):
    def test_prompt_sha_is_rendered_dummy_prompt_plus_repair(self):
        # The hash definition itself: rendered dummy prompt + repair
        # suffix. If build_prompt's framing changes, this recomputation
        # uses the same code path, so the definition test is paired
        # with the mutation tests below (template/guidance/repair
        # changes must all move the hash).
        material = (oj.build_prompt(oj._DUMMY_QUESTION, oj._DUMMY_PANELISTS)
                    + "\n--- repair suffix ---\n" + oj._REPAIR_SUFFIX)
        self.assertEqual(oj.prompt_sha256(),
                         hashlib.sha256(material.encode("utf-8")).hexdigest())

    def test_prompt_sha_moves_with_template_guidance_and_repair(self):
        before = oj.prompt_sha256()
        old_template, old_guidance, old_repair = oj.PROMPT_TEMPLATE, dict(oj.CRITERION_GUIDANCE), oj._REPAIR_SUFFIX
        try:
            oj.PROMPT_TEMPLATE = old_template + " EXTRA"
            self.assertNotEqual(oj.prompt_sha256(), before)
            oj.PROMPT_TEMPLATE = old_template
            oj.CRITERION_GUIDANCE["evidence"] = "changed guidance"
            self.assertNotEqual(oj.prompt_sha256(), before)
            oj.CRITERION_GUIDANCE.clear()
            oj.CRITERION_GUIDANCE.update(old_guidance)
            oj._REPAIR_SUFFIX = old_repair + " EXTRA"
            self.assertNotEqual(oj.prompt_sha256(), before)
        finally:
            oj.PROMPT_TEMPLATE = old_template
            oj.CRITERION_GUIDANCE.clear()
            oj.CRITERION_GUIDANCE.update(old_guidance)
            oj._REPAIR_SUFFIX = old_repair
        self.assertEqual(oj.prompt_sha256(), before)
        self.assertEqual(oj.PROMPT_VERSION, "ollama-judge-v4")

    def test_helper_envelope_binds_rubric_and_core_records_it(self):
        entry = {"panelist_id": "a",
                 "scores": {c: 3 for c in oj.CRITERIA},
                 "reasons": {c: "r" for c in oj.CRITERIA}}
        parsed = oj.parse_and_validate(json.dumps({"assessments": [entry]}), ["a"])
        env = oj._combined_envelope("qwen3:8b", parsed["assessments"], panel_sha="d" * 64)
        self.assertEqual(env["rubric_version"], oj.HELPER_RUBRIC_VERSION)
        self.assertNotEqual(env["rubric_version"], "default-v1")
        # Core accepts the new envelope keys and records the binding.
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A.", "a"), _panelist("b", "Plan B.", "b")])
            obj = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel,
                                   extra={"rubric_version": oj.HELPER_RUBRIC_VERSION,
                                          "window_limit": oj.EFFECTIVE_WINDOW_LIMIT})
            pp = _write(tmp, "panel.json", panel)
            ap = _write(tmp, "assess.json", obj)
            panel_loaded, _r, _w, _t = load_panel(pp, 250000, 50000, False)
            rubric, _v, mapping, _e = core.load_rubric(None)
            per, _sha, _a, _role, _cp, _pv, _pid, _ch, envelope = load_assessments(
                ap, panel_loaded, rubric, 250000, checkpoint_mapping=mapping)
            self.assertEqual(envelope["rubric_version"], oj.HELPER_RUBRIC_VERSION)
            per_panelist = {pid: {"scores": e["scores"], "reasons": e["reasons"],
                                  "strengths": [], "weaknesses": []} for pid, e in per.items()}
            raw = json.dumps(panel).encode()
            verdict, warnings, _ex = synthesise(
                panel_loaded, rubric, per_panelist, "supplied-assessments", None, raw,
                "panel.json", "default-v1", "a" * 64, 250000, {"truncated": False},
                envelope=envelope)
            self.assertEqual(verdict["state"]["assessor_rubric_version"],
                             oj.HELPER_RUBRIC_VERSION)
            # Distinct anchor texts: the mismatch must be warned, not silent.
            self.assertTrue(any("ollama-builtin-v1" in w and "default-v1" in w
                                for w in warnings))
            # Bad rubric/window types are rejected, not recorded.
            obj2 = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel,
                                    extra={"rubric_version": ""})
            ap2 = _write(tmp, "assess2.json", obj2)
            with self.assertRaises(JudgeError):
                load_assessments(ap2, panel_loaded, rubric, 250000, checkpoint_mapping=mapping)
            obj3 = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel,
                                    extra={"window_limit": 0})
            ap3 = _write(tmp, "assess3.json", obj3)
            with self.assertRaises(JudgeError):
                load_assessments(ap3, panel_loaded, rubric, 250000, checkpoint_mapping=mapping)

    def test_helper_resume_rejects_wrong_rubric_version(self):
        rec = {"panelist_id": "alpha",
               "assessment": {"panelist_id": "alpha",
                              "scores": {c: 3 for c in oj.CRITERIA},
                              "reasons": {c: "r" for c in oj.CRITERIA}},
               "model": "qwen3:8b", "host": "http://x",
               "prompt_version": oj.PROMPT_VERSION, "prompt_sha256": oj.prompt_sha256(),
               "rubric_version": "some-other-rubric", "temperature": 0,
               "panel_sha256": "d" * 64, "attempts": 1, "repair_used": False, "seconds": 1.0}
        with self.assertRaises(oj.JudgeError):
            oj._check_resume_envelope(rec, "qwen3:8b", "x", "d" * 64)


class TestN1WindowGuard(unittest.TestCase):
    def _call(self, prompt_eval_count):
        class Resp:
            def __init__(self, body):
                self.body = body
            def read(self):
                return json.dumps(self.body).encode()
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        body = {"response": "{}", "model": "qwen3:8b"}
        if prompt_eval_count != "MISSING":
            body["prompt_eval_count"] = prompt_eval_count
        with mock.patch.object(urllib.request, "urlopen", return_value=Resp(body)):
            return oj.call_ollama("http://x", "qwen3:8b", "prompt", 10)

    def test_missing_prompt_eval_count_fails_closed(self):
        with self.assertRaises(oj.JudgeError) as ctx:
            self._call("MISSING")
        self.assertIn("did not report prompt_eval_count", str(ctx.exception))

    def test_bool_prompt_eval_count_fails_closed(self):
        with self.assertRaises(oj.JudgeError):
            self._call(True)

    def test_at_and_above_effective_limit_fails(self):
        self.assertEqual(oj.EFFECTIVE_WINDOW_LIMIT, oj.NUM_CTX - oj.OUTPUT_MARGIN)
        for bad in (oj.EFFECTIVE_WINDOW_LIMIT, oj.NUM_CTX, oj.NUM_CTX + 100):
            with self.assertRaises(oj.JudgeError, msg=str(bad)):
                self._call(bad)

    def test_below_effective_limit_passes(self):
        text, meta = self._call(oj.EFFECTIVE_WINDOW_LIMIT - 1)
        self.assertEqual(text, "{}")
        self.assertEqual(meta["effective_window_limit"], oj.EFFECTIVE_WINDOW_LIMIT)


class TestP1_10Full(unittest.TestCase):
    def _verification(self, checked):
        checks = []
        for pid in checked:
            checks.append({"panelist_id": pid, "claim_excerpt": "Plan %s." % pid.upper(),
                           "claim_is_verbatim": False, "criterion_bearing": None,
                           "source_ref": "src", "status": "verified",
                           "evidence_note": "ok"})
        return {"verifier": "t", "verified_at": "2026-10-31T10:00:00Z", "checks": checks}

    def test_fact_check_note_refused_on_partial_coverage_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A.", "a"), _panelist("b", "Plan B.", "b")])
            vpath = _judge_heuristic(tmp, panel, self._verification(["a"]))
            verdict = json.load(open(vpath))
            self.assertEqual(verdict["verification"]["verification_outcome"], "passed")
            self.assertFalse(verdict["verification"]["verification_coverage"]["coverage_complete"])
            self.assertFalse(verdict["state"]["verification_coverage"]["coverage_complete"])
            # Fact-check wording refused even though outcome is "passed".
            r = _signoff(vpath, note="fact-checked against sources")
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("complete coverage", r.stderr)
            # Plain sign-off still succeeds, and records the gap.
            r = _signoff(vpath)
            self.assertEqual(r.returncode, 0, r.stderr)
            record = json.load(open(os.path.join(os.path.dirname(vpath), "sign-off.json")))
            self.assertFalse(record["verification_coverage_complete"])
            self.assertTrue(record["verdict_rederived"])
            # The report itself must not read as panel-wide.
            report = open(os.path.join(os.path.dirname(vpath), "report.md")).read()
            self.assertIn("NOT panel-wide", report)

    def test_fact_check_note_allowed_on_complete_coverage_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A.", "a"), _panelist("b", "Plan B.", "b")])
            vpath = _judge_heuristic(tmp, panel, self._verification(["a", "b"]))
            r = _signoff(vpath, note="fact-checked against sources")
            self.assertEqual(r.returncode, 0, r.stderr)
            record = json.load(open(os.path.join(os.path.dirname(vpath), "sign-off.json")))
            self.assertTrue(record["verification_coverage_complete"])


class TestP1_11Full(unittest.TestCase):
    # Rich vs bare answers so the heuristic verdict has rubric rows
    # (a tied no-consensus verdict shows no rubric table).
    _RICH = ("Plan A. Sold 100 units, bought-in components, deposits, "
             "named legal route, unique heritage craft, risks answered "
             "with pivot and deposits.")

    def test_signoff_refuses_edited_verdict_scores(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", self._RICH, "a"), _panelist("b", "Plan B.", "b")])
            vpath = _judge_heuristic(tmp, panel)
            # Genuine verdict re-derives and signs off.
            r = _signoff(vpath)
            self.assertEqual(r.returncode, 0, r.stderr)
            # Edit a rubric score in the verdict file (inputs unchanged).
            verdict = json.load(open(vpath))
            self.assertTrue(verdict["rubric"], "heuristic verdict should have rubric rows")
            verdict["rubric"][0]["score"] = 5 if verdict["rubric"][0]["score"] != 5 else 0
            with open(vpath, "w") as h:
                json.dump(verdict, h, indent=2)
            r = _signoff(vpath, force=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("re-derived", r.stderr)

    def test_signoff_refuses_edited_verdict_consensus(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", self._RICH, "a"), _panelist("b", "Plan B.", "b")])
            vpath = _judge_heuristic(tmp, panel)
            verdict = json.load(open(vpath))
            verdict["verdict"]["consensus"] = "majority" if verdict["verdict"]["consensus"] != "majority" else "no-consensus"
            with open(vpath, "w") as h:
                json.dump(verdict, h, indent=2)
            r = _signoff(vpath, force=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("re-derived", r.stderr)


if __name__ == "__main__":
    unittest.main()
