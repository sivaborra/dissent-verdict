"""Tests for the review 5.5 review fixes (2026-10-31): P0-1, P0-2 and
all 13 P1 items, in review-file order. Each test is a negative
control where the review gives a failing input, or a positive check
that the new trace/binding is present. No network, no model.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from panel_judge import core, ollama_judge as oj
from panel_judge.core import JudgeError, canonical_panel_sha256, load_assessments, load_panel, load_verification, synthesise

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


def _assessments_obj(panelists_scores, role="judge-model", extra=None, panel=None):
    entries = []
    for pid, scores in panelists_scores.items():
        entries.append({"panelist_id": pid, "scores": dict(scores), "reasons": _reasons()})
    obj = {"assessor": "m", "assessor_role": role, "assessor_model": "m",
           "assessor_provider": "ollama", "prompt_version": "x",
           "prompt_sha256": "a" * 64, "temperature": 0, "assessments": entries}
    if role == "human":
        obj = {"assessor": "h", "assessor_role": "human", "assessments": entries}
    if panel is not None:
        obj["panel_sha256"] = canonical_panel_sha256(panel)
    if extra:
        obj.update(extra)
    return obj


def _load(panel_obj, assess_obj, tmp):
    pp = _write(tmp, "panel.json", panel_obj)
    ap = _write(tmp, "assess.json", assess_obj)
    panel, _raw, _w, _t = load_panel(pp, 250000, 50000, False)
    rubric, _v, mapping, _e = core.load_rubric(None)
    return load_assessments(ap, panel, rubric, 250000, checkpoint_mapping=mapping)


class TestP0(unittest.TestCase):
    def test_p0_1_alias_collision_scores_reasons_checkpoints(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A.", "a"), _panelist("b", "Plan B.", "b")])
            # Scores: alias + canonical for the same criterion.
            bad = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel=panel)
            bad["assessments"][0]["scores"]["evidence"] = 1  # + evidence_strength 4
            with self.assertRaises(JudgeError):
                _load(panel, bad, tmp)
            # Reasons: alias + canonical.
            bad = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel=panel)
            bad["assessments"][0]["reasons"]["evidence"] = "alias reason"
            with self.assertRaises(JudgeError):
                _load(panel, bad, tmp)
            # Supplied checkpoint norm_scores: alias + canonical.
            bad = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel=panel)
            bad["checkpoints"] = [{"checkpoint_id": "evidence",
                                   "scores": {"evidence_strength": 4, "evidence": 1},
                                   "notes": "n"}]
            with self.assertRaises(JudgeError):
                _load(panel, bad, tmp)

    def test_p0_2_core_requires_panel_sha_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A.", "a"), _panelist("b", "Plan B.", "b")])
            # Missing panel_sha for judge-model -> fail.
            with self.assertRaises(JudgeError):
                _load(panel, _assessments_obj({"a": _scores(4), "b": _scores(3)}), tmp)
            # Wrong panel_sha -> fail.
            with self.assertRaises(JudgeError):
                _load(panel, _assessments_obj({"a": _scores(4), "b": _scores(3)},
                                               extra={"panel_sha256": "b" * 64}), tmp)
            # Correct panel_sha -> pass.
            _load(panel, _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel=panel), tmp)
            # Human role without panel_sha still passes (optional there).
            _load(panel, _assessments_obj({"a": _scores(4), "b": _scores(3)}, role="human"), tmp)

    def test_p0_2_helper_resume_rejects_panel_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = {"question": "Q?", "panelists": [{"id": "alpha", "answer": "A."},
                                                     {"id": "beta", "answer": "B."}]}
            pp = _write(tmp, "panel.json", panel)
            rec = {"panelist_id": "alpha",
                   "assessment": {"panelist_id": "alpha", "scores": {c: 3 for c in oj.CRITERIA},
                                  "reasons": {c: "r" for c in oj.CRITERIA}},
                   "model": "qwen3:8b", "host": "http://x", "prompt_version": oj.PROMPT_VERSION,
                   "prompt_sha256": oj.prompt_sha256(), "rubric_version": oj.HELPER_RUBRIC_VERSION, "temperature": 0,
                   "panel_sha256": "c" * 64, "attempts": 1, "repair_used": False, "seconds": 1.0}
            rp = _write(tmp, "resume.json", rec)
            # Panel mismatch: the helper's own panel_sha differs from the record's.
            self.assertEqual(oj.main(["--panel", pp, "--model", "qwen3:8b",
                                      "--resume-from", rp, "--out", os.path.join(tmp, "o.json")]), 1)

    def test_p0_2_shipped_combined_files_validate_against_petrol_fixture(self):
        # Re-stamped 2026-10-31 (panel_sha256 added, scores untouched).
        for name in ("out/ollama-v1-1-8b-shipped/combined.json",
                     "out/ollama-v1-1-14b-resumed-shipped/combined.json"):
            r = subprocess.run([sys.executable, "-m", "panel_judge", "validate",
                                "--panel", "fixtures/petrol-car-panel.json",
                                "--assessments", name],
                               cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, (name, r.stderr))


class TestP1(unittest.TestCase):
    def test_p1_1_helper_rejects_unknown_criterion_keys(self):
        entry = {"panelist_id": "a",
                 "scores": {c: 3 for c in oj.CRITERIA},
                 "reasons": {c: "r" for c in oj.CRITERIA}}
        entry["scores"]["regulatry"] = entry["scores"].pop("regulatory")
        with self.assertRaises(oj.JudgeError):
            oj.parse_and_validate(json.dumps({"assessments": [entry]}), ["a"])
        entry2 = {"panelist_id": "a",
                  "scores": {c: 3 for c in oj.CRITERIA},
                  "reasons": {c: "r" for c in oj.CRITERIA}}
        entry2["scores"]["foo"] = 5
        with self.assertRaises(oj.JudgeError):
            oj.parse_and_validate(json.dumps({"assessments": [entry2]}), ["a"])

    def test_p1_2_key_fixes_recorded_in_record_and_envelope(self):
        entry = {"panelist_id": "alpha",
                 "scores": {c: 4 for c in oj.CRITERIA},
                 "reasons": {c: "r" for c in oj.CRITERIA}}
        entry["scores"]["reg,ulatory"] = entry["scores"].pop("regulatory")
        parsed = oj.parse_and_validate(json.dumps({"assessments": [entry]}), ["alpha"])
        got = parsed["assessments"][0]
        self.assertIn("scores:reg,ulatory->regulatory", got["key_fixes"])
        env = oj._combined_envelope("qwen3:8b", parsed["assessments"],
                                    panel_sha="d" * 64)
        self.assertEqual(env["key_fixes"]["alpha"], got["key_fixes"])
        self.assertNotIn("key_fixes", env["assessments"][0])  # core entry keys stay closed
        rec = oj._per_panelist_record("alpha", got, "qwen3:8b", "http://x", 1, False, 1.0,
                                      panel_sha="d" * 64)
        self.assertEqual(rec["key_fixes"], got["key_fixes"])
        self.assertNotIn("key_fixes", rec["assessment"])

    def test_p1_3_strict_scores_reject_clamp_inputs(self):
        for bad in (9, -2, "3", 4.5, True, 2.5, None, float("inf")):
            with self.assertRaises(oj.JudgeError, msg=repr(bad)):
                oj._clamp_score(bad, "a", "evidence")
        self.assertEqual(oj._clamp_score(0, "a", "evidence"), 0)
        self.assertEqual(oj._clamp_score(5, "a", "evidence"), 5)

    def test_p1_4_skeleton_uses_null_not_zero(self):
        prompt = oj.build_prompt("Q?", [{"id": "a", "answer": "A"}])
        self.assertIn('"evidence": null', prompt)
        self.assertNotIn('"evidence": 0', prompt)
        self.assertIn("ollama-judge-v4", prompt)

    def test_p1_5_duplicate_json_keys_rejected(self):
        with self.assertRaises(oj.JudgeError):
            oj.parse_and_validate('{"assessments": [{"panelist_id": "a", "scores": '
                                  '{"evidence": 5, "evidence": 0}}]}', ["a"])
        with tempfile.TemporaryDirectory() as tmp:
            pp = os.path.join(tmp, "panel.json")
            with open(pp, "w") as h:
                h.write('{"panel_id": "t", "question": "Q?", "panel_id": "dup", "panelists": []}')
            with self.assertRaises(JudgeError):
                load_panel(pp, 250000, 50000, False)

    def test_p1_6_prompt_sha_covers_guidance_and_version(self):
        before = oj.prompt_sha256()
        old = dict(oj.CRITERION_GUIDANCE)
        try:
            oj.CRITERION_GUIDANCE["evidence"] = "changed guidance"
            self.assertNotEqual(oj.prompt_sha256(), before)
        finally:
            oj.CRITERION_GUIDANCE.clear()
            oj.CRITERION_GUIDANCE.update(old)
        self.assertEqual(oj.prompt_sha256(), before)
        self.assertEqual(oj.PROMPT_VERSION, "ollama-judge-v4")

    def test_p1_7_resume_conflict_raises_and_sha_temperature_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = {"question": "Q?", "panelists": [{"id": "alpha", "answer": "A."},
                                                     {"id": "beta", "answer": "B."}]}
            pp = _write(tmp, "panel.json", panel)
            psha = oj.panel_sha256(panel["question"], panel["panelists"])
            out_dir = os.path.join(tmp, "per")
            os.makedirs(out_dir)
            existing = {"panelist_id": "alpha",
                        "assessment": {"panelist_id": "alpha",
                                       "scores": {c: 3 for c in oj.CRITERIA},
                                       "reasons": {c: "r" for c in oj.CRITERIA}},
                        "model": "qwen3:8b", "host": "http://x",
                        "prompt_version": oj.PROMPT_VERSION, "prompt_sha256": oj.prompt_sha256(),
                        "rubric_version": oj.HELPER_RUBRIC_VERSION, "temperature": 0, "panel_sha256": psha,
                        "attempts": 2, "repair_used": True, "seconds": 9.0}
            with open(os.path.join(out_dir, "assessment-alpha.json"), "w") as h:
                json.dump(existing, h)
            different = {"panelist_id": "alpha",
                         "scores": {c: 5 for c in oj.CRITERIA},
                         "reasons": {c: "r" for c in oj.CRITERIA}}
            with self.assertRaises(oj.JudgeError):
                oj._copy_resumed_record(out_dir, "alpha", different, "qwen3:8b", "http://x", psha)
            # Same assessment: kept untouched, attempts preserved.
            same = {"panelist_id": "alpha",
                    "scores": {c: 3 for c in oj.CRITERIA},
                    "reasons": {c: "r" for c in oj.CRITERIA}}
            oj._copy_resumed_record(out_dir, "alpha", same, "qwen3:8b", "http://x", psha)
            kept = json.load(open(os.path.join(out_dir, "assessment-alpha.json")))
            self.assertEqual(kept["attempts"], 2)
            self.assertTrue(kept["repair_used"])
            # Missing prompt_sha256 in a resume source -> rejected.
            bad = dict(existing)
            del bad["prompt_sha256"]
            with self.assertRaises(oj.JudgeError):
                oj._check_resume_envelope(bad, "qwen3:8b", "x", psha)
            bad = dict(existing)
            bad["temperature"] = 1
            with self.assertRaises(oj.JudgeError):
                oj._check_resume_envelope(bad, "qwen3:8b", "x", psha)

    def test_p1_8_judge_model_provenance_required_and_signoff_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A.", "a"), _panelist("b", "Plan B.", "b")])
            obj = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel=panel)
            del obj["prompt_sha256"]
            with self.assertRaises(JudgeError):
                _load(panel, obj, tmp)
            obj = _assessments_obj({"a": _scores(4), "b": _scores(3)}, panel=panel)
            del obj["temperature"]
            with self.assertRaises(JudgeError):
                _load(panel, obj, tmp)
            # Sign-off refuses a judge-model verdict with null prompt SHA.
            from panel_judge.cli import _sign_off_completeness
            verdict = {"panel_id": "t", "question": "Q?", "verdict": {}, "rubric": [],
                       "checkpoints": [], "checkpoint_disagreements": [], "per_panelist": [],
                       "dissent": [], "disclaimer": core.DISCLAIMER,
                       "state": {"panel_sha256": "a" * 64, "panel_file": "p",
                                 "rubric_version": "default-v1", "rubric_criteria": CANON,
                                 "checkpoint_mapping": {}, "verification_outcome": "not-run",
                                 "judging_route": "supplied-assessments",
                                 "assessor_role": "judge-model", "assessor_model": "m",
                                 "assessor_provider": "ollama", "prompt_version": "v",
                                 "prompt_sha256": None, "temperature": 0},
                       "verification": None}
            with self.assertRaises(JudgeError):
                _sign_off_completeness(verdict)

    def test_p1_9_assessor_model_match_warns(self):
        panel = _panel([_panelist("a", "Plan A.", "a", model="model-x"),
                        _panelist("b", "Plan B.", "b", model="other")])
        rubric, _v, mapping, _e = core.load_rubric(None)
        per = {"a": {"scores": _scores(4), "reasons": _reasons(), "strengths": [], "weaknesses": []},
               "b": {"scores": _scores(3), "reasons": _reasons(), "strengths": [], "weaknesses": []}}
        raw = json.dumps(panel).encode()
        verdict, warnings, _ex = synthesise(
            panel, rubric, per, "supplied-assessments", None, raw, "panel.json",
            "default-v1", "a" * 64, 250000, {"truncated": False},
            envelope={"assessor_model": "model-x", "assessor_provider": "ollama",
                      "prompt_sha256": "a" * 64, "temperature": 0})
        self.assertTrue(any("model-x" in w for w in warnings))
        self.assertIn("a", verdict["state"]["assessor_model_matches_panelist_model"])
        self.assertFalse(verdict["state"]["self_judging"])  # id-based finding unchanged

    def test_p1_10_verification_coverage_labelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A mentions 100 units.", "a"),
                            _panelist("b", "Plan B.", "b")])
            pp = _write(tmp, "panel.json", panel)
            rubric, _v, _m, _e = core.load_rubric(None)
            panel_loaded, _r, _w, _t = load_panel(pp, 250000, 50000, False)
            ver = {"verifier": "t", "verified_at": "2026-10-31T10:00:00Z",
                   "checks": [{"panelist_id": "a", "claim_excerpt": "100 units",
                               "claim_is_verbatim": True, "criterion_bearing": None,
                               "source_ref": "src", "status": "verified",
                               "evidence_note": "ok"}]}
            vp = _write(tmp, "ver.json", ver)
            record, _sha, _path = load_verification(vp, panel_loaded, rubric)
            self.assertEqual(record["verification_outcome"], "passed")
            self.assertEqual(record["verification_coverage"]["panelists_unchecked"], ["b"])
            self.assertFalse(record["verification_coverage"]["coverage_complete"])

    def test_p1_11_signoff_refuses_changed_panel(self):
        with tempfile.TemporaryDirectory() as tmp:
            panel = _panel([_panelist("a", "Plan A.", "a"), _panelist("b", "Plan B.", "b")])
            pp = _write(tmp, "panel.json", panel)
            out = os.path.join(tmp, "out")
            r = subprocess.run([sys.executable, "-m", "panel_judge", "judge",
                                "--panel", pp, "--heuristic", "--out-dir", out],
                               cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            vpath = os.path.join(out, "verdict.json")
            r = subprocess.run([sys.executable, "-m", "panel_judge", "sign-off",
                                "--verdict", vpath, "--approver", "Tester",
                                "--decision", "approved"],
                               cwd=str(ROOT), capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            # Edit the panel after judging: sign-off must now refuse.
            panel["panelists"][0]["answer"] = "Plan A changed."
            _write(tmp, "panel.json", panel)
            r = subprocess.run([sys.executable, "-m", "panel_judge", "sign-off",
                                "--verdict", vpath, "--approver", "Tester",
                                "--decision", "approved", "--force"],
                               cwd=str(ROOT), capture_output=True, text=True)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("panel_sha256", r.stderr)

    def test_p1_12_dissent_names_largest_gap_not_weakest(self):
        panel = _panel([_panelist("a", "Plan A.", "rec"), _panelist("b", "Plan B.", "dissent")])
        rubric, _v, mapping, _e = core.load_rubric(None)
        # Recommended group scores [2,5,5,5,5]; dissenter [2,3,3,3,3]:
        # its weakest (evidence, 2) TIES the recommendation, so the old
        # text "rejected on evidence (2 vs 2)" was false. New text must
        # name a criterion with a real gap.
        per = {"a": {"scores": {"evidence_strength": 2, "capital_efficiency": 5,
                                 "regulatory_runway": 5, "differentiation": 5,
                                 "risk_mitigation": 5},
                      "reasons": _reasons(), "strengths": [], "weaknesses": []},
               "b": {"scores": {"evidence_strength": 2, "capital_efficiency": 3,
                                 "regulatory_runway": 3, "differentiation": 3,
                                 "risk_mitigation": 3},
                      "reasons": _reasons(), "strengths": [], "weaknesses": []}}
        raw = json.dumps(panel).encode()
        verdict, _w, _ex = synthesise(
            panel, rubric, per, "supplied-assessments", None, raw, "panel.json",
            "default-v1", "a" * 64, 250000, {"truncated": False})
        self.assertEqual(verdict["verdict"]["consensus"], "plurality")
        diss = [d for d in verdict["dissent"] if d["panelist_id"] == "b"]
        self.assertTrue(diss)
        self.assertNotIn("rejected on evidence_strength", diss[0]["verdict_response"])
        self.assertIn("capital_efficiency", diss[0]["verdict_response"])
        self.assertIn("3 vs 5", diss[0]["verdict_response"])

    def test_p1_13_six_criterion_half_margin_is_not_no_consensus(self):
        # Six equal-weight criteria: true 0.5 margin must not read as
        # 0.4999999999999998 and force no-consensus.
        rubric = [{"id": "c%d" % i, "name": "C%d" % i, "description": "d", "weight": 1.0 / 6}
                  for i in range(6)]
        mapping = {c["id"]: "evidence" for c in rubric}
        panel = _panel([_panelist("a", "Plan A.", "rec"), _panelist("b", "Plan B.", "dis")])
        per = {"a": {"scores": {c["id"]: 5 for c in rubric},
                      "reasons": {c["id"]: "r" for c in rubric}, "strengths": [], "weaknesses": []},
               "b": {"scores": {rubric[0]["id"]: 5, rubric[1]["id"]: 5, rubric[2]["id"]: 5,
                                 rubric[3]["id"]: 4, rubric[4]["id"]: 4, rubric[5]["id"]: 4},
                      "reasons": {c["id"]: "r" for c in rubric}, "strengths": [], "weaknesses": []}}
        raw = json.dumps(panel).encode()
        verdict, _w, _ex = synthesise(
            panel, rubric, per, "supplied-assessments", None, raw, "panel.json",
            "custom:abc", "a" * 64, 250000, {"truncated": False},
            checkpoint_mapping=mapping)
        self.assertNotEqual(verdict["verdict"]["consensus"], "no-consensus")


if __name__ == "__main__":
    unittest.main()
