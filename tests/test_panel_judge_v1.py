#!/usr/bin/env python3
"""v1.0 tests: P0-fixed behaviours, verification hook, extended rubrics,
sign-off, exit codes, envelope validation, OpenRouter/Ollama helpers.

Run: python3 -m unittest discover -s tests -v
"""
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "petrol-car-panel.json"
CRITERIA = ["evidence_strength", "capital_efficiency", "regulatory_runway",
            "differentiation", "risk_mitigation"]
DISCLAIMER = ("Scores rate the answers against the stated criteria. "
              "They are not a guarantee of success, market outcome, or regulatory outcome.")


def cli(*args, cwd=ROOT):
    return subprocess.run([sys.executable, "-m", "panel_judge"] + list(args),
                          cwd=cwd, capture_output=True, text=True)


def write_json(path, obj):
    p = Path(path)
    p.write_text(json.dumps(obj))
    return p


def panelist(pid, answer, label=None, is_dissent=False, kind=None):
    p = {"id": pid, "model": pid, "source": "model", "answer": answer}
    if label is not None:
        p["strategy_label"] = label
    if is_dissent:
        p["is_dissent"] = True
    if kind is not None:
        p["answer_kind"] = kind
    return p


def panel_with(panelists, panel_id="synthetic"):
    return {"panel_id": panel_id, "question": "Which strategy is best?",
            "panelists": panelists}


def human_assessments(scores_by_pid, extra=None):
    assessments = []
    for pid, scores in scores_by_pid.items():
        assessments.append({
            "panelist_id": pid, "scores": dict(scores),
            "reasons": {cid: f"reason for {cid}" for cid in CRITERIA},
            "strengths": [], "weaknesses": []})
    obj = {"assessor": "test human", "assessor_role": "human", "assessments": assessments}
    if extra:
        obj.update(extra)
    return obj


def model_assessments(scores_by_pid, **envelope):
    obj = human_assessments(scores_by_pid)
    obj["assessor"] = "test-model"
    obj["assessor_role"] = "judge-model"
    obj["assessor_model"] = "test-model"
    obj["assessor_provider"] = "openrouter"
    obj["prompt_version"] = "openrouter-judge-v2"
    obj.update(envelope)
    return obj


def all_scores(value, criteria=CRITERIA):
    return {cid: value for cid in criteria}


class TestEnvelope(unittest.TestCase):
    def setUp(self):
        panel = panel_with([panelist("a", "Plan A.", label="a"),
                            panelist("b", "Plan B.", label="b")], panel_id="env")
        self.panel = write_json("/tmp/v1-env-panel.json", panel)

    def judge_with(self, assessments_obj, out="/tmp/v1-env-out"):
        a = write_json("/tmp/v1-env-assessments.json", assessments_obj)
        return cli("judge", "--panel", str(self.panel), "--assessments", str(a),
                   "--out-dir", out)

    def test_model_envelope_required_fields(self):
        base = {"a": all_scores(4), "b": all_scores(3)}
        # Missing assessor_model for judge-model -> fail.
        obj = model_assessments(base)
        del obj["assessor_model"]
        r = self.judge_with(obj)
        self.assertNotEqual(r.returncode, 0)
        # Model ID with human role -> fail.
        obj = human_assessments(base)
        obj["assessor_model"] = "test-model"
        r = self.judge_with(obj)
        self.assertNotEqual(r.returncode, 0)
        # Unknown top-level key (incl. smuggled verification) -> fail.
        obj = model_assessments(base)
        obj["verification"] = {"checks": []}
        r = self.judge_with(obj)
        self.assertNotEqual(r.returncode, 0)
        # Bad prompt_sha256 -> fail; good one passes.
        obj = model_assessments(base, prompt_sha256="not-a-hash")
        r = self.judge_with(obj)
        self.assertNotEqual(r.returncode, 0)
        from panel_judge.core import canonical_panel_sha256
        _env_panel = {"question": "Which strategy is best?",
                      "panelists": [{"id": "a", "answer": "Plan A."},
                                    {"id": "b", "answer": "Plan B."}]}
        obj = model_assessments(base, prompt_sha256="a" * 64, temperature=0,
                                panel_sha256=canonical_panel_sha256(_env_panel),
                                usage={"prompt_tokens": 10, "completion_tokens": 5,
                                       "total_tokens": 15, "cost_amount": None,
                                       "cost_currency": None, "pricing_source": None})
        r = self.judge_with(obj, out="/tmp/v1-env-out-ok")
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(Path("/tmp/v1-env-out-ok/verdict.json").read_text())
        self.assertEqual(v["state"]["assessor_model"], "test-model")
        self.assertEqual(v["state"]["prompt_sha256"], "a" * 64)
        self.assertEqual(v["state"]["temperature"], 0)
        # Cost without currency -> fail; currency without amount -> fail.
        obj = model_assessments(base, usage={"cost_amount": 1.0, "cost_currency": None,
                                             "pricing_source": "provider-response"})
        r = self.judge_with(obj)
        self.assertNotEqual(r.returncode, 0)
        obj = model_assessments(base, usage={"cost_amount": None, "cost_currency": "USD",
                                             "pricing_source": None})
        r = self.judge_with(obj)
        self.assertNotEqual(r.returncode, 0)
        # Cost without pricing_source -> fail.
        obj = model_assessments(base, usage={"cost_amount": 1.0, "cost_currency": "USD",
                                             "pricing_source": None})
        r = self.judge_with(obj)
        self.assertNotEqual(r.returncode, 0)

    def test_identifier_aliases_and_differing_keys(self):
        panel = panel_with([panelist("a", "Plan A.", label="a"),
                            panelist("b", "Plan B.", label="b")], panel_id="alias")
        p = write_json("/tmp/v1-alias-panel.json", panel)
        # panelist alias accepted.
        obj = {"assessor": "h", "assessor_role": "human", "assessments": [
            {"panelist": "a", "scores": all_scores(4),
             "reasons": {cid: "r" for cid in CRITERIA}},
            {"panelist": "b", "scores": all_scores(3),
             "reasons": {cid: "r" for cid in CRITERIA}}]}
        a = write_json("/tmp/v1-alias-assessments.json", obj)
        r = cli("judge", "--panel", str(p), "--assessments", str(a), "--out-dir", "/tmp/v1-alias-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        # Differing duplicate identifier keys -> fail.
        obj["assessments"][0] = {"panelist_id": "a", "panelist": "b",
                                 "scores": all_scores(4), "reasons": {cid: "r" for cid in CRITERIA}}
        a = write_json("/tmp/v1-alias-assessments.json", obj)
        r = cli("judge", "--panel", str(p), "--assessments", str(a), "--out-dir", "/tmp/v1-alias-out2")
        self.assertNotEqual(r.returncode, 0)
        # Unknown key inside an entry -> fail.
        obj = human_assessments({"a": all_scores(4), "b": all_scores(3)})
        obj["assessments"][0]["surprise"] = 1
        a = write_json("/tmp/v1-alias-assessments.json", obj)
        r = cli("judge", "--panel", str(p), "--assessments", str(a), "--out-dir", "/tmp/v1-alias-out3")
        self.assertNotEqual(r.returncode, 0)
        # assessor_panelist_id naming a non-panelist -> fail.
        obj = human_assessments({"a": all_scores(4), "b": all_scores(3)},
                                extra={"assessor_panelist_id": "zzz"})
        a = write_json("/tmp/v1-alias-assessments.json", obj)
        r = cli("judge", "--panel", str(p), "--assessments", str(a), "--out-dir", "/tmp/v1-alias-out4")
        self.assertNotEqual(r.returncode, 0)

    def test_self_judging_disclosure_uses_panelist_id_when_assessor_absent(self):
        panel = panel_with([panelist("a", "Plan A.", label="a"),
                            panelist("b", "Plan B.", label="b")], panel_id="sj")
        p = write_json("/tmp/v1-sj-panel.json", panel)
        obj = {"assessor_role": "human",
               "assessor_panelist_id": "a",
               "assessments": [
                   {"panelist_id": "a", "scores": all_scores(4),
                    "reasons": {cid: "r" for cid in CRITERIA}},
                   {"panelist_id": "b", "scores": all_scores(3),
                    "reasons": {cid: "r" for cid in CRITERIA}}]}
        a = write_json("/tmp/v1-sj-assessments.json", obj)
        out = Path("/tmp/v1-sj-out")
        r = cli("judge", "--panel", str(p), "--assessments", str(a), "--out-dir", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("The assessor a also sat on this panel as a.", (out / "report.md").read_text())


class TestExitCodesAndAtomicity(unittest.TestCase):
    def test_route_conflict_exit_2(self):
        r = cli("judge", "--panel", str(FIXTURE), "--heuristic",
                "--assessments", "x.json", "--out-dir", "/tmp/v1-conflict-out")
        self.assertEqual(r.returncode, 2, r.stderr)

    def test_ollama_trap_exit_2_named_error_writes_nothing(self):
        out = Path("/tmp/v1-ollama-trap-out")
        r = cli("judge", "--panel", str(FIXTURE), "--ollama-model", "qwen3:8b",
                "--out-dir", str(out))
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("ollama-route-out-of-scope", r.stderr)
        self.assertIn("ollama_judge.py", r.stderr)
        self.assertIn("UNPROVEN", r.stderr)
        self.assertFalse((out / "verdict.json").exists())

    def test_non_positive_budget_exit_2(self):
        r = cli("judge", "--panel", str(FIXTURE), "--heuristic",
                "--max-panel-chars", "0", "--out-dir", "/tmp/v1-budget-out")
        self.assertEqual(r.returncode, 2, r.stderr)

    def test_validate_forbids_route_flags_allows_assessments_and_verification(self):
        r = cli("validate", "--panel", str(FIXTURE), "--heuristic")
        self.assertEqual(r.returncode, 2, r.stderr)
        r = cli("validate", "--panel", str(FIXTURE), "--ollama-model", "qwen3:8b")
        self.assertEqual(r.returncode, 2, r.stderr)
        r = cli("validate", "--panel", str(FIXTURE),
                "--assessments", "out/petrol-car-openrouter/assessments-v1.json")
        # Path is relative to ROOT cwd; assessments-v1 exists in repo.
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_failed_run_leaves_no_new_files_and_warns_on_stale(self):
        out = Path("/tmp/v1-stale-out")
        r = cli("judge", "--panel", str(FIXTURE), "--heuristic", "--out-dir", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        created = json.loads((out / "verdict.json").read_text())["state"]["created"]
        bad = panel_with([panelist("a", "Plan A.", label="a"),
                          panelist("b", "Plan B.", label="b")], panel_id="bad")
        bad["verdict"] = {}
        p = write_json("/tmp/v1-stale-panel.json", bad)
        r = cli("judge", "--panel", str(p), "--heuristic", "--out-dir", str(out))
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("warning: existing verdict.json", r.stderr)
        self.assertIn(created, r.stderr)
        self.assertIn("was not overwritten", r.stderr)
        # Stale verdict untouched.
        self.assertEqual(json.loads((out / "verdict.json").read_text())["state"]["created"], created)

    def test_failed_run_creates_no_out_dir(self):
        bad = panel_with([panelist("a", "Plan A.", label="a"),
                          panelist("b", "Plan B.", label="b")], panel_id="bad2")
        bad["dissent"] = []
        p = write_json("/tmp/v1-nodir-panel.json", bad)
        out = Path("/tmp/v1-nodir-out-xyz")
        r = cli("judge", "--panel", str(p), "--heuristic", "--out-dir", str(out))
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(out.exists())

    def test_panel_total_fit_after_truncation(self):
        # Question text alone exceeds the panel budget; answer truncation
        # cannot fix it -> fail even with --allow-truncate.
        panel = {"panel_id": "bigq", "question": "Q" * 500,
                 "panelists": [panelist("a", "Plan A.", label="a"),
                               panelist("b", "Plan B.", label="b")]}
        p = write_json("/tmp/v1-bigq-panel.json", panel)
        r = cli("judge", "--panel", str(p), "--heuristic", "--allow-truncate",
                "--max-panel-chars", "100", "--out-dir", "/tmp/v1-bigq-out")
        self.assertNotEqual(r.returncode, 0)
        # Answer truncation that DOES fit succeeds and records lengths.
        panel = {"panel_id": "trunc", "question": "Q?",
                 "panelists": [panelist("a", "X" * 100, label="a"),
                               panelist("b", "Plan B.", label="b")]}
        p = write_json("/tmp/v1-trunc-panel.json", panel)
        r = cli("judge", "--panel", str(p), "--heuristic", "--allow-truncate",
                "--max-answer-chars", "50", "--out-dir", "/tmp/v1-trunc-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(Path("/tmp/v1-trunc-out/verdict.json").read_text())
        self.assertTrue(v["state"]["truncated"])
        self.assertEqual(v["state"]["truncated_panelist_ids"], ["a"])
        self.assertEqual(v["state"]["original_answer_chars"]["a"], 100)
        self.assertEqual(v["state"]["truncated_answer_chars"]["a"], 50)

    def test_case_variant_label_warning_and_stripped_labels(self):
        panel = panel_with([panelist("a", "Plan A.", label=" Boutique-Sports "),
                            panelist("b", "Plan B.", label="boutique-sports")],
                           panel_id="casevar")
        p = write_json("/tmp/v1-casevar-panel.json", panel)
        a = write_json("/tmp/v1-casevar-assessments.json",
                       human_assessments({"a": all_scores(4), "b": all_scores(4)}))
        out = Path("/tmp/v1-casevar-out")
        r = cli("judge", "--panel", str(p), "--assessments", str(a), "--out-dir", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("differ only by case", r.stdout)
        self.assertIn("differ only by case", (out / "report.md").read_text())
        v = json.loads((out / "verdict.json").read_text())
        # G19: verdict labels are the stripped form.
        self.assertEqual(v["per_panelist"][0]["strategy_label"], "Boutique-Sports")

    def test_nul_rejected(self):
        panel = panel_with([panelist("a", "Plan A\u0000.", label="a"),
                            panelist("b", "Plan B.", label="b")], panel_id="nul")
        p = write_json("/tmp/v1-nul-panel.json", panel)
        r = cli("judge", "--panel", str(p), "--heuristic", "--out-dir", "/tmp/v1-nul-out")
        self.assertNotEqual(r.returncode, 0)


class TestVerification(unittest.TestCase):
    def setUp(self):
        panel = panel_with([panelist("a", "Plan A mentions 100 units sold in the United States.", label="a"),
                            panelist("b", "Plan B mentions 50 units sold in Europe.", label="b")],
                           panel_id="verif")
        self.panel = write_json("/tmp/v1-verif-panel.json", panel)
        self.assess = write_json("/tmp/v1-verif-assessments.json",
                                 human_assessments({"a": all_scores(4), "b": all_scores(3)}))

    def judge(self, verification=None, out="/tmp/v1-verif-out"):
        args = ["judge", "--panel", str(self.panel), "--assessments", str(self.assess),
                "--out-dir", out]
        if verification:
            args += ["--verification", str(verification)]
        return cli(*args)

    def test_scores_identical_with_and_without_verification(self):
        ver = write_json("/tmp/v1-verif-file.json", {
            "verifier": "test reviewer", "verified_at": "2026-10-30T10:00:00Z",
            "checks": [{"panelist_id": "a", "claim_excerpt": "100 units sold",
                        "claim_is_verbatim": True, "criterion_bearing": "evidence_strength",
                        "source_ref": "test source doc", "status": "verified",
                        "evidence_note": "Source confirms 100 units."}]})
        r1 = self.judge(out="/tmp/v1-verif-out-nv")
        r2 = self.judge(verification=ver, out="/tmp/v1-verif-out-v")
        self.assertEqual(r1.returncode, 0, r1.stderr)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        v1 = json.loads(Path("/tmp/v1-verif-out-nv/verdict.json").read_text())
        v2 = json.loads(Path("/tmp/v1-verif-out-v/verdict.json").read_text())
        for key in ("rubric", "checkpoints", "per_panelist", "dissent"):
            self.assertEqual(v1[key], v2[key], key)
        self.assertEqual(v1["verdict"], v2["verdict"])
        self.assertEqual(v2["verification"]["verification_outcome"], "passed")
        self.assertTrue(v2["verification"]["scores_unchanged_by_verification"])
        self.assertEqual(v2["state"]["verification_outcome"], "passed")
        self.assertIsNone(v1["verification"])
        self.assertEqual(v1["state"]["verification_outcome"], "not-run")
        report = Path("/tmp/v1-verif-out-v/report.md").read_text()
        self.assertIn("## Verification (not scoring)", report)
        self.assertIn("test reviewer", report)
        self.assertNotIn("## Verification", Path("/tmp/v1-verif-out-nv/report.md").read_text())

    def test_outcome_taxonomy_and_contradicted_never_lowers_scores(self):
        def ver_with(statuses):
            checks = []
            for i, status in enumerate(statuses):
                checks.append({"panelist_id": "a", "claim_excerpt": "100 units sold",
                               "claim_is_verbatim": True, "criterion_bearing": None,
                               "source_ref": "src" if status in ("verified", "contradicted") else None,
                               "status": status, "evidence_note": f"note {i}"})
            return write_json("/tmp/v1-verif-tax.json", {"verifier": "t", "checks": checks})
        cases = [(["verified", "verified"], "passed"),
                 (["verified", "contradicted"], "contradicted"),
                 (["verified", "unverifiable"], "mixed"),
                 (["unverifiable"], "mixed"),
                 (["not_checked"], "mixed")]
        for i, (statuses, expected) in enumerate(cases):
            out = f"/tmp/v1-verif-tax-out-{i}"
            r = self.judge(verification=ver_with(statuses), out=out)
            self.assertEqual(r.returncode, 0, r.stderr)
            v = json.loads(Path(out + "/verdict.json").read_text())
            self.assertEqual(v["verification"]["verification_outcome"], expected)
            # Contradiction never lowers scores: identical rubric to no-verification run.
            base = json.loads(Path("/tmp/v1-verif-out-nv/verdict.json").read_text()) \
                if Path("/tmp/v1-verif-out-nv/verdict.json").exists() else None
            if base is None:
                self.judge(out="/tmp/v1-verif-out-nv")
                base = json.loads(Path("/tmp/v1-verif-out-nv/verdict.json").read_text())
            self.assertEqual(v["rubric"], base["rubric"])

    def test_verification_validation_failures(self):
        # Empty checks.
        ver = write_json("/tmp/v1-verif-bad.json", {"verifier": "t", "checks": []})
        r = self.judge(verification=ver, out="/tmp/v1-verif-bad-out")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(Path("/tmp/v1-verif-bad-out/verdict.json").exists())
        # Verbatim excerpt not a substring.
        ver = write_json("/tmp/v1-verif-bad.json", {"verifier": "t", "checks": [
            {"panelist_id": "a", "claim_excerpt": "words never written",
             "claim_is_verbatim": True, "criterion_bearing": None,
             "source_ref": "src", "status": "verified", "evidence_note": "n"}]})
        r = self.judge(verification=ver, out="/tmp/v1-verif-bad-out")
        self.assertNotEqual(r.returncode, 0)
        # Verified check without a source.
        ver = write_json("/tmp/v1-verif-bad.json", {"verifier": "t", "checks": [
            {"panelist_id": "a", "claim_excerpt": "100 units sold",
             "claim_is_verbatim": True, "criterion_bearing": None,
             "source_ref": None, "status": "verified", "evidence_note": "n"}]})
        r = self.judge(verification=ver, out="/tmp/v1-verif-bad-out")
        self.assertNotEqual(r.returncode, 0)
        # Unknown per-check key; non-boolean verbatim flag; bad panelist.
        ver = write_json("/tmp/v1-verif-bad.json", {"verifier": "t", "checks": [
            {"panelist_id": "a", "claim_excerpt": "100 units sold",
             "claim_is_verbatim": True, "criterion_bearing": None,
             "source_ref": "src", "status": "verified", "evidence_note": "n",
             "surprise": 1}]})
        r = self.judge(verification=ver, out="/tmp/v1-verif-bad-out")
        self.assertNotEqual(r.returncode, 0)
        # validate also accepts (and checks) --verification.
        ver = write_json("/tmp/v1-verif-good.json", {"verifier": "t", "checks": [
            {"panelist_id": "a", "claim_excerpt": "100 units sold",
             "claim_is_verbatim": True, "criterion_bearing": None,
             "source_ref": "src", "status": "verified", "evidence_note": "n"}]})
        r = cli("validate", "--panel", str(self.panel), "--verification", str(ver))
        self.assertEqual(r.returncode, 0, r.stderr)


class TestExtendedRubric(unittest.TestCase):
    def setUp(self):
        panel = panel_with([panelist("a", "Plan A.", label="a"),
                            panelist("b", "Plan B.", label="b")], panel_id="ext")
        self.panel = write_json("/tmp/v1-ext-panel.json", panel)
        criteria = [{"id": cid, "name": cid, "description": f"d for {cid}", "weight": 1 / 6}
                    for cid in CRITERIA]
        criteria.append({"id": "environmental_impact", "name": "Environmental impact",
                         "description": "Low environmental harm.", "weight": 1 / 6})
        self.rubric = write_json("/tmp/v1-ext-rubric.json", {
            "rubric_schema_version": 2, "criteria": criteria,
            "checkpoint_mapping": {"environmental_impact": "feasibility"}})
        scores = {**all_scores(4), "environmental_impact": 5}
        scores_b = {**all_scores(3), "environmental_impact": 2}
        self.assess = write_json("/tmp/v1-ext-assessments.json", {
            "assessor": "h", "assessor_role": "human", "assessments": [
                {"panelist_id": "a", "scores": scores,
                 "reasons": {cid: "r" for cid in scores}},
                {"panelist_id": "b", "scores": scores_b,
                 "reasons": {cid: "r" for cid in scores_b}}]})

    def test_extended_supplied_route(self):
        out = Path("/tmp/v1-ext-out")
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(self.rubric),
                "--assessments", str(self.assess), "--out-dir", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads((out / "verdict.json").read_text())
        self.assertTrue(v["state"]["rubric_version"].startswith("custom-extended:"))
        self.assertEqual(v["state"]["rubric_criteria"],
                         CRITERIA + ["environmental_impact"])
        self.assertEqual(v["state"]["checkpoint_mapping"]["environmental_impact"], "feasibility")
        self.assertEqual(len(v["rubric"]), 6)
        feas = [c for c in v["checkpoints"] if c["checkpoint_id"] == "feasibility"
                and c["strategy_label"] == "a"][0]
        self.assertIn("environmental_impact", feas["scores"])
        self.assertIn("Extended rubric: 6 criteria including environmental_impact",
                      (out / "report.md").read_text())
        self.assertIn("| Criterion | Checkpoint |", (out / "report.md").read_text())

    def test_heuristic_rejects_extended_rubric_named_error(self):
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(self.rubric),
                "--heuristic", "--out-dir", "/tmp/v1-ext-heur-out")
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("heuristic-cannot-score-extended-rubric", r.stderr)
        self.assertFalse(Path("/tmp/v1-ext-heur-out/verdict.json").exists())

    def test_extended_validation_rules(self):
        # Cap: 4 added criteria (9 total) -> fail.
        criteria = [{"id": cid, "name": cid, "description": "d", "weight": 1 / 9}
                    for cid in CRITERIA]
        for extra in ("alpha_one", "alpha_two", "alpha_three", "alpha_four"):
            criteria.append({"id": extra, "name": extra, "description": "d", "weight": 1 / 9})
        rp = write_json("/tmp/v1-ext-cap.json", {
            "rubric_schema_version": 2, "criteria": criteria,
            "checkpoint_mapping": {e: "risk" for e in
                                   ("alpha_one", "alpha_two", "alpha_three", "alpha_four")}})
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(rp),
                "--heuristic", "--out-dir", "/tmp/v1-ext-cap-out")
        self.assertNotEqual(r.returncode, 0)
        # Alias collision: added id "evidence" -> fail.
        criteria = [{"id": cid, "name": cid, "description": "d", "weight": 1 / 6}
                    for cid in CRITERIA]
        criteria.append({"id": "evidence", "name": "Evidence", "description": "d", "weight": 1 / 6})
        rp = write_json("/tmp/v1-ext-alias.json", {
            "rubric_schema_version": 2, "criteria": criteria,
            "checkpoint_mapping": {"evidence": "evidence"}})
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(rp),
                "--heuristic", "--out-dir", "/tmp/v1-ext-alias-out")
        self.assertNotEqual(r.returncode, 0)
        # Canonical mapping contradiction -> fail.
        criteria = [{"id": cid, "name": cid, "description": "d", "weight": 1 / 6}
                    for cid in CRITERIA]
        criteria.append({"id": "accessibility", "name": "Accessibility",
                         "description": "d", "weight": 1 / 6})
        rp = write_json("/tmp/v1-ext-contra.json", {
            "rubric_schema_version": 2, "criteria": criteria,
            "checkpoint_mapping": {"accessibility": "risk", "evidence_strength": "risk"}})
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(rp),
                "--heuristic", "--out-dir", "/tmp/v1-ext-contra-out")
        self.assertNotEqual(r.returncode, 0)
        # Missing added mapping -> fail.
        rp = write_json("/tmp/v1-ext-missing.json", {
            "rubric_schema_version": 2, "criteria": criteria})
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(rp),
                "--heuristic", "--out-dir", "/tmp/v1-ext-missing-out")
        self.assertNotEqual(r.returncode, 0)
        # Canonical entries may be omitted (fixed mapping applies) -> pass.
        rp = write_json("/tmp/v1-ext-ok.json", {
            "rubric_schema_version": 2, "criteria": criteria,
            "checkpoint_mapping": {"accessibility": "risk"}})
        scores = {**all_scores(4), "accessibility": 4}
        scores_b = {**all_scores(3), "accessibility": 3}
        a = write_json("/tmp/v1-ext-ok-assess.json", {
            "assessor": "h", "assessor_role": "human", "assessments": [
                {"panelist_id": "a", "scores": scores, "reasons": {cid: "r" for cid in scores}},
                {"panelist_id": "b", "scores": scores_b, "reasons": {cid: "r" for cid in scores_b}}]})
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(rp),
                "--assessments", str(a), "--out-dir", "/tmp/v1-ext-ok-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        # Schema-version mismatch -> fail.
        rp = write_json("/tmp/v1-ext-schema.json", {
            "rubric_schema_version": 1, "criteria": criteria,
            "checkpoint_mapping": {"accessibility": "risk"}})
        r = cli("judge", "--panel", str(self.panel), "--rubric", str(rp),
                "--assessments", str(a), "--out-dir", "/tmp/v1-ext-schema-out")
        self.assertNotEqual(r.returncode, 0)


class TestSignOff(unittest.TestCase):
    def _verdict(self, out="/tmp/v1-signoff-out"):
        # Fresh dir per test run: a previous run's sign-off.json must not
        # make the first sign-off in this run fail.
        stale = Path(out) / "sign-off.json"
        if stale.exists():
            stale.unlink()
        r = cli("judge", "--panel", str(FIXTURE), "--heuristic", "--out-dir", out)
        self.assertEqual(r.returncode, 0, r.stderr)
        return Path(out) / "verdict.json"

    def test_sign_off_happy_path_and_resign_rules(self):
        vpath = self._verdict()
        r = cli("sign-off", "--verdict", str(vpath), "--approver", "Siva Borra",
                "--decision", "approved", "--note", "Read the report.")
        self.assertEqual(r.returncode, 0, r.stderr)
        record = json.loads((vpath.parent / "sign-off.json").read_text())
        import hashlib
        self.assertEqual(record["verdict_sha256"], hashlib.sha256(vpath.read_bytes()).hexdigest())
        self.assertEqual(record["decision"], "approved")
        self.assertEqual(record["approver"], "Siva Borra")
        self.assertEqual(record["verification_outcome"], "not-run")
        self.assertEqual(record["tool_version"], "panel-judge 1.1.0")
        # Re-sign without --force fails; with --force succeeds.
        r = cli("sign-off", "--verdict", str(vpath), "--approver", "Siva Borra",
                "--decision", "approved")
        self.assertNotEqual(r.returncode, 0)
        r = cli("sign-off", "--verdict", str(vpath), "--approver", "Siva Borra",
                "--decision", "not-approved", "--note", "Changed mind.", "--force")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads((vpath.parent / "sign-off.json").read_text())["decision"],
                         "not-approved")
        # Sign-off never modifies the verdict: state.sign_off stays None
        # (approval lives only in the sign-off.json sidecar).
        v_after = json.loads(vpath.read_text())
        self.assertIsNone(v_after["state"]["sign_off"])

    def test_sign_off_refuses_fact_checking_note_unless_passed(self):
        vpath = self._verdict(out="/tmp/v1-signoff-out2")
        r = cli("sign-off", "--verdict", str(vpath), "--approver", "Siva Borra",
                "--decision", "approved", "--note", "Fact-checked and verified.")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((vpath.parent / "sign-off.json").exists())
        r = cli("sign-off", "--verdict", str(vpath), "--approver", "Siva Borra",
                "--decision", "approved", "--note", "NOT VERIFIED, do not use yet.")
        self.assertNotEqual(r.returncode, 0)  # conservative: contains "verified"

    def test_sign_off_rejects_bad_disclaimer(self):
        vpath = self._verdict(out="/tmp/v1-signoff-out3")
        v = json.loads(vpath.read_text())
        v["disclaimer"] = "Different disclaimer."
        vpath.write_text(json.dumps(v))
        r = cli("sign-off", "--verdict", str(vpath), "--approver", "Siva Borra",
                "--decision", "approved")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse((vpath.parent / "sign-off.json").exists())


class TestHelpers(unittest.TestCase):
    def test_openrouter_helper_key_absence_and_userinfo_and_dry_run(self):
        env = {"PATH": "/usr/bin:/bin"}
        r = subprocess.run([sys.executable, "panel_judge/openrouter_judge.py",
                            "--panel", str(FIXTURE), "--model", "m",
                            "--out", "/tmp/v1-or-out.json"],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 1)
        self.assertIn("OPENROUTER_API_KEY", r.stderr)
        self.assertFalse(Path("/tmp/v1-or-out.json").exists())
        r = subprocess.run([sys.executable, "panel_judge/openrouter_judge.py",
                            "--panel", str(FIXTURE), "--model", "m",
                            "--base-url", "https://u:p@openrouter.ai/api/v1",
                            "--out", "/tmp/v1-or-out.json"],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 1)
        r = subprocess.run([sys.executable, "panel_judge/openrouter_judge.py",
                            "--panel", str(FIXTURE), "--model", "m", "--dry-run"],
                           cwd=ROOT, capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("openrouter-judge-v2", r.stdout)

    def test_openrouter_template_sha_is_template_only(self):
        sys.path.insert(0, str(ROOT))
        from panel_judge.openrouter_judge import PROMPT_TEMPLATE, prompt_sha256
        import hashlib
        self.assertEqual(prompt_sha256(),
                         hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest())
        self.assertNotIn("{question}", PROMPT_TEMPLATE.replace("{question}", ""))
        # Two prompts built from different panels share the template hash.
        from panel_judge.openrouter_judge import build_prompt, CANONICAL_CRITERIA
        p1 = build_prompt("Q1?", [{"id": "a", "answer": "A", "strategy_label": None,
                                   "answer_kind": "summary"}], CANONICAL_CRITERIA)
        p2 = build_prompt("Q2?", [{"id": "b", "answer": "B", "strategy_label": None,
                                   "answer_kind": "summary"}], CANONICAL_CRITERIA)
        self.assertNotEqual(p1, p2)  # instantiated prompts differ
        self.assertEqual(prompt_sha256(), prompt_sha256())  # template hash stable

    def test_openrouter_resume_rejects_corrupt_partial(self):
        sys.path.insert(0, str(ROOT))
        from panel_judge.openrouter_judge import validate_partial_file, HelperError
        partial = {"assessor_model": "m", "prompt_version": "openrouter-judge-v2",
                   "assessments": [
                       {"panelist_id": "a", "scores": {cid: 4 for cid in CRITERIA},
                        "reasons": {cid: "r" for cid in CRITERIA}},
                       {"panelist_id": "zzz", "scores": {cid: 4 for cid in CRITERIA},
                        "reasons": {cid: "r" for cid in CRITERIA}}]}
        p = write_json("/tmp/v1-resume-bad.json", partial)
        with self.assertRaises(HelperError):
            validate_partial_file(str(p), [{"id": "a"}, {"id": "b"}], CRITERIA, "m")
        partial["assessments"] = partial["assessments"][:1]
        partial["assessor_model"] = "other-model"
        p = write_json("/tmp/v1-resume-bad2.json", partial)
        with self.assertRaises(HelperError):
            validate_partial_file(str(p), [{"id": "a"}, {"id": "b"}], CRITERIA, "m")

    def test_ollama_helper_emits_full_envelope_shape(self):
        # Dry-run + envelope fields are structural; live Ollama is NOT
        # exercised. review fixes: prompt v4 (null skeleton, hash is
        # the fully rendered dummy-panel prompt + repair suffix).
        sys.path.insert(0, str(ROOT))
        from panel_judge import ollama_judge
        self.assertEqual(ollama_judge.PROMPT_VERSION, "ollama-judge-v4")
        import hashlib
        self.assertEqual(ollama_judge.prompt_sha256(),
                         hashlib.sha256(ollama_judge._rendered_prompt_material().encode()).hexdigest())
        self.assertIn("ollama-judge-v4", ollama_judge.build_prompt(
            "Q?", [{"id": "a", "answer": "A"}]))

    def test_ollama_helper_parses_its_own_skeleton_output(self):
        # P0-1: build_prompt()'s skeleton keys entries by panelist_id,
        # so parse_and_validate() must accept that key (panelist_id
        # first; panelist/id/name as legacy aliases), and must reject
        # differing identifier keys in one entry. Offline, no live
        # Ollama run. Skeleton-shaped, not the literal skeleton: the
        # literal skeleton's empty reasons would fail validation.
        sys.path.insert(0, str(ROOT))
        from panel_judge import ollama_judge
        import json as _json

        def entry(**extra):
            base = {"scores": {c: 4 for c in ollama_judge.CRITERIA},
                    "reasons": {c: "r" for c in ollama_judge.CRITERIA}}
            base.update(extra)
            return base

        skeleton_shaped = _json.dumps({"assessments": [
            entry(panelist_id="a"),
            entry(panelist_id="b"),
        ]})
        parsed = ollama_judge.parse_and_validate(skeleton_shaped, ["a", "b"])
        self.assertEqual([e["panelist_id"] for e in parsed["assessments"]],
                         ["a", "b"])
        self.assertEqual(parsed["assessments"][0]["scores"]["evidence"], 4)
        # Legacy panelist-keyed form still parses.
        legacy = _json.dumps({"assessments": [entry(panelist="a")]})
        self.assertEqual(
            ollama_judge.parse_and_validate(legacy, ["a"])["assessments"][0]["panelist_id"],
            "a")
        # Differing identifier keys in one entry are rejected.
        differing = _json.dumps({"assessments": [entry(panelist_id="a", panelist="b")]})
        with self.assertRaises(ollama_judge.JudgeError):
            ollama_judge.parse_and_validate(differing, ["a", "b"])


class TestReportRules(unittest.TestCase):
    def test_dissent_blockquoted_with_answer_kind_and_summary_prose_lead_in(self):
        out = Path("/tmp/v1-report-out")
        r = cli("judge", "--panel", str(FIXTURE), "--heuristic", "--out-dir", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        report = (out / "report.md").read_text()
        self.assertIn("> [grok-4-7, summary paraphrase]:", report)
        self.assertIn("Best-supported panelist(s) in the recommended strategy:", report)
        self.assertIn("Human sign-off: none at judging time", report)
        self.assertIn("Judged on summary answers", report)
        # The recommended summary is verdict prose (not blockquoted as a whole).
        summary = json.loads((out / "verdict.json").read_text())["verdict"]["recommended_strategy_summary"]
        self.assertIn(summary, report)

    def test_timestamp_format_and_tool_version(self):
        out = Path("/tmp/v1-ts-out")
        r = cli("judge", "--panel", str(FIXTURE), "--heuristic", "--out-dir", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads((out / "verdict.json").read_text())
        self.assertRegex(v["state"]["created"],
                         r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?Z$")
        self.assertEqual(v["state"]["tool_version"], "panel-judge 1.1.0")


if __name__ == "__main__":
    unittest.main()
