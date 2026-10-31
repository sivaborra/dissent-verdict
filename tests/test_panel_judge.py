#!/usr/bin/env python3
"""Tests for the generic panel-judge library. Run: python3 -m unittest discover -s tests -v"""
import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "fixtures" / "petrol-car-panel.json"
CRITERIA = ["evidence_strength", "capital_efficiency", "regulatory_runway",
            "differentiation", "risk_mitigation"]


def run_judge(panel, out_dir, extra=None):
    cmd = [sys.executable, "-m", "panel_judge", "judge", "--panel", str(panel),
           "--heuristic", "--out-dir", str(out_dir)] + (extra or [])
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def run_judge_assessments(panel, assessments, out_dir):
    cmd = [sys.executable, "-m", "panel_judge", "judge", "--panel", str(panel),
           "--assessments", str(assessments), "--out-dir", str(out_dir)]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)


def write_json(path, obj):
    p = Path(path)
    p.write_text(json.dumps(obj))
    return p


def panel_with(panelists, panel_id="synthetic"):
    return {"panel_id": panel_id, "question": "Which strategy is best?",
            "panelists": panelists}


def assessments_for(panelists_scores, checkpoints=None, assessor="test assessor",
                    assessor_panelist_id=None, assessor_role="human"):
    """panelists_scores: {panelist_id: {criterion: score}}; reasons auto-filled."""
    assessments = []
    for pid, scores in panelists_scores.items():
        assessments.append({
            "panelist_id": pid,
            "scores": dict(scores),
            "reasons": {cid: f"reason for {cid}" for cid in CRITERIA},
            "strengths": [], "weaknesses": []})
    obj = {"assessor": assessor, "assessor_role": assessor_role, "assessments": assessments}
    if checkpoints is not None:
        obj["checkpoints"] = checkpoints
    if assessor_panelist_id is not None:
        obj["assessor_panelist_id"] = assessor_panelist_id
    return obj


def all_scores(value):
    return {cid: value for cid in CRITERIA}


def panelist(pid, answer, label=None, is_dissent=False):
    p = {"id": pid, "model": pid, "source": "model", "answer": answer}
    if label is not None:
        p["strategy_label"] = label
    if is_dissent:
        p["is_dissent"] = True
    return p


class TestPanelJudge(unittest.TestCase):
    def test_validate_fixture(self):
        r = subprocess.run([sys.executable, "-m", "panel_judge", "validate",
                            "--panel", str(FIXTURE)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)

    def test_petrol_fixture_verdict_direction(self):
        """Boutique sports wins, Grok's work-4x4 is dissent/fallback — via rubric, not a hard-coded answer."""
        out = Path("/tmp/panel-judge-test-out")
        r = run_judge(FIXTURE, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads((out / "verdict.json").read_text())
        self.assertEqual(v["verdict"]["recommended_strategy_label"], "boutique-sports")
        self.assertEqual(v["verdict"]["leading_strategy_label"], "boutique-sports")
        self.assertEqual(v["verdict"]["consensus"], "majority")
        self.assertEqual([d["panelist_id"] for d in v["dissent"]], ["grok-4-7"])
        self.assertIn("fallback", v["dissent"][0]["verdict_response"])
        self.assertTrue(v["dissent"][0]["verdict_response"].startswith("partially adopted"))
        self.assertIn("strategy_label", v["dissent"][0])
        self.assertIn("material_difference_basis", v["dissent"][0])
        self.assertEqual(v["disclaimer"],
                         "Scores rate the answers against the stated criteria. "
                         "They are not a guarantee of success, market outcome, or regulatory outcome.")
        # v0.2: one checkpoint entry per (checkpoint, strategy group) pair.
        self.assertEqual(len(v["checkpoints"]), 6)
        self.assertEqual(sorted({c["strategy_label"] for c in v["checkpoints"]}),
                         ["boutique-sports", "work-4x4"])
        for c in v["checkpoints"]:
            if c["checkpoint_id"] == "risk":
                self.assertIn("risk_register", c)
        self.assertEqual(v["checkpoint_disagreements"], [])
        # G22: key order is presentation only; presence is normative.
        self.assertEqual(set(v.keys()),
                         {"panel_id", "question", "verdict", "rubric", "checkpoints",
                          "checkpoint_disagreements", "per_panelist", "dissent",
                          "disclaimer", "state", "verification"})
        self.assertIsNone(v["verification"])
        for row in v["rubric"]:
            self.assertIn("weight", row)
        for p in v["per_panelist"]:
            self.assertIn("answer_kind", p)
            self.assertIn("reasons", p)
            self.assertIn("overall_score", p)
            self.assertEqual(set(p["reasons"]), set(CRITERIA))
        st = v["state"]
        for key in ("rubric_file", "assessments_file", "assessor", "ollama_endpoint",
                    "prompt_version", "max_panel_chars", "panel_chars",
                    "panel_chars_after_truncation", "truncated_panelist_ids"):
            self.assertIn(key, st)
        self.assertIsNone(st["panel_chars_after_truncation"])
        self.assertEqual(st["truncated_panelist_ids"], [])
        report = (out / "report.md").read_text()
        self.assertIn("boutique-sports", report)
        self.assertIn(v["disclaimer"], report)
        self.assertIn("Checkpoint disagreements", report)

    def test_no_consensus_on_tied_panel(self):
        panel = {"panel_id": "tied", "question": "Q?",
                 "panelists": [
                     {"id": "a", "model": "A", "source": "model", "answer": "Do X with bought-in parts.",
                      "strategy_label": "x"},
                     {"id": "b", "model": "B", "source": "model", "answer": "Do Y with bought-in parts.",
                      "strategy_label": "y"}]}
        p = Path("/tmp/tied-panel.json")
        p.write_text(json.dumps(panel))
        r = run_judge(p, "/tmp/panel-judge-tied-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(Path("/tmp/panel-judge-tied-out/verdict.json").read_text())
        self.assertEqual(v["verdict"]["consensus"], "no-consensus")
        self.assertIsNone(v["verdict"]["recommended_strategy_label"])

    def test_panel_with_verdict_is_rejected(self):
        panel = {"panel_id": "bad", "question": "Q?", "verdict": {},
                 "panelists": [
                     {"id": "a", "model": "A", "source": "model", "answer": "X"},
                     {"id": "b", "model": "B", "source": "model", "answer": "Y"}]}
        p = Path("/tmp/bad-panel.json")
        p.write_text(json.dumps(panel))
        r = run_judge(p, "/tmp/panel-judge-bad-out")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(Path("/tmp/panel-judge-bad-out/verdict.json").exists())

    # ---- v0.2 synthetic cases -------------------------------------------

    def test_no_consensus_low_scores_heuristic(self):
        """All scores low and tied: no-consensus, no leader, unresolved dissent."""
        panel = panel_with([
            panelist("a", "Do X.", label="x"),
            panelist("b", "Do Y.", label="y")], panel_id="low")
        p = write_json("/tmp/low-panel.json", panel)
        r = run_judge(p, "/tmp/panel-judge-low-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(Path("/tmp/panel-judge-low-out/verdict.json").read_text())
        self.assertEqual(v["verdict"]["consensus"], "no-consensus")
        self.assertIsNone(v["verdict"]["recommended_strategy_label"])
        self.assertIn("No strategy is recommended", v["verdict"]["recommended_strategy_summary"])
        # Tied low scores here: no leader, empty rubric, null overall.
        self.assertIsNone(v["verdict"]["leading_strategy_label"])
        self.assertEqual(v["rubric"], [])
        self.assertIsNone(v["verdict"]["overall_score"])
        self.assertEqual(sorted(d["panelist_id"] for d in v["dissent"]), ["a", "b"])
        for d in v["dissent"]:
            self.assertTrue(d["verdict_response"].startswith("unresolved"), d["verdict_response"])

    def test_no_consensus_unique_leader_supplied(self):
        """Top below 2.5 with a unique leader: leading (not recommended) scores shown."""
        panel = panel_with([
            panelist("a", "Plan A for the United States.", label="a"),
            panelist("b", "Plan B for the United States.", label="b")], panel_id="nc-leader")
        p = write_json("/tmp/nc-leader-panel.json", panel)
        a = write_json("/tmp/nc-leader-assessments.json", assessments_for({
            "a": all_scores(2), "b": all_scores(1)}))
        r = run_judge_assessments(p, a, "/tmp/panel-judge-nc-leader-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(Path("/tmp/panel-judge-nc-leader-out/verdict.json").read_text())
        self.assertEqual(v["verdict"]["consensus"], "no-consensus")
        self.assertIsNone(v["verdict"]["recommended_strategy_label"])
        self.assertEqual(v["verdict"]["leading_strategy_label"], "a")
        self.assertEqual(v["verdict"]["overall_score"], 2.0)
        self.assertEqual({row["criterion_id"]: row["score"] for row in v["rubric"]},
                         all_scores(2))
        self.assertIn("leading, not recommended", v["verdict"]["recommended_strategy_summary"])
        self.assertEqual([d["panelist_id"] for d in v["dissent"]], ["b"])
        self.assertTrue(v["dissent"][0]["verdict_response"].startswith("unresolved"))

    def test_tied_top_supplied(self):
        """Exact tie at a passing score: margin 0 => no-consensus, tied top."""
        panel = panel_with([
            panelist("a", "Plan A.", label="a"),
            panelist("b", "Plan B.", label="b")], panel_id="tied-top")
        p = write_json("/tmp/tied-top-panel.json", panel)
        a = write_json("/tmp/tied-top-assessments.json", assessments_for({
            "a": all_scores(3), "b": all_scores(3)}))
        r = run_judge_assessments(p, a, "/tmp/panel-judge-tied-top-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(Path("/tmp/panel-judge-tied-top-out/verdict.json").read_text())
        self.assertEqual(v["verdict"]["consensus"], "no-consensus")
        self.assertIsNone(v["verdict"]["recommended_strategy_label"])
        self.assertIsNone(v["verdict"]["leading_strategy_label"])
        self.assertEqual(v["rubric"], [])
        self.assertIsNone(v["verdict"]["overall_score"])
        self.assertIn("tied", v["verdict"]["recommended_strategy_summary"])
        self.assertEqual(sorted(d["panelist_id"] for d in v["dissent"]), ["a", "b"])
        for d in v["dissent"]:
            self.assertTrue(d["verdict_response"].startswith("unresolved"))

    def test_minority_win_supplied(self):
        """A lone group out-scores a 3-member group: minority, winner stays in dissent as adopted."""
        panel = panel_with([
            panelist("solo", "Solo plan.", label="solo"),
            panelist("m1", "Big plan.", label="big"),
            panelist("m2", "Big plan.", label="big"),
            panelist("m3", "Big plan.", label="big")], panel_id="minority-win")
        p = write_json("/tmp/minority-panel.json", panel)
        a = write_json("/tmp/minority-assessments.json", assessments_for({
            "solo": all_scores(4), "m1": all_scores(3), "m2": all_scores(3), "m3": all_scores(3)}))
        r = run_judge_assessments(p, a, "/tmp/panel-judge-minority-out")
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads(Path("/tmp/panel-judge-minority-out/verdict.json").read_text())
        self.assertEqual(v["verdict"]["consensus"], "minority")
        self.assertEqual(v["verdict"]["recommended_strategy_label"], "solo")
        self.assertEqual(v["verdict"]["leading_strategy_label"], "solo")
        dissent = {d["panelist_id"]: d for d in v["dissent"]}
        self.assertIn("solo", dissent)
        self.assertTrue(dissent["solo"]["verdict_response"].startswith("adopted"),
                        dissent["solo"]["verdict_response"])
        self.assertIn("became the recommendation", dissent["solo"]["verdict_response"])
        for pid in ("m1", "m2", "m3"):
            self.assertIn(pid, dissent)

    def test_same_label_material_dissent_supplied(self):
        """Same label, disjoint markets => dissent; flagged-but-not-material => excluded + footer."""
        panel = panel_with([
            panelist("p1", "Sell in the United States and Middle East.", label="alpha"),
            panelist("p2", "Sell in the United States and Middle East.", label="alpha"),
            panelist("p3", "Sell only in Australia.", label="alpha"),
            panelist("p4", "Sell in the United States and Middle East.", label="alpha",
                     is_dissent=True)], panel_id="same-label")
        p = write_json("/tmp/same-label-panel.json", panel)
        a = write_json("/tmp/same-label-assessments.json", assessments_for({
            "p1": all_scores(4), "p2": all_scores(4), "p3": all_scores(4), "p4": all_scores(4)}))
        out = Path("/tmp/panel-judge-same-label-out")
        r = run_judge_assessments(p, a, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads((out / "verdict.json").read_text())
        # Single group => majority; only the disjoint-market panelist is dissent.
        self.assertEqual(v["verdict"]["consensus"], "majority")
        self.assertEqual([d["panelist_id"] for d in v["dissent"]], ["p3"])
        self.assertIn("target market", v["dissent"][0]["material_difference_basis"])
        report = (out / "report.md").read_text()
        self.assertIn("p4", report)
        self.assertIn("Excluded flagged panelists", report)

    def test_same_label_no_market_named_is_not_dissent(self):
        """Regression: a same-label panelist naming NO market is not material dissent.

        An empty market set must not count as disjoint from the recommended
        group's markets (this bug falsely listed fixture panelists as dissent
        in a v0.1 re-eval and failed test_petrol_fixture_verdict_direction).
        """
        panel = panel_with([
            panelist("p1", "Sell in the United States and Middle East.", label="alpha"),
            panelist("p2", "Sell in the United States and Middle East.", label="alpha"),
            panelist("p3", "Sell a good product at a fair price.", label="alpha")],
            panel_id="empty-market")
        p = write_json("/tmp/empty-market-panel.json", panel)
        a = write_json("/tmp/empty-market-assessments.json", assessments_for({
            "p1": all_scores(4), "p2": all_scores(4), "p3": all_scores(4)}))
        out = Path("/tmp/panel-judge-empty-market-out")
        r = run_judge_assessments(p, a, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads((out / "verdict.json").read_text())
        self.assertEqual(v["verdict"]["consensus"], "majority")
        self.assertEqual(v["dissent"], [])

    def test_custom_rubric_renamed_id_rejected(self):
        rubric = {"criteria": [
            {"id": "evidence", "name": "Evidence", "description": "d", "weight": 0.2},
            {"id": "capital_efficiency", "name": "Capital", "description": "d", "weight": 0.2},
            {"id": "regulatory_runway", "name": "Regulatory", "description": "d", "weight": 0.2},
            {"id": "differentiation", "name": "Differentiation", "description": "d", "weight": 0.2},
            {"id": "risk_mitigation", "name": "Risk", "description": "d", "weight": 0.2}]}
        rp = write_json("/tmp/renamed-rubric.json", rubric)
        panel = panel_with([panelist("a", "Plan A.", label="a"),
                            panelist("b", "Plan B.", label="b")], panel_id="renamed")
        p = write_json("/tmp/renamed-panel.json", panel)
        r = run_judge(p, "/tmp/panel-judge-renamed-out", extra=["--rubric", str(rp)])
        self.assertNotEqual(r.returncode, 0)
        # Partial weights (all-or-none rule) are also rejected.
        partial = {"criteria": [
            {"id": cid, "name": cid, "description": "d",
             **({"weight": 0.2} if i == 0 else {})} for i, cid in enumerate(CRITERIA)]}
        rp2 = write_json("/tmp/partial-rubric.json", partial)
        r2 = run_judge(p, "/tmp/panel-judge-partial-out", extra=["--rubric", str(rp2)])
        self.assertNotEqual(r2.returncode, 0)

    def test_panel_with_forbidden_extra_key_rejected(self):
        panel = panel_with([panelist("a", "Plan A.", label="a"),
                            panelist("b", "Plan B.", label="b")], panel_id="extra-key")
        panel["dissent"] = []
        p = write_json("/tmp/extra-key-panel.json", panel)
        r = run_judge(p, "/tmp/panel-judge-extra-key-out")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(Path("/tmp/panel-judge-extra-key-out/verdict.json").exists())
        panel2 = panel_with([panelist("a", "Plan A.", label="a"),
                             {**panelist("b", "Plan B.", label="b"), "foo": 1}],
                            panel_id="extra-panelist-key")
        p2 = write_json("/tmp/extra-panelist-key-panel.json", panel2)
        r2 = run_judge(p2, "/tmp/panel-judge-extra-panelist-key-out")
        self.assertNotEqual(r2.returncode, 0)

    def test_supplied_checkpoint_mismatch_recorded(self):
        """A supplied checkpoint score differing from the synthesised score
        goes to checkpoint_disagreements with resolution = synthesised score."""
        panel = panel_with([
            panelist("a", "Plan A.", label="alpha"),
            panelist("b", "Plan A.", label="alpha")], panel_id="cp-mismatch")
        p = write_json("/tmp/cp-mismatch-panel.json", panel)
        supplied = assessments_for({"a": all_scores(4), "b": all_scores(3)},
                                   checkpoints=[{"checkpoint_id": "evidence",
                                                 "scores": {"evidence_strength": 1},
                                                 "notes": "supplied evidence note"}])
        a = write_json("/tmp/cp-mismatch-assessments.json", supplied)
        out = Path("/tmp/panel-judge-cp-mismatch-out")
        r = run_judge_assessments(p, a, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads((out / "verdict.json").read_text())
        entries = [d for d in v["checkpoint_disagreements"]
                   if d["criterion_id"] == "evidence_strength"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["resolution"], 4)  # max supplied per-panelist score
        self.assertEqual(entries[0]["checkpoint_scores"],
                         {"supplied": 1, "synthesised": 4})
        report = (out / "report.md").read_text()
        self.assertIn("Checkpoint disagreements", report)
        self.assertIn("evidence_strength", report)
        # A supplied checkpoint scoring a criterion it does not own is rejected.
        bad = assessments_for({"a": all_scores(4), "b": all_scores(3)},
                              checkpoints=[{"checkpoint_id": "evidence",
                                            "scores": {"risk_mitigation": 2},
                                            "notes": "wrong checkpoint"}])
        ab = write_json("/tmp/cp-bad-assessments.json", bad)
        r2 = run_judge_assessments(p, ab, "/tmp/panel-judge-cp-bad-out")
        self.assertNotEqual(r2.returncode, 0)

    def test_self_judging_only_via_assessor_panelist_id(self):
        """Assessor name matching a panelist model must NOT trigger disclosure;
        only assessor_panelist_id does."""
        panel = panel_with([
            panelist("a", "Plan A.", label="alpha"),
            panelist("b", "Plan B.", label="beta")], panel_id="selfjudge")
        p = write_json("/tmp/selfjudge-panel.json", panel)
        # Assessor string equals panelist a's model name, but no panelist id given.
        a = write_json("/tmp/selfjudge-assessments.json",
                       assessments_for({"a": all_scores(4), "b": all_scores(3)},
                                       assessor="a"))
        out = Path("/tmp/panel-judge-selfjudge-out")
        r = run_judge_assessments(p, a, out)
        self.assertEqual(r.returncode, 0, r.stderr)
        v = json.loads((out / "verdict.json").read_text())
        self.assertFalse(v["state"]["self_judging"])
        self.assertNotIn("SELF-JUDGING", (out / "report.md").read_text())
        # With assessor_panelist_id naming a panelist, disclosure appears.
        a2 = write_json("/tmp/selfjudge2-assessments.json",
                        assessments_for({"a": all_scores(4), "b": all_scores(3)},
                                        assessor="Judge Model", assessor_panelist_id="a"))
        out2 = Path("/tmp/panel-judge-selfjudge2-out")
        r2 = run_judge_assessments(p, a2, out2)
        self.assertEqual(r2.returncode, 0, r2.stderr)
        v2 = json.loads((out2 / "verdict.json").read_text())
        self.assertTrue(v2["state"]["self_judging"])
        report2 = (out2 / "report.md").read_text()
        self.assertIn("The assessor Judge Model also sat on this panel as a.", report2)


if __name__ == "__main__":
    unittest.main()
