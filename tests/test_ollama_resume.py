"""Tests for ollama_judge --resume-from and --per-panelist (v1.1 unblock 1).

These tests use a stub panel and a stubbed call_ollama; no network, no
model. A resumed assessment is never re-judged, and a corrupt resume
source is rejected whole, never silently merged.
"""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from panel_judge import ollama_judge as oj


PANEL = {
    "question": "Q?",
    "panelists": [
        {"id": "alpha", "answer": "Alpha answer."},
        {"id": "beta", "answer": "Beta answer."},
    ],
}


def assessment(pid, score=3):
    return {
        "panelist_id": pid,
        "scores": {c: score for c in oj.CRITERIA},
        "reasons": {c: "reason for %s" % c for c in oj.CRITERIA},
    }


def envelope(model, entries):
    return {
        "assessor": model, "assessor_role": "judge-model",
        "assessor_model": model, "assessor_provider": "ollama",
        "prompt_version": oj.PROMPT_VERSION,
        "prompt_sha256": oj.prompt_sha256(),
        "rubric_version": oj.HELPER_RUBRIC_VERSION,
        "temperature": 0,
        "panel_sha256": oj.panel_sha256(PANEL["question"], PANEL["panelists"]),
        "assessments": entries,
    }


def model_output(pid, score=4):
    return json.dumps({"assessments": [assessment(pid, score)]})


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.panel_path = os.path.join(self.tmp.name, "panel.json")
        with open(self.panel_path, "w") as h:
            json.dump(PANEL, h)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, name, data):
        path = os.path.join(self.tmp.name, name)
        with open(path, "w") as h:
            json.dump(data, h)
        return path

    def test_resume_combined_merges_missing_without_rejudging(self):
        partial = self._write("partial.json", envelope("qwen3:8b", [assessment("alpha")]))
        out = os.path.join(self.tmp.name, "out.json")
        with mock.patch.object(oj, "call_ollama", return_value=(model_output("beta"), {"model": "qwen3:8b", "prompt_eval_count": 100, "eval_count": 50, "num_ctx": oj.NUM_CTX})) as call:
            rc = oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                          "--resume-from", partial, "--out", out])
        self.assertEqual(rc, 0)
        self.assertEqual(call.call_count, 1)
        data = json.load(open(out))
        self.assertEqual([a["panelist_id"] for a in data["assessments"]], ["alpha", "beta"])
        self.assertEqual(data["assessments"][0]["scores"]["evidence"], 3)  # resumed, untouched
        self.assertEqual(data["assessments"][1]["scores"]["evidence"], 4)  # newly judged

    def test_resume_complete_makes_no_request(self):
        partial = self._write("full.json", envelope("qwen3:8b", [assessment("alpha"), assessment("beta")]))
        out = os.path.join(self.tmp.name, "out.json")
        with mock.patch.object(oj, "call_ollama", side_effect=AssertionError("must not call")):
            rc = oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                          "--resume-from", partial, "--out", out])
        self.assertEqual(rc, 0)

    def test_resume_rejects_wrong_model_and_corrupt_entry(self):
        bad_model = self._write("bad.json", envelope("other-model", [assessment("alpha")]))
        self.assertEqual(oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                                  "--resume-from", bad_model, "--out", os.path.join(self.tmp.name, "o.json")]), 1)
        corrupt = assessment("alpha")
        del corrupt["scores"]["regulatory"]
        bad_entry = self._write("bad2.json", envelope("qwen3:8b", [corrupt]))
        self.assertEqual(oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                                  "--resume-from", bad_entry, "--out", os.path.join(self.tmp.name, "o.json")]), 1)

    def test_resume_from_directory_and_negative_control(self):
        d = os.path.join(self.tmp.name, "resume-dir")
        os.makedirs(d)
        rec = {"panelist_id": "alpha", "assessment": assessment("alpha"),
               "model": "qwen3:8b", "host": "http://example.invalid",
               "prompt_version": oj.PROMPT_VERSION, "prompt_sha256": oj.prompt_sha256(),
               "rubric_version": oj.HELPER_RUBRIC_VERSION, "temperature": 0, "attempts": 1, "repair_used": False, "seconds": 1.0,
               "panel_sha256": oj.panel_sha256(PANEL["question"], PANEL["panelists"])}
        with open(os.path.join(d, "assessment-alpha.json"), "w") as h:
            json.dump(rec, h)
        out = os.path.join(self.tmp.name, "out.json")
        with mock.patch.object(oj, "call_ollama", return_value=(model_output("beta"), {"model": "qwen3:8b", "prompt_eval_count": 100, "eval_count": 50, "num_ctx": oj.NUM_CTX})):
            rc = oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                          "--resume-from", d, "--out", out])
        self.assertEqual(rc, 0)
        # Negative control: break the saved file (missing criterion).
        bad = dict(rec); bad["assessment"] = assessment("alpha"); del bad["assessment"]["reasons"]["evidence"]
        with open(os.path.join(d, "assessment-alpha.json"), "w") as h:
            json.dump(bad, h)
        rc = oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                      "--resume-from", d, "--out", os.path.join(self.tmp.name, "o2.json")])
        self.assertEqual(rc, 1)

    def test_per_panelist_saves_immediately_and_skips_saved(self):
        out_dir = os.path.join(self.tmp.name, "per")
        with mock.patch.object(oj, "call_ollama", side_effect=lambda *a: model_output("alpha") if "alpha" in a[2] else model_output("beta")):
            # Note: build_prompt for one panelist contains that id; side_effect above is fragile,
            # so instead use a single mock returning a valid output per call via sequential calls.
            pass
        # Deterministic version: mock returns alpha first, then beta.
        calls = [(model_output("alpha"), {"model": "qwen3:8b", "prompt_eval_count": 100, "eval_count": 50, "num_ctx": oj.NUM_CTX}),
         (model_output("beta"), {"model": "qwen3:8b", "prompt_eval_count": 100, "eval_count": 50, "num_ctx": oj.NUM_CTX})]
        with mock.patch.object(oj, "call_ollama", side_effect=calls):
            rc = oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                          "--per-panelist", "--out-dir", out_dir,
                          "--out", os.path.join(self.tmp.name, "combined.json")])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.exists(os.path.join(out_dir, "assessment-alpha.json")))
        self.assertTrue(os.path.exists(os.path.join(out_dir, "assessment-beta.json")))
        # Second run: both skipped, no model calls.
        with mock.patch.object(oj, "call_ollama", side_effect=AssertionError("must not call")):
            rc = oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                          "--per-panelist", "--out-dir", out_dir])
        self.assertEqual(rc, 0)

    def test_per_panelist_corrupt_saved_file_is_detected_not_overwritten(self):
        out_dir = os.path.join(self.tmp.name, "per2")
        os.makedirs(out_dir)
        bad = assessment("alpha"); bad["scores"]["regulatory"] = 99  # clamped? use missing instead
        del bad["scores"]["regulatory"]
        rec = {"panelist_id": "alpha", "assessment": bad, "model": "qwen3:8b",
               "host": "http://x", "prompt_version": oj.PROMPT_VERSION,
               "prompt_sha256": oj.prompt_sha256(), "rubric_version": oj.HELPER_RUBRIC_VERSION, "temperature": 0,
               "attempts": 1, "repair_used": False, "seconds": 1.0,
               "panel_sha256": oj.panel_sha256(PANEL["question"], PANEL["panelists"])}
        with open(os.path.join(out_dir, "assessment-alpha.json"), "w") as h:
            json.dump(rec, h)
        with mock.patch.object(oj, "call_ollama", return_value=(model_output("beta"), {"model": "qwen3:8b", "prompt_eval_count": 100, "eval_count": 50, "num_ctx": oj.NUM_CTX})):
            rc = oj.main(["--panel", self.panel_path, "--model", "qwen3:8b",
                          "--per-panelist", "--out-dir", out_dir])
        self.assertEqual(rc, 1)
        # File untouched: still missing regulatory.
        data = json.load(open(os.path.join(out_dir, "assessment-alpha.json")))
        self.assertNotIn("regulatory", data["assessment"]["scores"])

    def test_per_panelist_requires_out_dir(self):
        with self.assertRaises(SystemExit):
            oj.main(["--panel", self.panel_path, "--per-panelist"])


class SpellingFixTests(unittest.TestCase):
    """Criterion-key spelling correction changes no content (2026-10-31).

    A comma in a key (reg,ulatory) is a parser normalisation, not a
    score invention: scores, reasons and panelist are unchanged. An
    unknown key, or two keys mapping to the same criterion, must
    still be rejected.
    """

    def test_comma_keys_are_fixed_not_invented(self):
        entry = assessment("alpha", score=4)
        entry["scores"]["reg,ulatory"] = entry["scores"].pop("regulatory")
        entry["reasons"]["risk_mit,igation"] = entry["reasons"].pop("risk_mitigation")
        parsed = oj.parse_and_validate(
            json.dumps({"assessments": [entry]}), ["alpha"])
        got = parsed["assessments"][0]
        self.assertEqual(got["scores"]["regulatory"], 4)
        self.assertEqual(got["scores"]["risk_mitigation"], 4)
        self.assertEqual(got["reasons"]["regulatory"], "reason for regulatory")
        self.assertEqual(got["reasons"]["risk_mitigation"], "reason for risk_mitigation")

    def test_unknown_key_and_duplicate_mapping_still_rejected(self):
        # A key that maps to no known criterion cannot supply that criterion.
        entry = assessment("alpha")
        entry["scores"]["nonsense"] = entry["scores"].pop("regulatory")
        with self.assertRaises(oj.JudgeError):
            oj.parse_and_validate(json.dumps({"assessments": [entry]}), ["alpha"])
        entry = assessment("alpha")
        entry["scores"]["reg,ulatory"] = 2  # duplicates the exact regulatory key
        with self.assertRaises(oj.JudgeError):
            oj.parse_and_validate(json.dumps({"assessments": [entry]}), ["alpha"])


if __name__ == "__main__":
    unittest.main()
