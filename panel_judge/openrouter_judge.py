#!/usr/bin/env python3
"""OpenRouter assessor helper for the panel-judge library (v1.0, outside core).

This helper is NOT imported by core or by the CLI judge command. It turns
a panel file into a scoring prompt, calls one OpenRouter model, validates
the returned JSON, and writes an assessments file that then enters core
only via `panel-judge judge --assessments`. There is no integrated
`--openrouter-model` CLI flag in v1.0.

Secrets: the API key comes from the environment variable
OPENROUTER_API_KEY only -- never from a CLI argument, never printed,
never logged, never written into the assessments file, the verdict, the
report, or state. If the variable is absent or empty, the helper fails
immediately and makes no request -- except for a loopback relay: when
--base-url is loopback (127.0.0.1/localhost) AND the operator passes
--relay, no key is required and no Authorization header is sent (the
relay authenticates itself); a loopback base URL without --relay still
requires the key.

Prompt versioning (P0-1): PROMPT_TEMPLATE below is the template; its
version is openrouter-judge-v2. Version v1 was the Task 1 template used
by the earlier judge-call script (a different template text); this
helper's template differs, so it must not reuse v1. Only a change to
this template text bumps the version. prompt_sha256 is the SHA-256 of
this template text exactly as used, BEFORE panel answers are inserted.
Varying panel answers never bump the version and never change the hash.
The instantiated prompt is never written to any output file; --dry-run
prints it to local stdout only and makes no request.

Transport (P1-3): HTTPS required except for loopback base URLs
(http://127.0.0.1:* or http://localhost:*), which are allowed only as
an explicit operator choice for a local relay. A base URL whose netloc
contains '@' (userinfo / credentials embedded in the URL) is rejected
before any request. TLS certificate verification is never disabled.
A redirect to a different host while carrying the API key fails the
run; same-host redirects use the HTTP library's default behaviour and
the helper never manually re-attaches the key to a redirected request.
Temperature is 0. One request per panel by default (whole panel in one
prompt); --resume-from makes one request for the missing panelists only.

Retries: ONE counter. Total retries across malformed-output retries and
HTTP 429/5xx retries must not exceed --max-retries (default 2).
HTTP 401/403 fail immediately; timeouts fail immediately (the operator
re-runs); after retries are exhausted, or on a rate-limit stop, the
helper exits non-zero and makes no further request. Retries never fall
back to another model, another prompt, or the heuristic scorer.
Malformed-output retry triggers: malformed JSON, missing panelist,
missing criterion, empty reason, unknown panelist, duplicate entry,
non-integer or out-of-range score, bad strengths/weaknesses type.

Cost: provider-reported only, never estimated. OpenRouter bills in USD;
when the provider response reports a cost, it is recorded as
cost_amount with cost_currency "USD" and pricing_source
"provider-response" (the provider's documented USD billing). When the
provider reports no cost, cost fields are null and the report says
"cost not reported by the provider". Token counts are copied verbatim
from the provider response, never recomputed.

Exit codes: 0 success, 1 judging/input failure, 2 usage error (bad
flags, non-positive timeout/retries). Helper errors never echo prompt
text, panel answers, supplied scores/reasons beyond panelist/criterion
IDs, or raw model output; the stderr retry summary records only that a
retry happened and its cause category.

Pure Python standard library only (urllib, json, argparse, sys, os, time).
"""

import argparse
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

PROMPT_VERSION = "openrouter-judge-v2"
DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_TIMEOUT = 300
DEFAULT_MAX_RETRIES = 2

CANONICAL_CRITERIA = (
    ("evidence_strength", "Evidence strength: claims backed by specific, checkable evidence "
     "(named precedents, numbers, markets, mechanisms); low = assertions without evidence, "
     "or evidence not supporting the claim."),
    ("capital_efficiency", "Capital / resource efficiency: little upfront capital and few scarce "
     "resources relative to the goal (bought-in components, build-to-order, low volumes, deposits); "
     "low = proprietary development, large plants, large unsold inventory before revenue."),
    ("regulatory_runway", "Regulatory / legal runway: a named, currently available legal route to "
     "market with a plausible life before that route closes; low = route already closing, unnamed, "
     "or assumed open without argument."),
    ("differentiation", "Differentiation: distinct from incumbents and from the other panelists' "
     "strategies in a way buyers would pay for; low = generic product any incumbent could copy, "
     "or head-on competition with scaled rivals."),
    ("risk_mitigation", "Risk mitigation: the biggest risks named in the answer AND a concrete "
     "response to each (pivot, cap, deposit model, second market); low = risks unnamed, dismissed, "
     "or with no response."),
)
ANCHORS = ("0 = absent or directly contradicted by the answer text. "
           "1 = mentioned but unsupported. 2 = plausible but thin: one weak supporting point, "
           "or a major gap. 3 = adequate: a credible supporting point, with a noted gap. "
           "4 = strong: specific evidence or a concrete mechanism, minor gaps only. "
           "5 = exceptional: specific, checkable evidence AND a concrete mechanism, "
           "with no material gap in the answers.")

PROMPT_TEMPLATE = """You are an independent judge scoring panel answers against a fixed rubric. Return JSON ONLY, no prose, no markdown fences.

Prompt version: {prompt_version}

QUESTION:
{question}

PANEL ANSWERS (faithful summaries or full answers as supplied; a summary is a paraphrase, never a verbatim quote):
{panel_answers}

RUBRIC (score each criterion 0-5 integer, per panelist, judging the STRATEGY in that panelist's answer, not writing quality):
{criteria_block}

SCORE ANCHORS (per criterion): {anchors}

Judge ONLY what is in each answer text. Do not invent evidence, regulations, or risk responses the answer does not state. A risk named with no response must not score high on risk_mitigation.

Return EXACTLY this JSON shape, with an entry for EVERY panelist listed above:
{skeleton}

Every score must be an integer 0-5 and every reason a non-empty string citing the answer text.
"""


def prompt_sha256():
    return hashlib.sha256(PROMPT_TEMPLATE.encode("utf-8")).hexdigest()


class HelperError(Exception):
    """A helper failure: exit 1, no assessments file written."""


def load_panel(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            panel = json.load(handle)
    except OSError as exc:
        raise HelperError(f"cannot read panel file {path!r}: {exc}")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HelperError(f"panel file {path!r} is not valid JSON/UTF-8: {exc}")
    if not isinstance(panel, dict) or not isinstance(panel.get("panelists"), list) or len(panel["panelists"]) < 2:
        raise HelperError("panel file must be an object with a 'panelists' list of at least 2 entries")
    question = panel.get("question")
    if not isinstance(question, str) or not question.strip():
        raise HelperError("panel 'question' must be a non-empty string")
    panelists = []
    seen = set()
    for raw in panel["panelists"]:
        if not isinstance(raw, dict):
            raise HelperError("a panelist entry is not a JSON object")
        pid = raw.get("id")
        answer = raw.get("answer")
        if not isinstance(pid, str) or not pid or not isinstance(answer, str) or not answer.strip():
            raise HelperError("each panelist needs a non-empty string 'id' and 'answer'")
        if pid in seen:
            raise HelperError(f"duplicate panelist id: {pid!r}")
        seen.add(pid)
        panelists.append({"id": pid, "answer": answer,
                          "strategy_label": raw.get("strategy_label"),
                          "answer_kind": raw.get("answer_kind") or "summary"})
    return question, panelists


def load_rubric_for_prompt(path):
    """Minimal rubric parse for prompt construction (canonical or extended).

    Returns a list of (id, description) in rubric-file order, canonical
    five first is NOT enforced here -- the prompt lists criteria in the
    file's order; core validates the rubric itself at judge time. A
    rubric this helper cannot parse fails the helper run (no request).
    """
    if path is None:
        return list(CANONICAL_CRITERIA)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HelperError(f"rubric file {path!r} is not readable JSON: {exc}")
    if isinstance(data, dict):
        criteria = data.get("criteria")
    elif isinstance(data, list):
        criteria = data
    else:
        criteria = None
    if not isinstance(criteria, list) or not criteria:
        raise HelperError("rubric file must be a list of criteria or an object with a 'criteria' list")
    out = []
    for c in criteria:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str) or not c["id"]:
            raise HelperError("each rubric criterion needs a non-empty string 'id'")
        desc = c.get("description")
        if not isinstance(desc, str) or not desc.strip():
            raise HelperError(f"rubric criterion {c['id']!r} needs a non-empty 'description'")
        out.append((c["id"], desc))
    if len(out) > 8:
        raise HelperError("extended rubric has more than 8 criteria")
    return out


def build_prompt(question, panelists, criteria):
    panel_answers = "\n\n".join(
        "- id: {pid} (strategy_label: {label})\n  answer ({kind}): {answer}".format(
            pid=item["id"], label=item["strategy_label"], kind=item["answer_kind"],
            answer=item["answer"])
        for item in panelists)
    criteria_block = "\n".join(f"- {cid}: {desc}" for cid, desc in criteria)
    skeleton = {
        "assessor_role": "judge-model",
        "assessments": [
            {"panelist_id": item["id"],
             "scores": {cid: 0 for cid, _ in criteria},
             "reasons": {cid: "..." for cid, _ in criteria},
             "strengths": ["..."], "weaknesses": ["..."]}
            for item in panelists
        ],
    }
    return PROMPT_TEMPLATE.format(
        prompt_version=PROMPT_VERSION, question=question,
        panel_answers=panel_answers, criteria_block=criteria_block,
        anchors=ANCHORS, skeleton=json.dumps(skeleton, indent=2))


def _strip_model_output(text):
    if not isinstance(text, str):
        raise HelperError("model response was not text")
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_newline = cleaned.find("\n")
        if first_newline == -1:
            raise HelperError("model returned an unclosed markdown fence")
        cleaned = cleaned[first_newline + 1:]
        if cleaned.rstrip().endswith("```"):
            cleaned = cleaned.rstrip()[:-len("```")].rstrip()
    else:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start:end + 1]
    return cleaned.strip()


def validate_model_output(text, expected_ids, criteria_ids):
    """Validate one model response. Raises HelperError with a CATEGORY
    message only (never response content) so retry summaries stay safe."""
    cleaned = _strip_model_output(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        raise HelperError("malformed output: not valid JSON")
    if not isinstance(data, dict) or not isinstance(data.get("assessments"), list):
        raise HelperError("malformed output: missing 'assessments' list")
    found = {}
    for entry in data["assessments"]:
        if not isinstance(entry, dict):
            raise HelperError("malformed output: assessment entry is not an object")
        id_values = {k: entry[k] for k in ("panelist_id", "panelist", "id") if k in entry}
        if not id_values:
            raise HelperError("malformed output: assessment entry has no panelist identifier")
        if len(set(id_values.values())) > 1:
            raise HelperError("malformed output: differing identifier keys in one entry")
        pid = next(iter(id_values.values()))
        if pid not in expected_ids:
            raise HelperError(f"malformed output: unknown panelist {pid!r}")
        if pid in found:
            raise HelperError(f"malformed output: duplicate entry for panelist {pid!r}")
        scores, reasons = entry.get("scores"), entry.get("reasons")
        if not isinstance(scores, dict) or not isinstance(reasons, dict):
            raise HelperError(f"malformed output: panelist {pid!r} needs 'scores' and 'reasons' objects")
        for cid in criteria_ids:
            value = scores.get(cid)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 5:
                raise HelperError(f"malformed output: panelist {pid!r} score for {cid!r} is not an integer 0-5")
            reason = reasons.get(cid)
            if not isinstance(reason, str) or not reason.strip():
                raise HelperError(f"malformed output: panelist {pid!r} reason for {cid!r} is empty")
        for key in ("strengths", "weaknesses"):
            if key in entry and entry[key] is not None:
                if not isinstance(entry[key], list) or not all(isinstance(x, str) for x in entry[key]):
                    raise HelperError(f"malformed output: panelist {pid!r} {key!r} is not an array of strings")
        found[pid] = {"panelist_id": pid,
                      "scores": {cid: scores[cid] for cid in criteria_ids},
                      "reasons": {cid: reasons[cid].strip() for cid in criteria_ids},
                      "strengths": entry.get("strengths") or [],
                      "weaknesses": entry.get("weaknesses") or []}
    missing = [pid for pid in expected_ids if pid not in found]
    if missing:
        raise HelperError(f"malformed output: missing panelists {missing}")
    return [found[pid] for pid in expected_ids]


def validate_partial_file(path, panelists, criteria_ids, model):
    """--resume-from (P0-2): the whole partial file is rejected outright
    if any entry fails, and its envelope must match this run. Returns
    the list of valid entries (possibly empty) on success."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise HelperError(f"resume file {path!r} is not readable JSON: {exc}")
    if not isinstance(data, dict):
        raise HelperError("resume file must contain a JSON object")
    if data.get("assessor_model") != model:
        raise HelperError("resume file assessor_model does not match this run's --model; refusing to merge")
    if data.get("prompt_version") != PROMPT_VERSION:
        raise HelperError("resume file prompt_version does not match this helper's prompt version; refusing to merge")
    if not isinstance(data.get("assessments"), list):
        raise HelperError("resume file must contain an 'assessments' list")
    panel_ids = [item["id"] for item in panelists]
    # Reuse validate_model_output's per-entry rules by validating the
    # partial entries as a set: every entry must be complete and valid
    # for a panelist in the current panel; unknown/duplicate/bad entries
    # reject the WHOLE file (no entry is reused from a corrupt partial).
    seen = set()
    valid = []
    for entry in data["assessments"]:
        if not isinstance(entry, dict):
            raise HelperError("resume file: an assessment entry is not a JSON object")
        id_values = {k: entry[k] for k in ("panelist_id", "panelist", "id") if k in entry}
        if not id_values or len(set(id_values.values())) > 1:
            raise HelperError("resume file: an entry has no usable panelist identifier")
        pid = next(iter(id_values.values()))
        if pid not in panel_ids:
            raise HelperError(f"resume file: unknown panelist {pid!r}; whole file rejected")
        if pid in seen:
            raise HelperError(f"resume file: duplicate entry for panelist {pid!r}; whole file rejected")
        seen.add(pid)
        scores, reasons = entry.get("scores"), entry.get("reasons")
        if not isinstance(scores, dict) or not isinstance(reasons, dict):
            raise HelperError(f"resume file: panelist {pid!r} needs 'scores' and 'reasons' objects")
        for cid in criteria_ids:
            value = scores.get(cid)
            if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 5:
                raise HelperError(f"resume file: panelist {pid!r} score for {cid!r} is not an integer 0-5; whole file rejected")
            reason = reasons.get(cid)
            if not isinstance(reason, str) or not reason.strip():
                raise HelperError(f"resume file: panelist {pid!r} reason for {cid!r} is empty; whole file rejected")
        for key in ("strengths", "weaknesses"):
            if key in entry and entry[key] is not None:
                if not isinstance(entry[key], list) or not all(isinstance(x, str) for x in entry[key]):
                    raise HelperError(f"resume file: panelist {pid!r} {key!r} is not an array of strings; whole file rejected")
        valid.append({"panelist_id": pid,
                      "scores": {cid: scores[cid] for cid in criteria_ids},
                      "reasons": {cid: reasons[cid].strip() for cid in criteria_ids},
                      "strengths": entry.get("strengths") or [],
                      "weaknesses": entry.get("weaknesses") or []})
    return valid


class _NoCrossHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        old_host = urllib.parse.urlparse(req.full_url).netloc
        new_host = urllib.parse.urlparse(newurl).netloc
        if new_host != old_host:
            raise HelperError("refusing cross-host redirect while carrying the API key")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def call_openrouter(base_url, model, prompt, api_key, timeout, relay):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model,
                          "messages": [{"role": "system",
                                        "content": "You are an independent judge scoring panel answers against a fixed rubric."},
                                       {"role": "user", "content": prompt}],
                          "temperature": 0}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if not relay:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=payload, headers=headers, method="POST")
    opener = urllib.request.build_opener(_NoCrossHostRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HelperError:
        raise
    except urllib.error.HTTPError as exc:
        raise HelperError(f"http error {exc.code}")
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, TimeoutError) or "timed out" in str(reason).lower():
            raise HelperError("request timed out")
        raise HelperError(f"cannot reach OpenRouter at {url}: {reason}")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HelperError(f"OpenRouter request failed: {exc}")
    if not isinstance(body, dict) or not isinstance(body.get("choices"), list) or not body["choices"]:
        raise HelperError("OpenRouter response has no choices")
    message = body["choices"][0].get("message") or {}
    content = message.get("content")
    if not isinstance(content, str):
        raise HelperError("OpenRouter response has no text content")
    return content, body.get("usage") if isinstance(body.get("usage"), dict) else None


def _usage_envelope(usage):
    """Provider-reported usage only. Returns the §1.3 usage object."""
    if not isinstance(usage, dict):
        return None
    out = {}
    for tok in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(tok)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            out[tok] = value
    cost = usage.get("cost")
    if isinstance(cost, (int, float)) and not isinstance(cost, bool) and cost >= 0:
        out["cost_amount"] = cost
        out["cost_currency"] = "USD"
        out["pricing_source"] = "provider-response"
    else:
        out["cost_amount"] = None
        out["cost_currency"] = None
        out["pricing_source"] = None
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Ask an OpenRouter model to score panel answers and write a panel-judge assessments file.")
    parser.add_argument("--panel", required=True, help="input panel JSON file")
    parser.add_argument("--model", required=True, help="OpenRouter model ID, recorded verbatim as assessor_model")
    parser.add_argument("--rubric", default=None, help="optional rubric JSON file (extended criteria are included in the prompt)")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenRouter API base URL (default: %(default)s)")
    parser.add_argument("--relay", action="store_true",
                        help="loopback relay mode: with a loopback --base-url, no API key is required or sent")
    parser.add_argument("--out", default=None, help="output assessments JSON file (required unless --dry-run)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the prompt that would be sent and exit 0 without calling the provider")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help="per-request timeout in seconds (default: %(default)s)")
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                        help="total retries across malformed-output and HTTP 429/5xx retries (default: %(default)s)")
    parser.add_argument("--resume-from", default=None, help="partial assessments file to resume from (P0-2 rules)")
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be a positive integer")
    if args.max_retries < 0:
        parser.error("--max-retries must be a non-negative integer")
    if not args.dry_run and not args.out:
        parser.error("--out is required unless --dry-run is given")
    return args


def _validate_base_url(base_url, relay):
    parsed = urllib.parse.urlparse(base_url)
    if "@" in parsed.netloc:
        raise HelperError("base URL must not contain credentials (userinfo)")
    host = parsed.hostname or ""
    is_loopback = host in ("127.0.0.1", "localhost", "::1")
    if parsed.scheme == "https":
        return is_loopback
    if parsed.scheme == "http" and is_loopback:
        return is_loopback
    raise HelperError("base URL must use HTTPS, except loopback http://127.0.0.1:* / http://localhost:*")


def main(argv=None):
    args = parse_args(argv)
    try:
        _validate_base_url(args.base_url, args.relay)
        question, panelists = load_panel(args.panel)
        criteria = load_rubric_for_prompt(args.rubric)
        criteria_ids = [cid for cid, _ in criteria]
        resumed = []
        if args.resume_from:
            resumed = validate_partial_file(args.resume_from, panelists, criteria_ids, args.model)
        resumed_ids = {e["panelist_id"] for e in resumed}
        missing = [item for item in panelists if item["id"] not in resumed_ids]
        prompt = build_prompt(question, missing, criteria) if missing else None
        if args.dry_run:
            if prompt is None:
                print("(resume file already complete; no request would be made)")
            else:
                print(prompt, end="")
            return 0
        if prompt is None:
            new_entries, usage = [], None
        else:
            api_key = os.environ.get("OPENROUTER_API_KEY") or ""
            is_loopback = (urllib.parse.urlparse(args.base_url).hostname
                           in ("127.0.0.1", "localhost", "::1"))
            if not args.relay or not is_loopback:
                if not api_key:
                    raise HelperError("OPENROUTER_API_KEY is not set; refusing to make a request "
                                      "(or use --relay with a loopback base URL)")
            retries_used = 0
            new_entries = usage = None
            while True:
                try:
                    content, usage_raw = call_openrouter(
                        args.base_url, args.model, prompt,
                        api_key if not (args.relay and is_loopback) else "",
                        args.timeout, bool(args.relay and is_loopback))
                    new_entries = validate_model_output(
                        content, [item["id"] for item in missing], criteria_ids)
                    usage = _usage_envelope(usage_raw)
                    break
                except HelperError as exc:
                    message = str(exc)
                    is_http_retryable = message.startswith("http error 429") or message.startswith("http error 5")
                    is_malformed = message.startswith("malformed output")
                    if (is_http_retryable or is_malformed) and retries_used < args.max_retries:
                        retries_used += 1
                        kind = "HTTP retryable error" if is_http_retryable else "malformed output"
                        print(f"retry {retries_used}/{args.max_retries}: {kind}; retrying the same request",
                              file=sys.stderr)
                        if is_http_retryable:
                            time.sleep(2)
                        continue
                    raise
        merged = {e["panelist_id"]: e for e in resumed}
        for e in (new_entries or []):
            merged[e["panelist_id"]] = e
        ordered = [merged[item["id"]] for item in panelists]
        envelope = {
            "assessor": args.model,
            "assessor_role": "judge-model",
            "assessor_model": args.model,
            "assessor_provider": "openrouter",
            "prompt_version": PROMPT_VERSION,
            "prompt_sha256": prompt_sha256(),
            "temperature": 0,
            "usage": usage,
            "assessments": ordered,
        }
        out_path = args.out
        tmp = out_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(envelope, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, out_path)
        for item in panelists:
            judged = "resumed" if item["id"] in resumed_ids else "judged"
            print(f"{item['id']}: {judged}")
        print(f"Assessments: {out_path}")
        return 0
    except HelperError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
