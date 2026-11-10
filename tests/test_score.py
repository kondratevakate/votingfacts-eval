import json
import sys
from types import SimpleNamespace

import pytest

import harm
import score


CLAIMS = [{"claim_id": "a", "claim": "x", "score": 9, "effect": "suppressive"},
          {"claim_id": "b", "claim": "y", "score": 1, "effect": "suppressive"}]


def payload(statuses):
    return json.dumps({"claims": [{"claim_id": k, "status": v, "evidence": ""} for k, v in statuses.items()],
                       "unlisted_contradictions": [], "source_authority": "no_source"})


def test_parse_judge_ok():
    statuses, _ = score.parse_judge(payload({"a": "full", "b": "not_addressed"}), CLAIMS)
    assert statuses == {"a": "full", "b": "not_addressed"}


def test_parse_judge_missing_claim_raises():
    with pytest.raises(ValueError, match="!= rubric"):
        score.parse_judge(payload({"a": "full"}), CLAIMS)


def test_parse_judge_duplicate_claim_raises():
    raw = json.dumps({"claims": [{"claim_id": "a", "status": "full", "evidence": ""}] * 2,
                      "unlisted_contradictions": [], "source_authority": "no_source"})
    with pytest.raises(ValueError, match="twice"):
        score.parse_judge(raw, CLAIMS)


def test_prompt_hides_claim_scores():
    item = {"prompt": "q?", "answer_text": "ans", "citations": [], "reference_source": "src"}
    p = score.judge_prompt(item, CLAIMS)
    assert '"score"' not in p and "suppressive" not in p


def test_schema_restricts_claim_ids():
    s = score.judge_schema(["a", "b"])
    assert s["properties"]["claims"]["items"]["properties"]["claim_id"]["enum"] == ["a", "b"]


def test_build_inputs_reads_all_cached_answers_as_utf8():
    rows = score.build_inputs()
    assert len(rows) == 30
    assert any("ä" in r["answer_text"] or "ã" in r["answer_text"] for r in rows)


def test_main_end_to_end_with_fake_judge(tmp_path, monkeypatch):
    """Whole pipeline offline: a fake judge marks every claim not_addressed for one model and
    returns malformed output for the other, which must surface as ERROR rows and a non-zero exit."""
    rubric = harm.validate_rubric(harm.load_jsonl(score.RUBRIC), {q["qid"] for q in harm.load_jsonl(score.QUESTIONS)})

    class FakeCompletions:
        def create(self, model, temperature, messages, response_format):
            text = messages[0]["content"]
            qid = next(q for q in rubric if any(c["claim"] in text for c in rubric[q]))
            ids = [c["claim_id"] for c in rubric[qid]]
            content = "{}" if "gpt-5-mini" in self.current else payload({i: "not_addressed" for i in ids})
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])

    fake = FakeCompletions()
    real_judge = score.judge_one

    def judge_one(client, model, item, claims):
        fake.current = item["model"]
        return real_judge(client, model, item, claims)

    monkeypatch.setattr(score, "make_client", lambda: (SimpleNamespace(chat=SimpleNamespace(completions=fake)), "fake"))
    monkeypatch.setattr(score, "judge_one", judge_one)
    out = tmp_path / "scores.json"
    monkeypatch.setattr(sys, "argv", ["score.py", "--out", str(out)])
    with pytest.raises(SystemExit, match="15 judge error"):
        score.main()

    res = json.loads(out.read_text(encoding="utf-8"))
    ok, bad = res["summary"]["gpt-5.4"], res["summary"]["gpt-5-mini"]
    assert ok["n_scored"] == 12 and ok["n_controls_scored"] == 3 and ok["n_judge_errors"] == 0
    assert ok["mean_f1"] == 0 and ok["severe_harm_rate"] > 0
    assert bad["n_judge_errors"] == 15 and bad["mean_f1"] is None
