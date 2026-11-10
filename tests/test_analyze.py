import pytest

import analyze


def test_kappa_perfect_and_chance():
    assert analyze.weighted_kappa([0, 1, 2, 2], [0, 1, 2, 2]) == 1.0
    assert analyze.weighted_kappa([0, 0, 2, 2], [2, 2, 0, 0]) < 0


def test_kappa_partial_disagreement_is_weighted():
    near = analyze.weighted_kappa([0, 1, 2, 2, 0], [0, 2, 2, 2, 0])  # one off by 1
    far = analyze.weighted_kappa([0, 1, 2, 2, 0], [2, 1, 2, 2, 0])   # one off by 2
    assert near > far


def test_kappa_rejects_bad_input():
    with pytest.raises(ValueError):
        analyze.weighted_kappa([], [])


def test_clopper_pearson_known_values():
    assert analyze.clopper_pearson(0, 10) == (0.0, pytest.approx(0.308, abs=1e-3))
    assert analyze.clopper_pearson(10, 10)[1] == 1.0
    lo, hi = analyze.clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187, abs=1e-3) and hi == pytest.approx(0.813, abs=1e-3)


def _harm_item(model, qid, severe, harms, statuses, mode="ground_truth"):
    return {"model": model, "qid": qid, "status": "ok", "scoring_mode": mode, "severe_harm": severe,
            "harms": harms, "claim_status": statuses}


def test_h1_h2_h4_on_toy_runs():
    e0 = {"verdicts": [{"model": "gpt-5.4", "qid": "a", "verdict": "correct"},
                       {"model": "gpt-5-mini", "qid": "a", "verdict": "incorrect"},
                       {"model": "gpt-5.4", "qid": "b", "verdict": "correct"},
                       {"model": "gpt-5-mini", "qid": "b", "verdict": "correct"}]}
    sev_om = [{"type": "omission", "tier": "severe"}]
    sev_com = [{"type": "commission", "tier": "severe"}]
    e1 = {"items": [_harm_item("gpt-5.4", "a", True, sev_om, {"c": "not_addressed"}),
                    _harm_item("gpt-5-mini", "a", True, sev_com, {"c": "full"}),
                    _harm_item("gpt-5.4", "b", True, sev_om, {"c": "not_addressed"}),
                    _harm_item("gpt-5-mini", "b", False, [], {"c": "full"})]}
    r1 = analyze.h1(e0, e1)
    assert r1["severe_harm_answers"] == 3 and r1["starter_incorrect"] == 1
    assert r1["sensitivity"] == pytest.approx(0.333, abs=1e-3)
    r2 = analyze.h2(e1)
    assert r2["omission_only"] == 2 and r2["share"] == pytest.approx(0.667, abs=1e-3)
    r4 = analyze.h4(e1)
    assert r4["discordant_5.4_only"] == 1 and r4["discordant_mini_only"] == 0


def test_h3_reports_not_run_when_missing():
    e1 = {"items": [_harm_item("m", "a", False, [], {"c": "full"})]}
    out = analyze.h3(e1, [], None)
    assert out == {"a_test_retest": "not_run", "b_cross_judge": "not_run"}
