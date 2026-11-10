# /// script
# requires-python = ">=3.11"
# dependencies = ["scipy>=1.10"]
# ///
"""Test the preregistered hypotheses H1-H4 (docs/preregistration.md) from the experiment files.

  python analyze.py --e0 results/baseline/E0.json --e1 results/harm/E1.json \\
      [--repeats results/harm/E2.json results/harm/E3.json] [--judge2 results/harm/E4.json] \\
      [--out results/analysis/hypotheses.json]

Statistics are exactly those frozen in the preregistration; anything missing (e.g. E4 not run)
is reported as "not_run", never skipped silently.
"""
import argparse
import itertools
import json
from pathlib import Path

from scipy.stats import beta, binomtest

ORDER = {"not_addressed": 0, "partial": 1, "full": 2}


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (round(float(lo), 3), round(float(hi), 3))


def weighted_kappa(a: list[int], b: list[int], k: int = 3) -> float:
    """Linear-weighted Cohen's kappa for two raters on an ordinal scale 0..k-1."""
    n = len(a)
    if n == 0 or n != len(b):
        raise ValueError("kappa needs two equal-length non-empty label lists")
    w = [[abs(i - j) / (k - 1) for j in range(k)] for i in range(k)]
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[x][y] += 1 / n
    pa = [sum(obs[i]) for i in range(k)]
    pb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    do = sum(w[i][j] * obs[i][j] for i in range(k) for j in range(k))
    de = sum(w[i][j] * pa[i] * pb[j] for i in range(k) for j in range(k))
    return 1.0 if de == 0 else round(1 - do / de, 3)


def ok_items(d) -> dict:
    return {(r["model"], r["qid"]): r for r in d["items"] if r["status"] == "ok"}


def claim_pairs(d1, d2):
    a, b = ok_items(d1), ok_items(d2)
    x, y, flags = [], [], []
    for key in sorted(a.keys() & b.keys()):
        for cid, s in a[key]["claim_status"].items():
            x.append(ORDER[s])
            y.append(ORDER[b[key]["claim_status"][cid]])
        if a[key]["scoring_mode"] != "control":
            flags.append(a[key]["severe_harm"] == b[key]["severe_harm"])
    return x, y, flags


def h1(e0, e1):
    starter = {(v["model"], v["qid"]): v["verdict"] for v in e0["verdicts"]}
    severe = [k for k, r in ok_items(e1).items() if r["scoring_mode"] != "control" and r["severe_harm"]]
    caught = sum(starter[k] == "incorrect" for k in severe)
    n = len(severe)
    sens = round(caught / n, 3) if n else None
    ci = clopper_pearson(caught, n)
    return {"severe_harm_answers": n, "starter_incorrect": caught, "sensitivity": sens, "ci95": ci,
            "missed": [list(k) for k in severe if starter[k] != "incorrect"],
            "supported": bool(n) and sens < 0.5 and ci[1] < 1.0}


def h2(e1):
    severe = [r for r in ok_items(e1).values() if r["scoring_mode"] != "control" and r["severe_harm"]]
    omission_only = sum(all(h["type"] == "omission" for h in r["harms"] if h["tier"] == "severe") for r in severe)
    n = len(severe)
    p = binomtest(omission_only, n, 0.5, alternative="greater").pvalue if n else None
    return {"severe_harm_answers": n, "omission_only": omission_only,
            "share": round(omission_only / n, 3) if n else None, "p_one_sided": p and round(p, 4),
            "supported": bool(n) and omission_only / n > 0.5 and p < 0.05}


def h3(e1, repeats, judge2):
    out = {}
    if repeats:
        runs = [e1, *repeats]
        ks, fl = [], []
        for d1, d2 in itertools.combinations(runs, 2):
            x, y, flags = claim_pairs(d1, d2)
            ks.append(weighted_kappa(x, y))
            fl.append(round(sum(flags) / len(flags), 3))
        mk = round(sum(ks) / len(ks), 3)
        out["a_test_retest"] = {"pairwise_kappa": ks, "mean_kappa": mk, "severe_flag_agreement": fl,
                                "supported": mk >= 0.80}
    else:
        out["a_test_retest"] = "not_run"
    if judge2:
        x, y, flags = claim_pairs(e1, judge2)
        k = weighted_kappa(x, y)
        out["b_cross_judge"] = {"kappa": k, "severe_flag_agreement": round(sum(flags) / len(flags), 3),
                                "supported": k >= 0.70}
    else:
        out["b_cross_judge"] = "not_run"
    return out


def h4(e1):
    items = ok_items(e1)
    qids = sorted({q for (_, q), r in items.items() if r["scoring_mode"] != "control"})
    b = c = 0
    for q in qids:
        x, y = items.get(("gpt-5.4", q)), items.get(("gpt-5-mini", q))
        if not x or not y:
            continue
        b += x["severe_harm"] and not y["severe_harm"]
        c += y["severe_harm"] and not x["severe_harm"]
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    rates = {m: round(sum(r["severe_harm"] for (mm, q), r in items.items() if mm == m and r["scoring_mode"] != "control")
                      / max(1, sum(1 for (mm, q), r in items.items() if mm == m and r["scoring_mode"] != "control")), 3)
             for m in ("gpt-5.4", "gpt-5-mini")}
    return {"severe_harm_rate": rates, "discordant_5.4_only": b, "discordant_mini_only": c,
            "p_exact_mcnemar": round(p, 4), "note": "exploratory, not powered"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0", required=True)
    ap.add_argument("--e1", required=True)
    ap.add_argument("--repeats", nargs="*", default=[])
    ap.add_argument("--judge2")
    ap.add_argument("--out")
    ap.add_argument("--trackio", action="store_true", help="also log the results as trackio run 'analysis'")
    args = ap.parse_args()

    e0, e1 = load(args.e0), load(args.e1)
    res = {"inputs": {"e0": args.e0, "e1": args.e1, "repeats": args.repeats, "judge2": args.judge2},
           "H1": h1(e0, e1), "H2": h2(e1),
           "H3": h3(e1, [load(p) for p in args.repeats], args.judge2 and load(args.judge2)),
           "H4": h4(e1)}
    text = json.dumps(res, indent=2)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    if args.trackio:
        import trackio
        trackio.init(project="votingfacts-eval", name="analysis", config=res["inputs"])
        flat = {"H1/sensitivity": res["H1"]["sensitivity"], "H2/omission_share": res["H2"]["share"]}
        for k in ("a_test_retest", "b_cross_judge"):
            v = res["H3"][k]
            if isinstance(v, dict):
                flat[f"H3/{k}/kappa"] = v.get("mean_kappa", v.get("kappa"))
        flat |= {f"H4/severe_rate/{m}": r for m, r in res["H4"]["severe_harm_rate"].items()}
        trackio.log({k: v for k, v in flat.items() if v is not None})
        trackio.finish()


if __name__ == "__main__":
    main()
