# /// script
# requires-python = ">=3.11"
# dependencies = ["trackio>=0.32"]
# ///
"""Log one results file (starter baseline or per-claim harm run) to trackio, so every experiment
in docs/preregistration.md is recorded with its exact config next to its metrics.

  python track.py results/harm/E1.json                      # harm run; config from its "run" block
  python track.py results/baseline/E0.json --name E0 \\
      --backfill-commit 12952c2 --judge deepseek/deepseek-v4-flash-0731

Files written before score.py recorded a "run" block need --backfill-commit (the commit whose
scorer produced them); this raises instead of guessing.

View:  trackio show --project votingfacts-eval
"""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).parent
PROJECT = "votingfacts-eval"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_for(d: dict, args) -> dict:
    if "run" in d:
        cfg = dict(d["run"])
        if args.name:
            cfg["name"] = args.name
        return cfg
    if not args.backfill_commit:
        raise SystemExit("results file has no 'run' block: pass --backfill-commit <sha> (and --judge if absent)")
    judge = d.get("judge_model") or args.judge
    if not judge:
        raise SystemExit("judge model unknown: pass --judge")
    scorer = "starter" if "verdicts" in d else "harm"
    return {
        "name": args.name or "unnamed",
        "scorer": scorer,
        "git_commit": args.backfill_commit,
        "judge_model": judge,
        "temperature": 0,
        "severity_version": d.get("severity_version", "n/a"),
        "sha256": {p.name: sha256(p) for p in (HERE / "data" / "rubric.jsonl", HERE / "data" / "severity.json",
                                              HERE / "results" / "answers.jsonl")},
        "backfilled": True,
    }


def items_table(d: dict):
    import trackio
    if "verdicts" in d:
        cols = ["model", "qid", "risk_tier", "verdict", "error_direction", "source_authority"]
        rows = [[v[c] for c in cols] for v in d["verdicts"]]
    else:
        cols = ["model", "qid", "risk_tier", "scoring_mode", "status", "f1", "recall", "precision", "severe_harm", "n_harms"]
        rows = [[r["model"], r["qid"], r["risk_tier"], r["scoring_mode"], r["status"], r["f1"], r["recall"],
                 r["precision"], r["severe_harm"], len(r["harms"])] for r in d["items"]]
    return trackio.Table(columns=cols, data=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("--name")
    ap.add_argument("--backfill-commit")
    ap.add_argument("--judge")
    ap.add_argument("--project", default=PROJECT)
    args = ap.parse_args()

    import trackio
    path = Path(args.results)
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = config_for(d, args) | {"results_file": path.as_posix(), "results_sha256": sha256(path)}
    trackio.init(project=args.project, name=cfg["name"], config=cfg)
    metrics = {f"{m}/{k}": v for m, s in d["summary"].items() for k, v in s.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    trackio.log(metrics | {"items": items_table(d)})
    trackio.finish()
    print(f"logged {cfg['name']} ({len(metrics)} metrics) to trackio project {args.project}")


if __name__ == "__main__":
    main()
