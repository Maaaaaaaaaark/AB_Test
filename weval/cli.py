"""weval command line interface.

Local development machine: data-prep, build, review, worked-examples, and
scripted dry-runs of run/report/estimate (no torch, no GPU).

Cluster: precheck, run --backend transformers, inspect-params, report,
estimate - all GPU generation happens there, never here.

No command ever prints or stores HF tokens.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ConfigBundle, load_configs


def _cfg(root: Path) -> ConfigBundle:
    return load_configs(root)


def _out_dir(root: Path) -> Path:
    return root / "artifacts"


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------

def cmd_precheck(args: argparse.Namespace) -> int:
    from .precheck import print_precheck, run_precheck

    cfg = _cfg(args.root)
    rep = run_precheck(cfg, _out_dir(args.root) / "precheck")
    print_precheck(rep)
    return 0


def cmd_data_prep(args: argparse.Namespace) -> int:
    from .data_prep import (
        prepare_hf_dataset,
        prepare_musique,
        verify_gsm8k,
        verify_hotpotqa,
        verify_math,
        verify_processbench,
    )

    cfg = _cfg(args.root)
    data_dir = _out_dir(args.root) / "data"
    zip_path = args.musique_zip
    if not zip_path.exists():
        print(f"ERROR: MuSiQue zip not found at {zip_path}", file=sys.stderr)
        print("  download it first:", file=sys.stderr)
        print("  python -m gdown --output musique_v1.0.zip "
              "https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h", file=sys.stderr)
        return 2
    if not args.skip_musique:
        prov = prepare_musique(
            cfg, data_dir, zip_path,
            "https://drive.google.com/uc?id=1tGdADlNjWFaHLeZZGShh2IRcpO6Lv24h",
            "cc-by-4.0 (MuSiQue authors' release)",
        )
        print(f"musique: {prov.n_records} records; {prov.notes}")
    if not args.skip_hf:
        from .datasets import gsm8k, hotpotqa, mathbench, processbench

        jobs = [
            ("processbench", processbench, "apache-2.0 (QwenLM/ProcessBench)",
             verify_processbench, "https://huggingface.co/datasets/QwenLM/ProcessBench"),
            ("gsm8k", gsm8k, "mit (openai/gsm8k)",
             verify_gsm8k, "https://huggingface.co/datasets/openai/gsm8k"),
            ("hotpotqa", hotpotqa, "cc-by-sa-4.0 (hotpotqa/hotpot_qa)",
             verify_hotpotqa, "https://huggingface.co/datasets/hotpotqa/hotpot_qa"),
            ("math", mathbench, "mit (hendrycks/competition_math mirror; see UNRESOLVED.md)",
             verify_math, "https://huggingface.co/datasets/hendrycks/competition_math"),
        ]
        for name, adapter, license_, verify, url in jobs:
            print(f"{name}: downloading + verifying ...")
            prov = prepare_hf_dataset(cfg, data_dir, adapter, name, license_, verify, url)
            print(f"{name}: {prov.n_records} records; claims: {prov.verified_claims}")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    from .data_prep import load_provenance
    from .datasets.build import build_all

    cfg = _cfg(args.root)
    out_dir = _out_dir(args.root)
    prov = load_provenance(out_dir / "data")
    res = build_all(cfg, out_dir / "data", out_dir, prov)
    print("manifests:")
    for k, v in sorted(res.manifests.items()):
        print(f"  {k}: {v}")
    if res.review_file:
        print(f"review file: {res.review_file}")
    for k, v in sorted(res.blocked.items()):
        print(f"BLOCKED {k}: {v}")
    if res.blocked:
        return 1
    return 0


def _backend_factory(args: argparse.Namespace, cfg: ConfigBundle):
    if args.backend == "scripted":
        from .models.scripted import ScriptedBackend

        script = Path(args.script) if args.script else None
        if args.default_text:
            return lambda name: ScriptedBackend(
                script_path=script,
                default_text=args.default_text,
                reasoning_mode=cfg.model(name).reasoning_mode,
            )
        return lambda name: ScriptedBackend(
            script_path=script,
            reasoning_mode=cfg.model(name).reasoning_mode,
        )
    if args.backend == "transformers":
        from .models.transformers_backend import TransformersBackend

        snapshots = Path(args.snapshots)
        if not snapshots.exists():
            raise SystemExit(f"ERROR: --snapshots dir {snapshots} does not exist")

        def factory(name: str):
            m = cfg.model(name)
            snap = snapshots / name
            if not snap.exists():
                raise SystemExit(f"ERROR: no snapshot dir {snap} for config {name}")
            return TransformersBackend(snap, m, cfg.root, device=args.device)

        return factory
    raise SystemExit(f"unknown backend {args.backend!r}")


def cmd_run(args: argparse.Namespace) -> int:
    from .run.runner import PlanGateError, run_plan

    cfg = _cfg(args.root)
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    backend_factory = _backend_factory(args, cfg)
    try:
        summary = run_plan(
            cfg, args.plan, backend_factory, _out_dir(args.root),
            seeds=seeds, resume=not args.no_resume,
        )
    except PlanGateError as e:
        print(f"plan gating blocked: {e}", file=sys.stderr)
        return 2
    print("run summary:")
    for k, v in sorted(summary.counters.items()):
        print(f"  {k}: {v}")
    if summary.unresolved:
        print(f"UNRESOLVED infrastructure failures: {len(summary.unresolved)}")
        for u in summary.unresolved[:20]:
            print(f"  {u['config']}/{u['task']}/s{u['seed']} {u['item_id']}: "
                  f"{u.get('kind')} after {len(u.get('attempts', []))} attempts")
        if summary.gating.get("report_incomplete_on_unresolved", True):
            return 1
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .metrics.report import build_report, write_report
    from .metrics.report_md import write_report_md

    cfg = _cfg(args.root)
    report = build_report(cfg, args.plan, _out_dir(args.root), n_boot=args.n_boot)
    jp = write_report(_out_dir(args.root), args.plan, report)
    mp = write_report_md(_out_dir(args.root), args.plan, report)
    print(f"report written: {jp}")
    print(f"markdown:      {mp}")
    print(f"thresholds status: {report.thresholds_status} "
          f"(recommendation {'gated' if report.thresholds_status != 'confirmed' else 'active'})")
    return 0


def cmd_worked_examples(args: argparse.Namespace) -> int:
    from .worked_examples import write_worked_examples

    cfg = _cfg(args.root)
    p = write_worked_examples(cfg, _out_dir(args.root), args.root / "docs", args.plan)
    print(f"written: {p}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from .review import ReviewError, apply_decision, print_summary

    path = _out_dir(args.root) / "manifests" / "musique_negative_review.json"
    if args.action == "list":
        print_summary(path, args.only)
        return 0
    try:
        entry = apply_decision(path, args.parent, args.action, note=args.note, target=args.target)
    except ReviewError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    if args.target == "positive":
        print(f"{args.parent} [positive]: {entry.get('positive_review_status')} "
              f"(human_reviewed={entry.get('positive_human_reviewed')})")
    else:
        print(f"{args.parent} [negative]: {entry['review_status']} "
              f"(human_reviewed={entry.get('human_reviewed')})")
    return 0


def cmd_inspect_params(args: argparse.Namespace) -> int:
    from .models.params import count_text_parameters

    cfg = _cfg(args.root)
    model_cfg = cfg.model(args.config)
    snap = Path(args.snapshot_dir)
    if not snap.exists():
        print(f"ERROR: snapshot dir {snap} does not exist", file=sys.stderr)
        return 2
    audit = count_text_parameters(
        snap, param_exclusions=model_cfg.param_exclusions, detail=True
    )
    out = audit.to_dict()
    out["config"] = args.config
    out["repo"] = model_cfg.repo
    out["revision"] = model_cfg.revision
    pdir = _out_dir(args.root) / "model_params"
    pdir.mkdir(parents=True, exist_ok=True)
    with open(pdir / f"{args.config}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"n_text = {audit.n_text:,}  (total in files {audit.n_total_in_files:,}, "
          f"excluded params {audit.n_excluded:,} across "
          f"{audit.n_excluded_tensors:,} tensors, tied dedup {audit.tied_dedup_applied})")
    print(f"saved: {pdir / f'{args.config}.json'}")
    return 0


def cmd_estimate(args: argparse.Namespace) -> int:
    from .estimate import estimate, print_estimate

    cfg = _cfg(args.root)
    est = estimate(cfg, _out_dir(args.root), args.source, args.target, n_boot=args.n_boot)
    edir = _out_dir(args.root) / "estimates"
    edir.mkdir(parents=True, exist_ok=True)
    path = edir / f"{args.source}__to_{args.target}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(est, f, indent=2, ensure_ascii=False)
    print_estimate(est)
    print(f"saved: {path}")
    return 0


# --------------------------------------------------------------------------
# argument parser
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="weval",
        description="worker-model evaluation: data prep, gated plans, per-item generation, reports",
    )
    p.add_argument("--root", type=Path, default=Path("."),
                   help="project root containing configs/ and artifacts/ (default: cwd)")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("precheck", help="cluster readiness check (token presence, gated access, template locks, GPU/bf16)")
    sp.set_defaults(func=cmd_precheck)

    sp = sub.add_parser("data-prep", help="unpack/verify MuSiQue + download/verify pinned HF datasets")
    sp.add_argument("--musique-zip", type=Path, default=Path("_recon/musique_v1.0.zip"))
    sp.add_argument("--skip-musique", action="store_true")
    sp.add_argument("--skip-hf", action="store_true")
    sp.set_defaults(func=cmd_data_prep)

    sp = sub.add_parser("build", help="build all manifests + the negative review file")
    sp.set_defaults(func=cmd_build)

    sp = sub.add_parser("run", help="generate predictions for a plan (gated)")
    sp.add_argument("--plan", required=True, choices=["smoke", "pilot", "selection", "holdout", "reference"])
    sp.add_argument("--backend", default="scripted", choices=["scripted", "transformers"])
    sp.add_argument("--snapshots", type=Path, default=Path("models"),
                    help="transformers: dir with one snapshot subdir per config name")
    sp.add_argument("--device", default="auto")
    sp.add_argument("--seeds", default=None, help="comma-separated override (default: plan seeds)")
    sp.add_argument("--script", default=None, help="scripted: JSONL script file (item_id -> text)")
    sp.add_argument("--default-text", default=None, help="scripted: default output text")
    sp.add_argument("--no-resume", action="store_true")
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("report", help="re-parse + re-score predictions and write report.json/report.md")
    sp.add_argument("--plan", required=True, choices=["smoke", "pilot", "selection", "holdout", "reference"])
    sp.add_argument("--n-boot", type=int, default=1000)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("worked-examples", help="render docs/worked_examples.md from real manifest items")
    sp.add_argument("--plan", default="pilot", choices=["smoke", "pilot"])
    sp.set_defaults(func=cmd_worked_examples)

    sp = sub.add_parser("review", help="human review of MuSiQue positives and paired negatives")
    sp.add_argument("action", choices=["list", "approve", "exclude", "reset"])
    sp.add_argument("--parent", default=None, help="parent record id (required for approve/exclude/reset)")
    sp.add_argument("--target", default="negative", choices=["negative", "positive"],
                    help="which item of the parent the decision applies to (default: negative)")
    sp.add_argument("--note", default=None, help="reason (required for exclude; recorded verbatim)")
    sp.add_argument("--only", default=None, choices=["proposed", "approved", "excluded"])
    sp.set_defaults(func=cmd_review)

    sp = sub.add_parser("inspect-params", help="count unique text parameters from a snapshot (safetensors headers only)")
    sp.add_argument("--config", required=True)
    sp.add_argument("--snapshot-dir", required=True)
    sp.set_defaults(func=cmd_inspect_params)

    sp = sub.add_parser("estimate", help="extrapolate pilot/smoke measurements to a target plan")
    sp.add_argument("--source", required=True, choices=["smoke", "pilot"])
    sp.add_argument("--target", required=True, choices=["pilot", "selection", "holdout"])
    sp.add_argument("--n-boot", type=int, default=1000)
    sp.set_defaults(func=cmd_estimate)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
