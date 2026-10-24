import argparse
import json
import sys
from .config import load_config
from .pipeline import preflight, run


def main(argv=None):
    parser = argparse.ArgumentParser(description="CREST reconstructed underwater SfM pipeline")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, description in [("plan", "Show resolved configuration without loading models"),
                              ("doctor", "Check input paths and dependencies"),
                              ("run", "Run into a new directory; never overwrite a prior run")]:
        cmd = sub.add_parser(name, help=description)
        cmd.add_argument("--config", required=True)
        cmd.add_argument("--until", choices=["prepare", "reconstruct"], default="reconstruct")
        if name == "run":
            cmd.add_argument("--run-id")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "plan":
            print(json.dumps({"config": cfg, "until": args.until,
                              "note": "Reconstructed method; defaults do not establish historical parameters."}, indent=2))
        elif args.command == "doctor":
            issues = preflight(cfg, args.until)
            print("\n".join("MISSING: " + issue for issue in issues) if issues else "Ready for requested stages.")
            if issues:
                raise SystemExit(1)
        else:
            print(f"Results: {run(cfg, args.run_id, args.until)}")
    except (ValueError, FileNotFoundError, FileExistsError, ImportError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
