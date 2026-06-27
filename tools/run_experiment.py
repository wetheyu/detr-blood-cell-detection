import argparse
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def bool_flag_name(name):
    return "--" + name


def maybe_resume_from_run_checkpoint(args, auto_resume=True):
    if not auto_resume:
        return None

    output_dir = args.get("output_dir")
    if not output_dir:
        return None

    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    checkpoint_path = output_path / "checkpoint.pth"
    if checkpoint_path.exists():
        args["resume"] = str(checkpoint_path)
        return checkpoint_path
    return None


def build_command(config, auto_resume=True):
    args = dict(config.get("args", {}))
    resumed_from = maybe_resume_from_run_checkpoint(args, auto_resume=auto_resume)
    command = [sys.executable, "main.py"]
    for key, value in args.items():
        if isinstance(value, bool):
            if value:
                command.append(bool_flag_name(key))
            continue
        command.extend([bool_flag_name(key), str(value)])
    return command, resumed_from


def main():
    parser = argparse.ArgumentParser(description="Run a DETR experiment from a JSON config.")
    parser.add_argument("config", type=Path, help="Path to an experiment JSON file.")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running it.")
    parser.add_argument("--fresh", action="store_true",
                        help="Ignore an existing run checkpoint and start from the config resume path.")
    args = parser.parse_args()

    config_path = args.config
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    command, resumed_from = build_command(config, auto_resume=not args.fresh)
    print("Experiment:", config.get("name", config_path.stem))
    print("Description:", config.get("description", ""))
    if resumed_from is not None:
        print("Auto resume:", resumed_from)
    print("Command:", " ".join(command))
    if args.dry_run:
        return

    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
