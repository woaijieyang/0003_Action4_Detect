#!/usr/bin/env python3
"""Project-local resource-admitted task dispatcher; JSON specs contain argv arrays."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path.cwd()))
    parser.add_argument("--queue", default="outputs/action4_dispatch")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("enqueue", "profile"):
        command = commands.add_parser(name)
        command.add_argument("--spec", required=True)
    run = commands.add_parser("run")
    run.add_argument("--poll-seconds", type=float, default=1.)
    run.add_argument("--max-seconds", type=float)
    commands.add_parser("status")
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--job-id", required=True)
    worker.add_argument("--token", required=True)
    args = parser.parse_args()
    from model.training.action4_dispatch import Dispatcher, bounded, read
    dispatcher = Dispatcher(args.root, args.queue)
    if args.command in ("enqueue", "profile"):
        result = dispatcher.enqueue(read(bounded(dispatcher.root, args.spec, exists=True)), profile=args.command == "profile")
    elif args.command == "run":
        result = dispatcher.run(poll_seconds=args.poll_seconds, max_seconds=args.max_seconds)
    elif args.command == "status":
        result = dispatcher.status()
    else:
        return dispatcher.worker(args.job_id, args.token)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
