import argparse

from .launcher import launch


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m minidist.run",
        description="Launch one training process per local worker.",
    )
    parser.add_argument(
        "--nproc-per-node",
        "--nproc_per_node",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--master-addr",
        "--master_addr",
        default="127.0.0.1",
    )
    parser.add_argument(
        "--master-port",
        "--master_port",
        type=int,
        default=29500,
    )
    parser.add_argument("--run-id", "--run_id")
    parser.add_argument("training_script")
    parser.add_argument("training_script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    return launch(
        nproc_per_node=args.nproc_per_node,
        script=args.training_script,
        script_args=args.training_script_args,
        master_addr=args.master_addr,
        master_port=args.master_port,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    raise SystemExit(main())
