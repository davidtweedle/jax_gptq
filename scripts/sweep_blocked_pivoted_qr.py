import argparse
import subprocess
import sys


def _parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _parse_shape_list(value: str) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        rows_str, cols_str = item.lower().split("x", maxsplit=1)
        shapes.append((int(rows_str), int(cols_str)))
    return shapes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shapes",
        default="128x512,256x1024",
        help="Comma-separated matrix shapes, e.g. 128x512,256x1024",
    )
    parser.add_argument(
        "--panel-sizes",
        default="8,16,32,64,128",
        help="Comma-separated panel sizes",
    )
    parser.add_argument("--pivot-mode", choices=("largest", "smallest"), default="largest")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Enable JAX_GPTQ_QR_TIMING=1 for each run",
    )
    parser.add_argument(
        "--fused-timing",
        action="store_true",
        help="Enable JAX_GPTQ_QR_FUSED_TIMING=1 for each run",
    )
    parser.add_argument(
        "--kernel-debug",
        action="store_true",
        help="Enable JAX_GPTQ_TPU_KERNEL_DEBUG=1 for each run",
    )
    args = parser.parse_args()

    shapes = _parse_shape_list(args.shapes)
    panel_sizes = _parse_int_list(args.panel_sizes)

    for rows, cols in shapes:
        for panel_size in panel_sizes:
            print(f"=== shape={rows}x{cols} panel_size={panel_size} pivot_mode={args.pivot_mode} ===")
            cmd = [
                sys.executable,
                "scripts/bench_blocked_pivoted_qr.py",
                "--rows",
                str(rows),
                "--cols",
                str(cols),
                "--panel-size",
                str(panel_size),
                "--pivot-mode",
                args.pivot_mode,
                "--seed",
                str(args.seed),
            ]

            env = None
            if args.timing or args.fused_timing or args.kernel_debug:
                import os

                env = os.environ.copy()
                if args.timing:
                    env["JAX_GPTQ_QR_TIMING"] = "1"
                if args.fused_timing:
                    env["JAX_GPTQ_QR_FUSED_TIMING"] = "1"
                if args.kernel_debug:
                    env["JAX_GPTQ_TPU_KERNEL_DEBUG"] = "1"

            subprocess.run(cmd, check=True, env=env)
            print()


if __name__ == "__main__":
    main()
