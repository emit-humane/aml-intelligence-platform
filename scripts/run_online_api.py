"""
Launch the AML Intelligence Platform API server.

Usage:
    python -m scripts.run_online_api
    python -m scripts.run_online_api --port 8000 --reload
"""

from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Start AML API server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", default=False)
    args = parser.parse_args()

    cmd = [
        sys.executable, "-m", "uvicorn",
        "system2_platform.api.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    print(f"[run_online_api] Starting: {' '.join(cmd)}")
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
