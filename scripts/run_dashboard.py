"""
CAEG-Net Interactive Research Dashboard Launcher
=================================================
Runs the Streamlit dashboard via subprocess with appropriate settings.
"""

import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_PATH = REPO_ROOT / "dashboard" / "app.py"

def main():
    print("=" * 70)
    print("  Launching CAEG-Net Interactive Research Dashboard")
    print(f"  Target: {APP_PATH}")
    print("=" * 70)

    if not APP_PATH.exists():
        print(f"[ERROR] Dashboard entry point not found at {APP_PATH}")
        sys.exit(1)

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.headless=false",
    ]

    try:
        subprocess.run(cmd, cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        print("\nDashboard session stopped.")

if __name__ == "__main__":
    main()
