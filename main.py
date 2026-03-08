# main.py
# Runs all analysis phases in order.
# You can also run each phase independently from the src/ folder.

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).parent / "src"

phases = [
    "phase1_data_prep.py",
    "phase2_descriptive.py",
    "phase3_paired_tests.py",
    "phase4_method_comparison.py",
    "phase5_anova.py",
    "phase6_temporal.py",
    "phase7_economic.py",
]

for phase in phases:
    script = SRC / phase
    if not script.exists():
        print(f"[SKIP] {phase} — not yet implemented")
        continue

    print(f"\n{'=' * 60}")
    print(f"Running {phase}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(SRC),   # run from src/ so relative imports work
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {phase} failed. Stopping.")
        sys.exit(result.returncode)

print("\nAll phases complete.")
