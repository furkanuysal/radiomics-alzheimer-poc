import subprocess
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INFERENCE_SCRIPT = PROJECT_ROOT / "src" / "inference" / "run_mri_analysis.py"

def analyze_mri(mri_path: str) -> dict:
    """
    Calls run_mri_analysis.py and returns parsed JSON output
    """
    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--mri",
        mri_path
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr)

    return json.loads(result.stdout)
