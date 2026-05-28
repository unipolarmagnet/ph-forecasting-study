"""Subprocess worker for async /pipeline/run jobs.

Reads one JSON body from stdin, runs `serve._run_pipeline_logic`, writes JSON result
to stdout and exits. serve.py spawns one of these per async job so that
POST /pipeline/<id>/kill can hard-terminate the OS process (which releases CUDA
memory and avoids torch-thread surgery).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "models"))


def main():
    body = json.loads(sys.stdin.read())
    # Lazy-import after sys.path is set up — pulls in MODELS, _cfg, _run_pipeline_logic.
    from serve import _run_pipeline_logic     # noqa: E402
    try:
        result = _run_pipeline_logic(body)
    except Exception as e:
        result = dict(error=f"{type(e).__name__}: {str(e)[:300]}")
    sys.stdout.write(json.dumps(result, default=str))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
