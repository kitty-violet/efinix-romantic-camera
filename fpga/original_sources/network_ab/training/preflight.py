"""Read-only source/style audit with reports inside the authorized workspace."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone

from .data import audit_incoming, audit_style
from .runtime import configure_workspace, contract_guard, fingerprints, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root")
    args = parser.parse_args()
    root = configure_workspace(args.root)
    _, plan = contract_guard()
    style = audit_style()
    incoming = {name: audit_incoming(root / "data/incoming" / name) for name in ("content", "holdout")}
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
                          "--format=csv"], capture_output=True, text=True, check=True).stdout.strip()
    report = {"time_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
              "gpu": gpu, "model": {k: plan[k] for k in ("model_id", "contract_sha256", "macs_per_frame",
                            "trainable_parameter_count", "quantized_operator_count")},
              "fingerprints": fingerprints(), "style_images": style, "incoming": incoming,
              "content_state": "REQUIRES_AUDIT" if incoming["content"] else "AWAITING_UPLOAD",
              "formal_training_allowed": False}
    write_json(root / "runs/preflight.json", report)
    print(json.dumps({"report": str(root / "runs/preflight.json"), "style_count": len(style),
                      "content_state": report["content_state"], "model": report["model"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
