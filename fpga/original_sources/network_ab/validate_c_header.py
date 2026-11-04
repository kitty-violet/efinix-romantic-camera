"""Compile and execute the generated static C table; no training imports."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent

def record(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    compiler = args.compiler.resolve(strict=True)
    header = HERE / "generated/model_b_v1/model_b_contract.h"
    deploy = HERE / "generated/model_b_v1/deployment_static.json"
    probe = HERE / "tests/contract_header_probe.c"
    inputs = [record(p) for p in (header, deploy, probe, compiler, Path(__file__))]
    shutil.copyfile(header, output / header.name)
    shutil.copyfile(probe, output / probe.name)
    env = dict(os.environ)
    env["PATH"] = str(compiler.parent) + os.pathsep + env.get("PATH", "")
    for name in ("TEMP", "TMP", "TMPDIR"):
        env[name] = str(output)
    exe = output / "contract_header_probe.exe"
    command = [str(compiler), "-std=c11", "-Wall", "-Wextra", "-Werror",
               "-I", str(output), str(output / probe.name), "-o", str(exe)]
    completed = subprocess.run(command, cwd=output, env=env, capture_output=True)
    (output / "compile.log").write_bytes(completed.stdout + completed.stderr)
    if completed.returncode or completed.stdout or completed.stderr:
        raise RuntimeError("C header compile must succeed without diagnostics")
    run = subprocess.run([str(exe)], cwd=output, env=env, capture_output=True)
    (output / "actual.txt").write_bytes(run.stdout)
    if run.returncode or run.stderr:
        raise RuntimeError("C table execution failed")
    config = json.loads(deploy.read_text(encoding="utf-8"))
    expected = [f"CONTRACT {config['contract_sha256']} START 0"]
    kinds = {"conv3x3": 1, "dw3x3": 2, "pw1x1": 3}
    for op in config["operators"]:
        scalar = [op["layer_id"], kinds[op["kind"]]] + [
            op[key] for key in ("cin", "cout", "terms", "kernel", "stride", "oc8_groups")]
        masks = op["relu_masks"] + [0] * (4 - op["oc8_groups"])
        masks += op["lane_masks"] + [0] * (4 - op["oc8_groups"])
        expected.append(" ".join(map(str, scalar + masks)))
    if run.stdout.decode("ascii").splitlines() != expected:
        raise RuntimeError("compiled C table differs from deployment JSON")
    if inputs != [record(p) for p in (header, deploy, probe, compiler, Path(__file__))]:
        raise RuntimeError("input changed during validation")
    report = {"status": "PASS_HOST_C_HEADER_AND_TABLE", "operator_count": 24,
              "start_allowed": False, "pytorch_executed": False,
              "sapphire_firmware_validated": False, "command": command, "inputs": inputs,
              "outputs": [record(p) for p in (exe, output / "compile.log", output / "actual.txt")]}
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("CONTRACT_C_HEADER_PASS operators=24 warnings=0 start_allowed=false host_only=true")

if __name__ == "__main__":
    main()
