"""Check imported assets and local documentation; never run hardware tools."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests/source-assets.json"
CAD = "mechanical/enclosure_v9"
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(value: str) -> Path:
    relative = PurePosixPath(value)
    if not value or relative.is_absolute() or ".." in relative.parts or "\\" in value:
        raise ValueError(f"Unsafe repository path: {value!r}")
    resolved = (ROOT / value).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Path outside repository: {value!r}")
    return resolved


def verify_fpga(indexed: dict[str, dict], errors: list[str]) -> int:
    package = json.loads(repository_path("fpga/MANIFEST.json").read_text(encoding="utf-8"))
    if package["version"] != "production-r15-complete":
        errors.append("Unexpected FPGA package version")
    listed = package["files"]
    expected = {"fpga/MANIFEST.json"}
    for name, digest in listed.items():
        full_name = f"fpga/{name}"
        repository_path(full_name)
        expected.add(full_name)
        if full_name not in indexed or indexed[full_name]["sha256"] != digest:
            errors.append(f"Upstream package manifest mismatch: {full_name}")
    actual = {name for name in indexed if name.startswith("fpga/")}
    if actual != expected:
        errors.append(f"FPGA asset coverage mismatch: {sorted(actual ^ expected)}")
    disk_files = {path.relative_to(ROOT).as_posix()
                  for path in (ROOT / "fpga").rglob("*")
                  if path.is_file() and "build" not in path.relative_to(ROOT / "fpga").parts
                  and not any(part.startswith("work_") for part in path.relative_to(ROOT / "fpga").parts)}
    if not expected.issubset(disk_files):
        errors.append("Some FPGA package files are absent from disk")

    project = repository_path("fpga/project/ti60f225_oob.xml")
    tree = ET.parse(project)
    references = 0
    for element in tree.iter():
        tag = element.tag.split("}")[-1]
        if tag in {"design_file", "sdc_file", "peri_file", "default_peri_file"}:
            target = element.attrib["name"]
            full_name = f"fpga/project/{target}"
            if not repository_path(full_name).is_file() or full_name not in indexed:
                errors.append(f"Missing or unlisted Efinity reference: {target}")
            references += 1
    data = repository_path("fpga/firmware/prebuilt/control.bin").read_bytes()
    if len(data) > 8192:
        errors.append("CPU firmware exceeds the 8 KiB OCR")
        return references
    data += bytes(8192 - len(data))
    for lane in range(4):
        filename = f"EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{lane}.bin"
        expected_bytes = data[lane::4]
        for folder in ("fpga/project", "fpga/firmware/prebuilt"):
            path = repository_path(f"{folder}/{filename}")
            lines = path.read_text(encoding="ascii").splitlines()
            if any(not re.fullmatch(r"[01]{8}", line) for line in lines):
                errors.append(f"Invalid ROM lane data: {folder}/{filename}")
            elif bytes(int(line, 2) for line in lines) != expected_bytes:
                errors.append(f"Firmware/ROM lane mismatch: {folder}/{filename}")
    return references


def verify_network(indexed: dict[str, dict], errors: list[str]) -> None:
    graph = json.loads(repository_path("fpga/network/graph.json").read_text(encoding="utf-8"))
    if graph["schema"] != "S4R1_8_32_8_V1" or graph["input_shape_hwc"] != [720, 1280, 3]:
        errors.append("Unexpected network schema or image dimensions")
    nodes = graph["nodes"]
    convs = [node for node in nodes if "kernel" in node]
    weights = sum(node["channels"] * (node["cin"] // node["groups"]) *
                  node["kernel"] ** 2 for node in convs)
    biases = sum(node["channels"] for node in convs)
    if (len(nodes), len(convs), weights, biases) != (13, 11, 3808, 227):
        errors.append("Network topology or parameter count differs from S4R1")
    known = {"input"}
    for node in nodes:
        if node["id"] in known or not set(node["inputs"]).issubset(known):
            errors.append(f"Invalid network dependency: {node['id']}")
        known.add(node["id"])
    for slot in range(5):
        folder = f"fpga/network/models/slot{slot}"
        for filename in ("selected.pt", "graph.json", "parameter-count.json",
                         "coefficients.bin", "resident-padded.bin", "reference-vectors.json"):
            if f"{folder}/{filename}" not in indexed:
                errors.append(f"Missing model delivery asset: {folder}/{filename}")
        counts = json.loads(repository_path(f"{folder}/parameter-count.json").read_text(encoding="utf-8"))
        if (counts["model_parameters"], counts["additional_q17_multipliers"]) != (4035, 227):
            errors.append(f"Incorrect parameter metadata: slot{slot}")
        style_graph = json.loads(repository_path(f"{folder}/graph.json").read_text(encoding="utf-8"))
        if style_graph["nodes"] != nodes or style_graph["input_shape_hwc"] != [720, 1280, 3]:
            errors.append(f"Style topology differs from shared graph: slot{slot}")
    if repository_path("fpga/network/models/slot5/selected.pt").exists():
        errors.append("Original-image bypass must not be presented as a sixth CNN checkpoint")

    runtime = json.loads(
        repository_path("fpga/network/runtime_data/manifest.json").read_text(encoding="utf-8"))
    image = indexed["fpga/network/runtime_data/flash-user-data.bin"]
    if runtime["flash_offset"] != 0x340000 or runtime["flash_end"] - runtime["flash_offset"] != image["bytes"]:
        errors.append("Runtime data-region address/length mismatch")
    if runtime["image_sha256"] != image["sha256"] or runtime["hardware_access"] is not False:
        errors.append("Runtime image identity or hardware-verification metadata mismatch")
    if [entry["slot"] for entry in runtime["styles"]] != list(range(6)):
        errors.append("Runtime catalog must contain slots 0 through 5")
    for entry in runtime["styles"]:
        if entry["compute_sha256"] != indexed["fpga/network/programs/compute.bin"]["sha256"]:
            errors.append(f"Runtime compute program mismatch: slot{entry['slot']}")
        coefficient = indexed[f"fpga/network/models/slot{entry['slot']}/resident-padded.bin"]
        if entry["coefficient_sha256"] != coefficient["sha256"]:
            errors.append(f"Runtime coefficient identity mismatch: slot{entry['slot']}")

    core = repository_path("fpga/network/rtl/cnn_production_frame.xml")
    design_files = [element.attrib["name"] for element in ET.parse(core).iter()
                    if element.tag.split("}")[-1] == "design_file"]
    if len(design_files) != 42:
        errors.append("Expected 42 readable compute-core design files")
    for name in design_files:
        full_name = f"fpga/network/rtl/{name}"
        if full_name not in indexed or not repository_path(full_name).is_file():
            errors.append(f"Missing readable compute-core source: {name}")


def verify(allow_pointers: bool) -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported manifest schema")
    assets = manifest["assets"]
    errors: list[str] = []
    indexed: dict[str, dict] = {}
    pointer_count = 0
    actual_count = 0
    for asset in assets:
        name = asset["path"]
        if name in indexed:
            errors.append(f"Duplicate manifest entry: {name}")
            continue
        indexed[name] = asset
        path = repository_path(name)
        if not isinstance(asset["bytes"], int) or asset["bytes"] < 0:
            raise ValueError(f"Invalid byte size: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", asset["sha256"]):
            raise ValueError(f"Invalid SHA-256: {name}")
        if not path.is_file():
            errors.append(f"Missing asset: {name}")
            continue
        with path.open("rb") as stream:
            prefix = stream.read(1024)
        if prefix.startswith(LFS_HEADER):
            pointer = re.fullmatch(
                rb"version https://git-lfs.github.com/spec/v1\r?\n"
                rb"oid sha256:([0-9a-f]{64})\r?\nsize ([0-9]+)\r?\n?",
                prefix,
            )
            if not asset["lfs"] or not pointer or path.stat().st_size != len(prefix):
                errors.append(f"Malformed or unexpected LFS pointer: {name}")
            elif pointer[1].decode("ascii") != asset["sha256"] or int(pointer[2]) != asset["bytes"]:
                errors.append(f"LFS pointer differs from manifest: {name}")
            elif not allow_pointers:
                errors.append(f"LFS content not downloaded; run git lfs pull: {name}")
            else:
                pointer_count += 1
        elif path.stat().st_size != asset["bytes"]:
            errors.append(f"File size differs from manifest: {name}")
        elif file_sha256(path) != asset["sha256"]:
            errors.append(f"File SHA-256 differs from manifest: {name}")
        else:
            actual_count += 1

    total_bytes = sum(asset["bytes"] for asset in assets)
    if manifest["asset_count"] != len(assets) or manifest["total_bytes"] != total_bytes:
        errors.append("Manifest totals differ from asset entries")
    fpga_assets = [asset for asset in assets if asset["path"].startswith("fpga/")]
    archive = manifest["fpga_archive"]
    if archive["file_count"] != len(fpga_assets) or archive["total_bytes"] != sum(
        asset["bytes"] for asset in fpga_assets
    ):
        errors.append("FPGA archive totals differ from imported assets")
    if not re.fullmatch(r"[0-9a-f]{64}", archive["sha256"]):
        errors.append("Invalid source archive SHA-256")
    project_references = verify_fpga(indexed, errors)
    verify_network(indexed, errors)

    required = [
        "README.md", "CHANGELOG.md", "CONTRIBUTING.md", "THIRD_PARTY_NOTICES.md",
        ".gitattributes", ".gitignore", ".github/workflows/verify.yml",
        "fpga/README.md", "fpga/MANIFEST.json", "fpga/project/native222.v",
        "fpga/project/ble_uart_buttons.sv", "fpga/project/legacy222_names.sv",
        "fpga/project/five_style_name_osd.sv", "fpga/project/ti60f225_oob.xml",
        "fpga/firmware/main.c", "fpga/tools/verify_package.py", f"{CAD}/README.md",
        "fpga/tools/verify_complete.py", "fpga/SOURCE_ORIGINS.json",
        "fpga/network/model.py", "fpga/network/infer_package.py",
        "fpga/network/models/slot5/identity.json", "fpga/network/rtl/cnn_production_frame.xml",
        "fpga/network/runtime_data/flash-user-data.bin",
        "docs/environment.md", "docs/usage.md", "docs/ble-protocol.md",
        "docs/competition-checklist.md", "docs/submission.md", "docs/verification.md",
        "docs/fpga-project.md", "docs/network.md",
    ]
    for name in required:
        if not repository_path(name).is_file():
            errors.append(f"Missing entry point: {name}")

    link_count = 0
    markdown_files = sorted(
        path for path in ROOT.rglob("*.md") if ".git" not in path.relative_to(ROOT).parts
    )
    for document in markdown_files:
        text = document.read_text(encoding="utf-8-sig")
        for match in re.finditer(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", text):
            target = match[1].strip()
            if target.startswith("<") and ">" in target:
                target = target[1:target.index(">")]
            else:
                target = target.split(maxsplit=1)[0]
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            resolved = (document.parent / unquote(parsed.path)).resolve()
            link_count += 1
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                errors.append(f"Broken local link in {document.relative_to(ROOT).as_posix()}: {target}")

    csv_path = repository_path(f"{CAD}/零件清单.csv")
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8:
        errors.append("Mechanical parts CSV must describe six formal parts and two fit coupons")
    for row in rows:
        for column in ("STL文件", "STP文件"):
            name = f"{CAD}/{row[column]}"
            if name not in indexed or not repository_path(name).is_file():
                errors.append(f"Part CSV references an unlisted asset: {name}")

    report = json.loads(
        repository_path(f"{CAD}/04_核对与源文件/交付文件最终核对.json").read_text(encoding="utf-8-sig")
    )
    for part in report["parts"]:
        folder = "02_先打螺纹试配件" if part["id"].startswith("T") else "01_正式打印零件"
        for key, extension in (("stl", "stl"), ("step", "stp")):
            name = f"{CAD}/{folder}/{part['id']}.{extension}"
            if name not in indexed or indexed[name]["sha256"] != part["sha256"][key]:
                errors.append(f"Asset differs from historical final CAD checksum: {name}")

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        print(f"FAILED: {len(errors)} issue(s)", file=sys.stderr)
        return 1

    mode = "pointer-aware (not a full LFS download check)" if pointer_count else "full files"
    print(f"PASS: {len(assets)} source assets; {total_bytes:,} bytes; {mode}")
    print(f"Verified: {actual_count} file hashes; {pointer_count} LFS pointers")
    print(f"Documentation: {len(markdown_files)} Markdown files; {link_count} local links")
    print(f"FPGA: original package hashes, {project_references} project references, and 8 ROM lane copies match")
    print("Network: five checkpoints, S4R1 topology, runtime identities, and 42 core-source references match")
    print("CAD: parts CSV references and historical final SHA-256 values match")
    print("Scope: file integrity only; no synthesis, timing, board, or physical print tests")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-lfs-pointers", action="store_true",
        help="Accept pointer metadata matching the manifest; does not verify remote LFS objects.",
    )
    args = parser.parse_args()
    try:
        return verify(args.allow_lfs_pointers)
    except (OSError, ValueError, KeyError, TypeError, csv.Error, ET.ParseError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
