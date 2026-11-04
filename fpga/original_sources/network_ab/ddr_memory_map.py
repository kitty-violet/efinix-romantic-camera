"""Validate the Q019 DDR map and generate matching RTL/firmware constants.

JSON is the only address source.  CPU addresses and DDR physical addresses are
separate spaces; the Sapphire AXI A window maps explicitly into development.
The generated files are committed artifacts.  Run with --check before builds
to reject stale constants without modifying them.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "hw/efinity/config/ddr_memory_map.json"
RTL_HEADER = Path("hw/efinity/prototypes/ddr_fabric/ddr_memory_map.vh")
C_HEADER = Path("hw/efinity/config/generated/ddr_memory_map.h")
ADDRESS_SPACE_SIZE = 1 << 32
KINDS = {"frame", "raw_frame", "reserved", "feature", "temporary", "development", "unallocated"}
RAW_PAYLOAD_BYTES = 1920 * 1080  # RAW8, packed as four pixels per 32-bit camera word.


@dataclass(frozen=True)
class Region:
    name: str
    kind: str
    base: int
    size: int

    @property
    def end(self) -> int:
        """Exclusive upper address; used to avoid inclusive-end off-by-one bugs."""
        return self.base + self.size


@dataclass(frozen=True)
class MemoryMap:
    physical_base: int
    physical_size: int
    alignment_bytes: int
    regions: tuple[Region, ...]
    cpu_source_base: int
    cpu_size: int
    cpu_target_region: str
    cpu_target_offset: int

    def region(self, name: str) -> Region:
        for region in self.regions:
            if region.name == name:
                return region
        raise ValueError(f"unknown DDR region: {name}")

    @property
    def cpu_ddr_base(self) -> int:
        return self.region(self.cpu_target_region).base + self.cpu_target_offset

    @property
    def raw_frames(self) -> tuple[Region, ...]:
        return tuple(region for region in self.regions if region.kind == "raw_frame")

    def constants(self) -> dict[str, int]:
        constants = {
            "PHYSICAL_BASE": self.physical_base,
            "PHYSICAL_SIZE": self.physical_size,
            "PHYSICAL_LIMIT": self.physical_base + self.physical_size - 1,
            "ALIGNMENT_BYTES": self.alignment_bytes,
        }
        for region in self.regions:
            prefix = region.name.upper()
            constants[f"{prefix}_BASE"] = region.base
            constants[f"{prefix}_SIZE"] = region.size
            constants[f"{prefix}_LIMIT"] = region.end - 1
        constants.update({
            "RAW_BASE": self.raw_frames[0].base,
            "RAW_BANK_STRIDE": self.raw_frames[0].size,
            "RAW_BANK_COUNT": len(self.raw_frames),
            "CPU_SOURCE_BASE": self.cpu_source_base,
            "CPU_SOURCE_SIZE": self.cpu_size,
            "CPU_SOURCE_LIMIT": self.cpu_source_base + self.cpu_size - 1,
            "CPU_DDR_BASE": self.cpu_ddr_base,
            "CPU_DDR_SIZE": self.cpu_size,
            "CPU_DDR_LIMIT": self.cpu_ddr_base + self.cpu_size - 1,
        })
        return constants


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{label}: expected integer or hexadecimal string")
    try:
        result = int(value, 0) if isinstance(value, str) else value
    except ValueError as error:
        raise ValueError(f"{label}: invalid integer {value!r}") from error
    if result < 0 or result >= ADDRESS_SPACE_SIZE:
        raise ValueError(f"{label}: outside unsigned 32-bit range")
    return result


def _range(base: int, size: int, alignment: int, label: str) -> None:
    if size == 0:
        raise ValueError(f"{label}: zero size")
    if base % alignment or size % alignment:
        raise ValueError(f"{label}: base and size must be {alignment}-byte aligned")
    if base + size > ADDRESS_SPACE_SIZE:
        raise ValueError(f"{label}: address wrap beyond 32-bit range")


def validate_document(document: dict) -> MemoryMap:
    if document.get("schema_version") != 1:
        raise ValueError("unsupported DDR map schema_version")
    try:
        alignment = _integer(document["alignment_bytes"], "alignment_bytes")
        if alignment != 16:
            raise ValueError("DDR AXI requires 16-byte map alignment")
        physical = document["physical"]
        physical_base = _integer(physical["base"], "physical.base")
        physical_size = _integer(physical["size"], "physical.size")
        _range(physical_base, physical_size, alignment, "physical DDR")
        if physical_base != 0 or physical_size != 256 * 1024 * 1024:
            raise ValueError("Ti60 V4 physical DDR must be 256 MiB starting at zero")

        regions = []
        seen = set()
        for item in document["regions"]:
            name = item["name"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]*", name):
                raise ValueError(f"invalid region name: {name!r}")
            if name in seen:
                raise ValueError(f"duplicate region: {name}")
            seen.add(name)
            kind = item["kind"]
            if kind not in KINDS:
                raise ValueError(f"{name}: unknown region kind {kind!r}")
            base = _integer(item["base"], f"{name}.base")
            size = _integer(item["size"], f"{name}.size")
            _range(base, size, alignment, name)
            if base < physical_base or base + size > physical_base + physical_size:
                raise ValueError(f"{name}: region outside physical DDR")
            regions.append(Region(name, kind, base, size))

        regions.sort(key=lambda region: region.base)
        cursor = physical_base
        for region in regions:
            if region.base < cursor:
                raise ValueError(f"{region.name}: overlapping DDR region")
            if region.base != cursor:
                raise ValueError(f"{region.name}: uncovered DDR gap; mark it reserved or unallocated")
            cursor = region.end
        if cursor != physical_base + physical_size:
            raise ValueError("DDR partition table does not cover the physical address space")

        # Derive the raw bridge's base/stride/count from actual allocated regions;
        # a second manually maintained pool address would allow silent drift.
        raw_frames = [region for region in regions if region.kind == "raw_frame"]
        if len(raw_frames) != 3:
            raise ValueError("RAW frame pool requires exactly three banks")
        if any(region.size != raw_frames[0].size for region in raw_frames):
            raise ValueError("RAW frame banks must have equal size")
        if any(right.base != left.end for left, right in zip(raw_frames, raw_frames[1:])):
            raise ValueError("RAW frame banks must be contiguous")
        if raw_frames[0].size < RAW_PAYLOAD_BYTES:
            raise ValueError("RAW frame bank cannot hold the 1920x1080 RAW8 payload")

        cpu = document["cpu_window"]
        source_base = _integer(cpu["source_base"], "cpu_window.source_base")
        cpu_size = _integer(cpu["size"], "cpu_window.size")
        _range(source_base, cpu_size, alignment, "CPU source window")
        target_offset = _integer(cpu["target_offset"], "cpu_window.target_offset")
        if target_offset % alignment:
            raise ValueError("CPU target offset must be 16-byte aligned")
        memory_map = MemoryMap(physical_base, physical_size, alignment, tuple(regions),
                               source_base, cpu_size, cpu["target_region"], target_offset)
        target = memory_map.region(memory_map.cpu_target_region)
        if target.kind != "development":
            raise ValueError("CPU development window must map into a development region")
        if target_offset + cpu_size > target.size:
            raise ValueError("CPU mapping exceeds its target region")
        return memory_map
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid DDR map structure: {error}") from error


def load_memory_map(path: Path = DEFAULT_CONFIG) -> MemoryMap:
    with path.open(encoding="utf-8-sig") as handle:
        document = json.load(handle)
    if not isinstance(document, dict):
        raise ValueError("DDR map must be a JSON object")
    return validate_document(document)


def render_headers(memory_map: MemoryMap) -> dict[Path, str]:
    notice = "Generated from hw/efinity/config/ddr_memory_map.json; do not edit."
    constants = memory_map.constants()
    rtl = [f"// {notice}", "`ifndef TI60_DDR_MEMORY_MAP_VH", "`define TI60_DDR_MEMORY_MAP_VH", ""]
    c = [f"/* {notice} */", "#ifndef TI60_DDR_MEMORY_MAP_H", "#define TI60_DDR_MEMORY_MAP_H",
         "#include <stdint.h>", ""]
    for name, value in constants.items():
        rtl.append(f"`define TI60_DDR_{name} 32'h{value:08x}")
        c.append(f"#define TI60_DDR_{name} UINT32_C(0x{value:08x})")
    rtl += ["", "`endif", ""]
    c += ["", "#endif", ""]
    return {RTL_HEADER: "\n".join(rtl), C_HEADER: "\n".join(c)}


def sync_headers(memory_map: MemoryMap, *, check: bool, root: Path = PROJECT_ROOT) -> None:
    stale = []
    for relative, content in render_headers(memory_map).items():
        path = root / relative
        if check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                stale.append(str(relative))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(content)
    if stale:
        raise ValueError("generated DDR constants drift: " + ", ".join(stale))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--check", action="store_true", help="reject generated-file drift; write nothing")
    args = parser.parse_args()
    try:
        memory_map = load_memory_map(args.config)
        sync_headers(memory_map, check=args.check)
    except (OSError, ValueError) as error:
        parser.exit(1, f"DDR_MEMORY_MAP_FAIL: {error}\n")
    action = "CHECK" if args.check else "GENERATE"
    print(f"DDR_MEMORY_MAP_{action}_PASS regions={len(memory_map.regions)} "
          f"cpu=0x{memory_map.cpu_source_base:08x}->0x{memory_map.cpu_ddr_base:08x} "
          f"bytes={memory_map.cpu_size}")


if __name__ == "__main__":
    main()
