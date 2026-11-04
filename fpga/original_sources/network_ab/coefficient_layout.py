"""Versioned single-Model-B coefficient image and five-bank loading contract.

This is a synthetic, reversible capacity/loader proof, not trained weights.
Residual scale records remain invalid and therefore complete-network startup
is explicitly prohibited. No existing model, QAT audit, or RTL is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from model_ab_spec import build_model_b, validate_model
from q17_scale_audit import known_operation_contract


SCHEMA = "ti60-model-b-coefficients-v1"
FORMAT_VERSION = 1
MAGIC = b"TI60CFB1"
MODEL_CODE = 2
BANKS = 5
WORD_BYTES = 8
ALIGNMENT = 64
HEADER = struct.Struct("<8s8I24s")
SECTION_ENTRY = struct.Struct("<8I")
OP_ENTRY = struct.Struct("<16I")
IMAGE_FLAGS = 3  # bit0 synthetic, bit1 unresolved residual scaling
I32_MIN, I32_MAX = -(1 << 31), (1 << 31) - 1
Q17_MIN, Q17_MAX = -(1 << 17), (1 << 17) - 1


class LayoutError(ValueError):
    pass


def align64(value: int) -> int:
    return (value + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT


@dataclass(frozen=True)
class Operation:
    name: str
    kind: str
    input_channels: int
    output_channels: int
    kernel: int
    stride: int

    @property
    def padded_channels(self) -> int:
        return (self.output_channels + 7) // 8 * 8

    @property
    def terms(self) -> int:
        return self.kernel**2 * (1 if self.kind == "dw3x3" else self.input_channels)

    @property
    def raw_weights(self) -> int:
        return self.output_channels * self.terms


@dataclass(frozen=True)
class Parameters:
    # Original tensor order is OC, then ky,kx,IC (DW has one IC per OC).
    weights: tuple[tuple[int, ...], ...]
    bias: tuple[int, ...]
    multiplier: tuple[int, ...]
    relu: bool


def expanded_operations() -> tuple[Operation, ...]:
    model = build_model_b()
    validate_model("B", model)
    result = []
    for layer in model.layers:
        if layer.operator.startswith("dw3x3_s"):
            stride = int(layer.operator[len("dw3x3_s")])
            result.append(Operation(layer.name + ".dw", "dw3x3", layer.input.channels,
                                    layer.input.channels, 3, stride))
            result.append(Operation(layer.name + ".pw", "pw1x1", layer.input.channels,
                                    layer.output.channels, 1, 1))
        elif layer.operator in ("conv3x3", "dw3x3", "pw1x1"):
            result.append(Operation(layer.name, layer.operator, layer.input.channels,
                                    layer.output.channels, 1 if layer.operator == "pw1x1" else 3, 1))
    if [(op.name, op.output_channels) for op in result] != known_operation_contract(model.model_id):
        raise LayoutError("operation order differs from the existing QAT audit contract")
    if sum(op.raw_weights for op in result) != model.weight_count:
        raise LayoutError("expanded weights differ from the frozen model")
    return tuple(result)


def weight_positions(op: Operation) -> Iterable[tuple[int, int]]:
    """Emit (OC, original flattened term), including padded OC lanes.

    Every eight yielded bytes form one 64-bit word. Convolution/PW use
    OC8, IC block (up to eight actual channels), ky,kx, within-block IC,
    OC lane. DW uses OC8,ky,kx,OC lane. RGB input IC3 is not padded to IC8.
    """
    for output_block in range(0, op.padded_channels, 8):
        if op.kind == "dw3x3":
            for tap in range(op.kernel**2):
                for lane in range(8):
                    yield output_block + lane, tap
        else:
            for input_block in range(0, op.input_channels, 8):
                for ky in range(op.kernel):
                    for kx in range(op.kernel):
                        for ic in range(input_block, min(input_block + 8, op.input_channels)):
                            term = (ky * op.kernel + kx) * op.input_channels + ic
                            for lane in range(8):
                                yield output_block + lane, term


def build_layout() -> dict:
    operations = expanded_operations()
    names = ("operators", "weights", "bias", "multiplier", "relu", "residual_reserved")
    relative = {}
    lengths = {"operators": len(operations) * OP_ENTRY.size, "residual_reserved": 3 * 64}
    for kind in ("weights", "bias", "multiplier", "relu"):
        cursor = 0
        for op in operations:
            if kind != "relu":
                cursor = align64(cursor)
            count = (op.padded_channels * op.terms if kind == "weights" else
                     op.padded_channels * 4 if kind in ("bias", "multiplier") else
                     op.padded_channels // 8)
            relative[(op.name, kind)] = (cursor, count)
            cursor += count
        lengths[kind] = align64(cursor)
    cursor = align64(HEADER.size + len(names) * SECTION_ENTRY.size)
    sections = []
    for code, name in enumerate(names, 1):
        sections.append({"name": name, "kind": code, "offset": cursor,
                         "length": lengths[name], "alignment": ALIGNMENT,
                         "flags": 1 if name == "residual_reserved" else 0})
        cursor = align64(cursor + lengths[name])
    section_map = {section["name"]: section for section in sections}
    records = []
    for index, op in enumerate(operations):
        record = {"index": index, **asdict(op), "padded_channels": op.padded_channels,
                  "raw_weights": op.raw_weights, "terms": op.terms}
        for kind in ("weights", "bias", "multiplier", "relu"):
            offset, length = relative[(op.name, kind)]
            record[kind] = {"offset": section_map[kind]["offset"] + offset, "length": length}
        records.append(record)
    return {"schema": SCHEMA, "version": FORMAT_VERSION, "model_id": build_model_b().model_id,
            "model_code": MODEL_CODE, "graph_nodes": len(build_model_b().layers),
            "quantized_operators": len(operations), "valid_channels": sum(op.output_channels for op in operations),
            "padded_channels": sum(op.padded_channels for op in operations),
            "raw_weight_bytes": sum(op.raw_weights for op in operations),
            "oc_padded_weight_bytes": sum(op.padded_channels * op.terms for op in operations),
            "sections": sections, "operators": records, "total_bytes": cursor,
            "residual_scaling": {"status": "UNDEFINED", "valid": False,
                                 "records": ["res1_add", "res2_add", "res3_add"],
                                 "bytes_per_record_reserved": 64,
                                 "note": "Q045 requires both inputs aligned to common INT32 then one output quantization; no same-scale assumption."},
            "quality_gate": "UNMEASURED", "complete_network_start_allowed": False,
            "scope": "One synthetic Model B style network; no gesture network or second style parameters."}


def _checked_integer(value: int, low: int, high: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise LayoutError(f"{field} must be integer in [{low},{high}]")
    return value


def validate_parameters(parameters: dict[str, Parameters]) -> None:
    operations = expanded_operations()
    if list(parameters) != [op.name for op in operations]:
        raise LayoutError("parameter set must contain exactly the frozen ordered 24 operations")
    for op in operations:
        p = parameters[op.name]
        if not isinstance(p, Parameters) or not isinstance(p.relu, bool):
            raise LayoutError(f"{op.name}: expected Parameters with boolean layer ReLU")
        if len(p.weights) != op.output_channels or any(len(row) != op.terms for row in p.weights):
            raise LayoutError(f"{op.name}: original OC/ky/kx/IC tensor dimensions mismatch")
        if len(p.bias) != op.output_channels or len(p.multiplier) != op.output_channels:
            raise LayoutError(f"{op.name}: per-output-channel table length mismatch")
        for oc, row in enumerate(p.weights):
            for term, weight in enumerate(row):
                _checked_integer(weight, -128, 127, f"{op.name}.weights[{oc}][{term}]")
        for value in p.bias:
            _checked_integer(value, I32_MIN, I32_MAX, f"{op.name}.bias")
        for value in p.multiplier:
            _checked_integer(value, Q17_MIN, Q17_MAX, f"{op.name}.multiplier signed18 in INT32 field")


def synthetic_parameters() -> dict[str, Parameters]:
    result = {}
    scales = (Q17_MIN, -65536, -1, 0, 1, 65536, Q17_MAX)
    for index, op in enumerate(expanded_operations()):
        weights = tuple(tuple(((index * 31 + oc * 47 + term * 13) % 256) - 128
                              for term in range(op.terms)) for oc in range(op.output_channels))
        bias = tuple(I32_MIN if oc == 0 else I32_MAX if oc == 1 else
                     (index * 101 + oc * 503 - 8192) for oc in range(op.output_channels))
        multiplier = tuple(scales[(index + oc) % len(scales)] for oc in range(op.output_channels))
        result[op.name] = Parameters(weights, bias, multiplier, bool(index % 2 and op.name != "rgb_head"))
    return result


def _descriptor(record: dict) -> bytes:
    return OP_ENTRY.pack(record["index"], {"conv3x3": 1, "dw3x3": 2, "pw1x1": 3}[record["kind"]],
                         record["input_channels"], record["output_channels"], record["kernel"], record["stride"],
                         record["padded_channels"], record["weights"]["offset"], record["weights"]["length"],
                         record["bias"]["offset"], record["multiplier"]["offset"], record["padded_channels"],
                         record["relu"]["offset"], record["relu"]["length"], 0, 0)


def encode_parameters(parameters: dict[str, Parameters]) -> bytes:
    validate_parameters(parameters)
    layout = build_layout()
    image = bytearray(layout["total_bytes"])
    descriptor_offset = layout["sections"][0]["offset"]
    for op, record in zip(expanded_operations(), layout["operators"]):
        p = parameters[op.name]
        desc = descriptor_offset + record["index"] * OP_ENTRY.size
        image[desc:desc + OP_ENTRY.size] = _descriptor(record)
        start = record["weights"]["offset"]
        for index, (oc, term) in enumerate(weight_positions(op)):
            image[start + index] = p.weights[oc][term] & 255 if oc < op.output_channels else 0
        for kind in ("bias", "multiplier"):
            values = getattr(p, kind)
            for oc, value in enumerate(values):
                struct.pack_into("<i", image, record[kind]["offset"] + oc * 4, value)
        for group in range(op.padded_channels // 8):
            valid_lanes = min(8, op.output_channels - group * 8)
            image[record["relu"]["offset"] + group] = (1 << valid_lanes) - 1 if p.relu else 0
    # Reproducible synthetic build identifier covers the resident parameter area.
    build_id = zlib.crc32(image[layout["sections"][1]["offset"]:]) & 0xFFFFFFFF
    image[:HEADER.size] = HEADER.pack(MAGIC, FORMAT_VERSION, len(image), len(layout["operators"]),
                                     len(layout["sections"]), HEADER.size, MODEL_CODE, build_id, IMAGE_FLAGS, bytes(24))
    for index, section in enumerate(layout["sections"]):
        payload = image[section["offset"]:section["offset"] + section["length"]]
        SECTION_ENTRY.pack_into(image, HEADER.size + index * SECTION_ENTRY.size,
                                section["kind"], section["offset"], section["length"], ALIGNMENT,
                                zlib.crc32(payload) & 0xFFFFFFFF, section["flags"], 0, 0)
    return bytes(image)


def decode_image(image: bytes, *, expected_sha256: str | None = None,
                 require_complete_network: bool = False) -> dict[str, Parameters]:
    layout = build_layout()
    if len(image) != layout["total_bytes"]:
        raise LayoutError("image byte length differs from frozen layout")
    if expected_sha256 is not None and hashlib.sha256(image).hexdigest() != expected_sha256.lower():
        raise LayoutError("image SHA-256 mismatch")
    header = HEADER.unpack_from(image)
    if header[:7] != (MAGIC, FORMAT_VERSION, len(image), len(layout["operators"]),
                      len(layout["sections"]), HEADER.size, MODEL_CODE) or header[8:] != (IMAGE_FLAGS, bytes(24)):
        raise LayoutError("invalid magic/version/model/header flags/reserved bytes")
    if require_complete_network:
        raise LayoutError("complete network startup rejected: synthetic coefficients and residual scaling UNDEFINED")
    for index, section in enumerate(layout["sections"]):
        entry = SECTION_ENTRY.unpack_from(image, HEADER.size + index * SECTION_ENTRY.size)
        payload = image[section["offset"]:section["offset"] + section["length"]]
        expected = (section["kind"], section["offset"], section["length"], ALIGNMENT,
                    zlib.crc32(payload) & 0xFFFFFFFF, section["flags"], 0, 0)
        if entry != expected:
            raise LayoutError(f"invalid {section['name']} range/alignment/CRC/flags")
    result = {}
    descriptor_offset = layout["sections"][0]["offset"]
    for op, record in zip(expanded_operations(), layout["operators"]):
        desc = descriptor_offset + record["index"] * OP_ENTRY.size
        if image[desc:desc + OP_ENTRY.size] != _descriptor(record):
            raise LayoutError(f"{op.name}: descriptor differs from model contract")
        weights = [[0] * op.terms for _ in range(op.output_channels)]
        start = record["weights"]["offset"]
        for index, (oc, term) in enumerate(weight_positions(op)):
            byte = image[start + index]
            if oc < op.output_channels:
                weights[oc][term] = byte - 256 if byte >= 128 else byte
            elif byte:
                raise LayoutError(f"{op.name}: nonzero padded weight lane")
        bias = tuple(struct.unpack_from("<i", image, record["bias"]["offset"] + oc * 4)[0]
                     for oc in range(op.output_channels))
        multiplier = tuple(struct.unpack_from("<i", image, record["multiplier"]["offset"] + oc * 4)[0]
                           for oc in range(op.output_channels))
        flags = [bool(image[record["relu"]["offset"] + oc // 8] & (1 << (oc % 8)))
                 for oc in range(op.output_channels)]
        if any(flag != flags[0] for flag in flags):
            raise LayoutError(f"{op.name}: channel ReLU differs from declared layer ReLU contract")
        result[op.name] = Parameters(tuple(tuple(row) for row in weights), bias, multiplier, flags[0])
    # Re-encoding validates every padding byte, residual valid=0 record, signed18
    # field, build identifier and zero-initialized unused OC lane, not only CRC.
    if encode_parameters(result) != image:
        raise LayoutError("noncanonical padding/residual/build identifier or parameter image")
    return result


def bank_location(byte_offset: int, depth: int) -> tuple[int, int, int]:
    _checked_integer(depth, 1, 1 << 20, "bank depth")
    _checked_integer(byte_offset, 0, BANKS * depth * WORD_BYTES - 1, "byte offset")
    word, byte_lane = divmod(byte_offset, WORD_BYTES)
    address, bank = divmod(word, BANKS)
    return bank, address, byte_lane


def capacity_report(depth: int) -> dict:
    _checked_integer(depth, 1, 1 << 20, "bank depth")
    used = build_layout()["total_bytes"]
    words = used // WORD_BYTES
    per_bank = [(words + BANKS - 1 - bank) // BANKS for bank in range(BANKS)]
    capacity = BANKS * depth * WORD_BYTES
    return {"banks": BANKS, "word_bits": WORD_BYTES * 8, "depth": depth,
            "capacity_bytes": capacity, "image_bytes": used, "fits": used <= capacity,
            "unused_bytes": max(0, capacity - used), "shortfall_bytes": max(0, used - capacity),
            "used_words_per_bank": per_bank, "required_depth": max(per_bank),
            "physical_resource_gate": "SEPARATE_MAP_EVIDENCE_REQUIRED"}


def load_banks(image: bytes, *, depth: int, expected_sha256: str) -> tuple[bytes, ...]:
    if not capacity_report(depth)["fits"]:
        raise LayoutError(f"5x64x{depth} capacity cannot hold {len(image)} bytes")
    decode_image(image, expected_sha256=expected_sha256)
    banks = [bytearray(depth * WORD_BYTES) for _ in range(BANKS)]
    for offset, byte in enumerate(image):
        bank, address, lane = bank_location(offset, depth)
        banks[bank][address * WORD_BYTES + lane] = byte
    return tuple(bytes(bank) for bank in banks)


def read_banks(banks: tuple[bytes, ...], image_length: int) -> bytes:
    if len(banks) != BANKS or not banks[0] or len(banks[0]) % WORD_BYTES or any(len(bank) != len(banks[0]) for bank in banks):
        raise LayoutError("expected five equal nonempty 64-bit-word banks")
    depth = len(banks[0]) // WORD_BYTES
    _checked_integer(image_length, 1, BANKS * depth * WORD_BYTES, "image length")
    result = bytearray()
    for offset in range(BANKS * depth * WORD_BYTES):
        bank, address, lane = bank_location(offset, depth)
        value = banks[bank][address * WORD_BYTES + lane]
        if offset < image_length:
            result.append(value)
        elif value:
            raise LayoutError("unused bank capacity must be zero after full-image loading")
    return bytes(result)


def emit_artifacts(directory: Path, depth: int = 512) -> dict:
    parameters = synthetic_parameters()
    image = encode_parameters(parameters)
    sha = hashlib.sha256(image).hexdigest()
    banks = load_banks(image, depth=depth, expected_sha256=sha)
    loaded = read_banks(banks, len(image))
    if decode_image(loaded, expected_sha256=sha) != parameters:
        raise AssertionError("complete parameter roundtrip mismatch")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "coefficients.bin").write_bytes(image)
    for bank, data in enumerate(banks):
        text = "".join(f"{int.from_bytes(data[address:address + 8], 'little'):016x}\n"
                       for address in range(0, len(data), 8))
        (directory / f"bank{bank}.hex").write_text(text, encoding="ascii", newline="\n")
    manifest = build_layout()
    manifest.update({"sha256": sha, "crc32": f"{zlib.crc32(image) & 0xFFFFFFFF:08x}",
                     "build_identifier": f"{HEADER.unpack_from(image)[7]:08x}",
                     "byte_order": "little", "weights_source": "SYNTHETIC",
                     "multiplier_contract": "signed18 Q17 stored as sign-extended INT32; fixed shift17",
                     "weight_order": "OC8, ICblock8(actual IC only), ky,kx, IC within block, OC lane; DW OC8,ky,kx,lane",
                     "bank_mapping": "word=byte_offset//8; bank=word%5; address=word//5; lane=byte_offset%8",
                     "capacity_256": capacity_report(256), "capacity_512": capacity_report(512),
                     "loaded_bank_depth": depth, "roundtrip": "PASS",
                     "loader_contract": ["Validate magic/version/model/length, 64-byte section alignment and ranges.",
                                         "Validate section CRC32, complete image SHA-256, descriptors and zero padding.",
                                         "Reject insufficient bank depth; write every bank word including zero unused capacity.",
                                         "Read back image and verify SHA-256 before exposing coefficients to an operator.",
                                         "Complete-network start remains rejected while residual records are invalid or weights synthetic."],
                     "section_crc32": {s["name"]: f"{zlib.crc32(image[s['offset']:s['offset'] + s['length']]) & 0xFFFFFFFF:08x}"
                                         for s in manifest["sections"]}})
    (directory / "coefficient_layout.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bank-depth", type=int, default=512)
    args = parser.parse_args()
    manifest = emit_artifacts(args.output_dir, args.bank_depth)
    print(f"PASS COEFFICIENT_LAYOUT version={FORMAT_VERSION} ops={manifest['quantized_operators']} "
          f"bytes={manifest['total_bytes']} depth={args.bank_depth} sha256={manifest['sha256']}")
    print("COMPLETE_NETWORK_START=REJECTED residual_scaling=UNDEFINED weights=SYNTHETIC quality=UNMEASURED")


if __name__ == "__main__":
    main()
