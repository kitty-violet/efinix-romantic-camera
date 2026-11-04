"""Early BRAM/DDR/tile schedule model for the 720p model-B accelerator.

The model deliberately separates facts from assumptions:

* Ti60 RAM count, v13 usage, Q17 pressure-shell usage and the DDR/AXI peak are
  measured or frozen project facts.
* Tile halo, logical traffic and RAM packing are deterministic calculations.
* AXI efficiency is *not* assumed.  The output reports the minimum measured
  sustained bandwidth that Q027 must reach to retain 15% headroom.

This is an early architecture gate, not a replacement for RTL P&R or a board
DDR microbenchmark.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ddr_memory_map import load_memory_map


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "hw" / "efinity" / "prototypes" / "operator_core"))
import operator_throughput_model as throughput  # noqa: E402


FPS = 30
RAM_BLOCK_BITS = 10_240
DEVICE_RAM_BLOCKS = 256
RAM_TARGET_BLOCKS = int(DEVICE_RAM_BLOCKS * 0.85)
RAM_HARD_LIMIT_BLOCKS = int(DEVICE_RAM_BLOCKS * 0.90)
V13_RAM_BLOCKS = 157
V13_DEBUG_RAM_BLOCKS = 2
Q17_ENGINE_RAM_BLOCKS = 20  # measured 14 logical groups folded into five 64-bit banks
CNN_DMA_FIFO_RAM_BLOCKS = 8  # 4 KiB read + 4 KiB write
CNN_CONTROL_RAM_BLOCKS = 2
PORT_AWARE_LINE_BUFFER_BLOCKS_TILE_LE_320 = 32
AREA_DOWNSCALER_CANDIDATE_SAVING_BLOCKS = 10  # measured v13 157 -> v14 147

DDR_PEAK_BYTES_PER_SECOND = 1_600_000_000  # 16-bit DDR3-800 and 128-bit AXI @100 MHz
DDR_ALLOWED_UTILIZATION = 0.85

# CSI Pack_40 discards the low two RAW10 bits and stores four RAW8 samples in
# each 32-bit camera word.  This is therefore one byte per sensor pixel, not a
# packed-RAW10 or 32-bit-per-pixel frame.
RAW_1080_BYTES = 1920 * 1080
RGB_720_BYTES = 1280 * 720 * 3
BOTTLENECK_BYTES = 320 * 180 * 32

# Frozen logical DDR map from design-review Q019/Q112.  The tile DMA adapter
# receives the active RGB frame bases at runtime, but the Python contract uses
# frame A and output C so its vectors remain deterministic.
DDR_MEMORY_MAP = load_memory_map()
RGB_INPUT_A_BASE = DDR_MEMORY_MAP.region("frame_a").base
RGB_OUTPUT_C_BASE = DDR_MEMORY_MAP.region("frame_c").base
FEATURE0_BASE = DDR_MEMORY_MAP.region("feature0").base
FEATURE1_BASE = DDR_MEMORY_MAP.region("feature1").base
AXI_DATA_BYTES = 16
AXI_MAX_BURST_BEATS = 128


@dataclass(frozen=True)
class SpatialOp:
    name: str
    kind: str
    input_width: int
    output_width: int
    input_channels: int
    output_channels: int
    output_height: int
    stride: int = 1

    @property
    def has_line_buffer(self) -> bool:
        return self.kind in {"conv3", "dw3"}


@dataclass(frozen=True)
class GroupResult:
    name: str
    output_width: int
    tile_width: int
    tile_count: int
    line_buffer_blocks: int
    line_buffers: tuple[dict[str, int | str], ...]
    source_read_bytes: int
    operation_output_values: dict[str, int]


def clamp_interval(lo: int, hi: int, width: int) -> tuple[int, int]:
    return max(0, lo), min(width - 1, hi)


def interval_width(interval: tuple[int, int]) -> int:
    lo, hi = interval
    return hi - lo + 1


def reverse_interval(op: SpatialOp, output_interval: tuple[int, int]) -> tuple[int, int]:
    lo, hi = output_interval
    if op.kind in {"pw1", "identity"}:
        candidate = (lo, hi)
    elif op.kind in {"conv3", "dw3"} and op.stride == 1:
        candidate = (lo - 1, hi + 1)
    elif op.kind in {"conv3", "dw3"} and op.stride == 2:
        candidate = (2 * lo - 1, 2 * hi + 1)
    elif op.kind == "nn2":
        candidate = (lo // 2, hi // 2)
    else:
        raise ValueError(f"unsupported spatial operation: {op}")
    return clamp_interval(*candidate, op.input_width)


def ram_blocks_for_two_rows(width: int, channels: int) -> int:
    # Rows are independent circular banks.  Round each row separately so the
    # estimate exposes real 10K-block fragmentation instead of only total bits.
    bits_per_row = width * channels * 8
    return 2 * math.ceil(bits_per_row / RAM_BLOCK_BITS)


def evaluate_group(name: str, operations: tuple[SpatialOp, ...], tile_width: int) -> GroupResult:
    output_width = operations[-1].output_width
    if tile_width <= 0 or output_width % tile_width != 0:
        raise ValueError(f"{name}: tile width {tile_width} must divide {output_width}")

    operation_output_widths = {operation.name: 0 for operation in operations if operation.kind != "nn2"}
    max_line_widths = {operation.name: 0 for operation in operations if operation.has_line_buffer}
    aggregate_line_input_columns = {
        operation.name: 0 for operation in operations if operation.has_line_buffer
    }
    source_read_columns = 0
    tile_count = output_width // tile_width

    for tile in range(tile_count):
        desired = (tile * tile_width, (tile + 1) * tile_width - 1)
        for operation in reversed(operations):
            if operation.kind != "nn2":
                operation_output_widths[operation.name] += interval_width(desired)
            input_interval = reverse_interval(operation, desired)
            if operation.has_line_buffer:
                max_line_widths[operation.name] = max(
                    max_line_widths[operation.name], interval_width(input_interval)
                )
                aggregate_line_input_columns[operation.name] += interval_width(input_interval)
            desired = input_interval
        source_read_columns += interval_width(desired)

    line_buffers = tuple(
        {
            "operation": operation.name,
            "columns": max_line_widths[operation.name],
            "channels": operation.input_channels,
            "aggregate_input_columns": aggregate_line_input_columns[operation.name],
            "row_read_words_64bit": aggregate_line_input_columns[operation.name]
            * operation.output_height
            * operation.stride
            * math.ceil(operation.input_channels / 8),
            "capacity_lower_bound_blocks": ram_blocks_for_two_rows(
                max_line_widths[operation.name], operation.input_channels
            ),
        }
        for operation in operations
        if operation.has_line_buffer
    )
    line_buffer_blocks = sum(
        int(line_buffer["capacity_lower_bound_blocks"])
        for line_buffer in line_buffers
    )
    operation_output_values = {
        operation.name: operation_output_widths[operation.name]
        * operation.output_height
        * operation.output_channels
        for operation in operations
        if operation.kind != "nn2"
    }
    first = operations[0]
    source_read_bytes = source_read_columns * first.output_height * first.input_channels
    # For stride-2 encoder stages, the first op's output height differs from
    # its input height.  The source is always the full 720-line RGB tensor.
    if name == "encoder":
        source_read_bytes = source_read_columns * 720 * first.input_channels

    return GroupResult(
        name=name,
        output_width=output_width,
        tile_width=tile_width,
        tile_count=tile_count,
        line_buffer_blocks=line_buffer_blocks,
        line_buffers=line_buffers,
        source_read_bytes=source_read_bytes,
        operation_output_values=operation_output_values,
    )


def build_spatial_group_definitions(
    output_tile_width: int,
) -> tuple[tuple[str, tuple[SpatialOp, ...], int], ...]:
    if output_tile_width % 4 != 0:
        raise ValueError("720p output tile width must be divisible by 4")
    bottleneck_tile = output_tile_width // 4

    encoder_ops = (
        SpatialOp("stem", "conv3", 1280, 1280, 3, 8, 720),
        SpatialOp("down1.dw", "dw3", 1280, 640, 8, 8, 360, stride=2),
        SpatialOp("down1.pw", "pw1", 640, 640, 8, 16, 360),
        SpatialOp("down2.dw", "dw3", 640, 320, 16, 16, 180, stride=2),
        SpatialOp("down2.pw", "pw1", 320, 320, 16, 32, 180),
    )
    residual_ops = (
        SpatialOp("resX_conv1.dw", "dw3", 320, 320, 32, 32, 180),
        SpatialOp("resX_conv1.pw", "pw1", 320, 320, 32, 32, 180),
        SpatialOp("resX_conv2.dw", "dw3", 320, 320, 32, 32, 180),
        SpatialOp("resX_conv2.pw", "pw1", 320, 320, 32, 32, 180),
    )
    decoder_ops = (
        SpatialOp("up1_low_dw", "dw3", 320, 320, 32, 32, 180),
        SpatialOp("up1_low_pw", "pw1", 320, 320, 32, 16, 180),
        SpatialOp("up1_nn", "nn2", 320, 640, 16, 16, 360),
        SpatialOp("up1_high_dw", "dw3", 640, 640, 16, 16, 360),
        SpatialOp("up2_low_dw", "dw3", 640, 640, 16, 16, 360),
        SpatialOp("up2_low_pw", "pw1", 640, 640, 16, 8, 360),
        SpatialOp("up2_nn", "nn2", 640, 1280, 8, 8, 720),
        SpatialOp("up2_high_dw", "dw3", 1280, 1280, 8, 8, 720),
        SpatialOp("rgb_head", "pw1", 1280, 1280, 8, 3, 720),
    )

    return (
        ("encoder", encoder_ops, bottleneck_tile),
        ("residual", residual_ops, bottleneck_tile),
        ("decoder", decoder_ops, output_tile_width),
    )


def build_groups(output_tile_width: int) -> tuple[GroupResult, GroupResult, GroupResult]:
    return tuple(
        evaluate_group(name, operations, tile_width)
        for name, operations, tile_width in build_spatial_group_definitions(output_tile_width)
    )


def build_tile_manifest(output_tile_width: int) -> dict[str, object]:
    """Create the exact horizontal interval contract consumed by future RTL.

    The final output interval is the non-overlapping core that may be committed
    to DDR.  Every earlier interval includes the clipped receptive-field halo
    required to produce that core.  Keeping both intervals explicit prevents a
    DMA or wavefront controller from accidentally writing halo columns twice.
    """

    groups: list[dict[str, object]] = []
    for name, operations, tile_width in build_spatial_group_definitions(output_tile_width):
        output_width = operations[-1].output_width
        if output_width % tile_width != 0:
            raise ValueError(f"{name}: tile width {tile_width} must divide {output_width}")

        tiles: list[dict[str, object]] = []
        for tile_index in range(output_width // tile_width):
            core_interval = (tile_index * tile_width, (tile_index + 1) * tile_width - 1)
            desired = core_interval
            reverse_steps: list[dict[str, object]] = []
            for operation in reversed(operations):
                output_interval = desired
                input_interval = reverse_interval(operation, output_interval)
                reverse_steps.append(
                    {
                        "operation": operation.name,
                        "kind": operation.kind,
                        "stride": operation.stride,
                        "input_width": operation.input_width,
                        "output_width": operation.output_width,
                        "input_interval": {
                            "x0": input_interval[0],
                            "x1": input_interval[1],
                            "columns": interval_width(input_interval),
                        },
                        "output_interval": {
                            "x0": output_interval[0],
                            "x1": output_interval[1],
                            "columns": interval_width(output_interval),
                        },
                    }
                )
                desired = input_interval

            source_interval = desired
            tiles.append(
                {
                    "tile_index": tile_index,
                    "commit_interval": {
                        "x0": core_interval[0],
                        "x1": core_interval[1],
                        "columns": interval_width(core_interval),
                    },
                    "source_interval": {
                        "x0": source_interval[0],
                        "x1": source_interval[1],
                        "columns": interval_width(source_interval),
                    },
                    "steps": list(reversed(reverse_steps)),
                }
            )

        groups.append(
            {
                "name": name,
                "output_width": output_width,
                "tile_width": tile_width,
                "tile_count": len(tiles),
                "tiles": tiles,
            }
        )

    return {
        "output_tile_width": output_tile_width,
        "coordinate_convention": "inclusive_zero_based_x",
        "commit_rule": "write_only_commit_interval; halo_columns_are_read_or_computed_but_not_committed_twice",
        "groups": groups,
    }


def build_tile_descriptor_sequence(output_tile_width: int = 320) -> list[dict[str, int]]:
    """Flatten the manifest into the runtime order used by the RTL sequencer."""

    manifest = build_tile_manifest(output_tile_width)
    groups = {group["name"]: group for group in manifest["groups"]}
    sequence: list[dict[str, int]] = []
    for group_name, group_id, repetitions in (
        ("encoder", 0, 1),
        ("residual", 1, 3),
        ("decoder", 2, 1),
    ):
        for repetition in range(repetitions):
            residual_block = repetition + 1 if group_name == "residual" else 0
            for tile in groups[group_name]["tiles"]:
                sequence.append(
                    {
                        "group_id": group_id,
                        "residual_block": residual_block,
                        "tile_index": tile["tile_index"],
                        "source_x0": tile["source_interval"]["x0"],
                        "source_x1": tile["source_interval"]["x1"],
                        "commit_x0": tile["commit_interval"]["x0"],
                        "commit_x1": tile["commit_interval"]["x1"],
                    }
                )
    return sequence


def write_tile_descriptor_vectors(path: Path, output_tile_width: int = 320) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "{group_id} {residual_block} {tile_index} {source_x0} {source_x1} "
        "{commit_x0} {commit_x1}".format(**descriptor)
        for descriptor in build_tile_descriptor_sequence(output_tile_width)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def build_dma_context(descriptor: dict[str, int]) -> dict[str, object]:
    """Resolve one horizontal descriptor to its frozen DDR tensor ownership.

    Source and destination row counts are deliberately independent.  The
    encoder consumes 720 RGB rows and produces 180 feature rows, while the
    decoder performs the inverse.  A residual descriptor additionally rereads
    only its non-Halo core as the skip path.
    """

    group_id = descriptor["group_id"]
    residual_block = descriptor["residual_block"]
    if group_id == 0:
        return {
            "main": {
                "base": RGB_INPUT_A_BASE,
                "height": 720,
                "stride_bytes": 1280 * 3,
                "bytes_per_pixel": 3,
                "x0": descriptor["source_x0"],
                "x1": descriptor["source_x1"],
            },
            "skip": None,
            "write": {
                "base": FEATURE0_BASE,
                "height": 180,
                "stride_bytes": 320 * 32,
                "bytes_per_pixel": 32,
                "x0": descriptor["commit_x0"],
                "x1": descriptor["commit_x1"],
            },
        }
    if group_id == 1 and residual_block in (1, 2, 3):
        source_base = FEATURE0_BASE if residual_block in (1, 3) else FEATURE1_BASE
        destination_base = FEATURE1_BASE if residual_block in (1, 3) else FEATURE0_BASE
        source = {
            "base": source_base,
            "height": 180,
            "stride_bytes": 320 * 32,
            "bytes_per_pixel": 32,
            "x0": descriptor["source_x0"],
            "x1": descriptor["source_x1"],
        }
        return {
            "main": source,
            "skip": {
                **source,
                "x0": descriptor["commit_x0"],
                "x1": descriptor["commit_x1"],
            },
            "write": {
                "base": destination_base,
                "height": 180,
                "stride_bytes": 320 * 32,
                "bytes_per_pixel": 32,
                "x0": descriptor["commit_x0"],
                "x1": descriptor["commit_x1"],
            },
        }
    if group_id == 2:
        return {
            "main": {
                "base": FEATURE1_BASE,
                "height": 180,
                "stride_bytes": 320 * 32,
                "bytes_per_pixel": 32,
                "x0": descriptor["source_x0"],
                "x1": descriptor["source_x1"],
            },
            "skip": None,
            "write": {
                "base": RGB_OUTPUT_C_BASE,
                "height": 720,
                "stride_bytes": 1280 * 3,
                "bytes_per_pixel": 3,
                "x0": descriptor["commit_x0"],
                "x1": descriptor["commit_x1"],
            },
        }
    raise ValueError(
        f"invalid DMA descriptor group={group_id} residual={residual_block}"
    )


def build_row_dma_command(
    descriptor: dict[str, int], stream: str, row: int, tensor: dict[str, int]
) -> dict[str, int | str]:
    payload_address = (
        tensor["base"]
        + row * tensor["stride_bytes"]
        + tensor["x0"] * tensor["bytes_per_pixel"]
    )
    payload_bytes = (
        tensor["x1"] - tensor["x0"] + 1
    ) * tensor["bytes_per_pixel"]
    aligned_address = payload_address & ~(AXI_DATA_BYTES - 1)
    prefix_bytes = payload_address - aligned_address
    beats = _ceil_div(prefix_bytes + payload_bytes, AXI_DATA_BYTES)
    suffix_bytes = beats * AXI_DATA_BYTES - prefix_bytes - payload_bytes
    if stream == "write" and (prefix_bytes != 0 or suffix_bytes != 0):
        raise ValueError(
            "current write contract must be whole 128-bit beats; "
            f"addr=0x{payload_address:08x} bytes={payload_bytes}"
        )
    return {
        "stream": stream,
        "group_id": descriptor["group_id"],
        "residual_block": descriptor["residual_block"],
        "tile_index": descriptor["tile_index"],
        "row": row,
        "address": aligned_address,
        "beats": beats,
        "prefix_bytes": prefix_bytes,
        "payload_bytes": payload_bytes,
        "suffix_bytes": suffix_bytes,
        "x0": tensor["x0"],
        "x1": tensor["x1"],
    }


def build_tile_dma_command_streams(
    output_tile_width: int = 320,
) -> dict[str, list[dict[str, int | str]]]:
    streams: dict[str, list[dict[str, int | str]]] = {
        "main": [],
        "skip": [],
        "write": [],
    }
    for descriptor in build_tile_descriptor_sequence(output_tile_width):
        context = build_dma_context(descriptor)
        for stream in ("main", "skip", "write"):
            tensor = context[stream]
            if tensor is None:
                continue
            streams[stream].extend(
                build_row_dma_command(descriptor, stream, row, tensor)
                for row in range(tensor["height"])
            )
    return streams


def split_axi_bursts(
    address: int,
    beats: int,
    beat_bytes: int = AXI_DATA_BYTES,
    max_burst_beats: int = AXI_MAX_BURST_BEATS,
) -> list[tuple[int, int]]:
    """Apply the same maximum-length and 4 KiB rules as the RTL AXI DMA."""

    if address % beat_bytes or beats <= 0:
        raise ValueError("AXI command must be aligned and non-empty")
    bursts: list[tuple[int, int]] = []
    current = address
    remaining = beats
    while remaining:
        beats_to_4k = (4096 - (current & 0xFFF)) // beat_bytes
        burst_beats = min(remaining, max_burst_beats, beats_to_4k)
        if burst_beats <= 0:
            raise ValueError("cannot construct legal 4 KiB-bounded burst")
        bursts.append((current, burst_beats))
        current += burst_beats * beat_bytes
        remaining -= burst_beats
    return bursts


def write_tile_dma_vectors(directory: Path, output_tile_width: int = 320) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    streams = build_tile_dma_command_streams(output_tile_width)
    for stream, commands in streams.items():
        lines = [
            (
                "{group_id} {residual_block} {tile_index} {row} "
                "{address:08x} {beats} {prefix_bytes} {payload_bytes} {x0} {x1}"
            ).format(**command)
            for command in commands
        ]
        (directory / f"tile_dma_{stream}_vectors.txt").write_text(
            "\n".join(lines) + "\n", encoding="ascii"
        )


def expand_residual_overrides(template: dict[str, int]) -> dict[str, int]:
    result: dict[str, int] = {}
    for block in range(1, 4):
        for name, values in template.items():
            result[name.replace("resX", f"res{block}")] = values
    return result


def evaluate_candidate(output_tile_width: int, cpu_profile: str) -> dict[str, object]:
    encoder, residual, decoder = build_groups(output_tile_width)
    peak_line_buffers = max(
        encoder.line_buffer_blocks,
        residual.line_buffer_blocks,
        decoder.line_buffer_blocks,
    )

    cpu_ram_blocks = {
        # Official Titanium RAM estimator: 32 KiB x32 memory = 26 blocks;
        # independent 4 KiB I/D caches cost four blocks each.
        "onchip32k_plus_8k_caches_estimator": 34,
        # Isolated map result from the SC431 example configuration: Sapphire
        # Standard, 8 KiB on-chip RAM, 4 KiB I-cache, 4 KiB D-cache and the
        # required UART/SPI/I2C/APB/AXI control interfaces.  max_mult=0 removes
        # the four DSPs but does not change its measured 28 RAM blocks.
        "sapphire_sc431_standard_measured": 28,
        # Isolated 2026.1 map result for the project-specific Sapphire Lite:
        # 4 KiB on-chip RAM, 4 KiB I-cache, no D-cache/MULDIV/FPU, plus the
        # required AXI/APB/UART/SPI/I2C/timer/interrupt interfaces.  max_mult=0
        # is retained, so this profile uses no DSP blocks.
        "sapphire_lite_4k_i4_measured": 18,
        # Same control-plane interfaces with a 1 KiB I-cache.  The 4 KiB OCR
        # remains four byte-lane RAMs, while the smaller I-cache reduces the
        # isolated map from 18 to 15 RAM blocks.
        "sapphire_lite_4k_i1_measured": 15,
        # Same 1 KiB I-cache/4 KiB OCR configuration with only selected
        # Sapphire arrays annotated syn_ramstyle="registers".  The project-wide
        # small-memory threshold remains 64, so video FIFOs elsewhere in the
        # complete design are unaffected.  The balanced profile maps at 11 RAM
        # and routes at 8,390 XLR with WNS/WHS +2.950/+0.090 ns at 100 MHz.
        "sapphire_lite_4k_i1_local_balanced_measured": 11,
        # RAM-pressure profile additionally maps the 32x32 CPU register file
        # to logic.  Isolated full P&R measures 7 RAM, 10,987 XLR and
        # WNS/WHS +3.532/+0.033 ns at 100 MHz.  This is the current main
        # candidate because whole-system RAM, not CPU timing, is the binding
        # early resource constraint.
        "sapphire_lite_4k_i1_local_ram_saver_measured": 7,
        # Cacheless Lite with 2 KiB OCR.  It is a fallback/measurement profile,
        # not yet the frozen firmware choice: final boot/fault code size and
        # uncached DDR control latency still have to be measured.
        "sapphire_lite_2k_nocache_measured": 12,
    }[cpu_profile]
    ram_total = (
        V13_RAM_BLOCKS
        - V13_DEBUG_RAM_BLOCKS
        + Q17_ENGINE_RAM_BLOCKS
        + CNN_DMA_FIFO_RAM_BLOCKS
        + CNN_CONTROL_RAM_BLOCKS
        + peak_line_buffers
        + cpu_ram_blocks
    )
    port_aware_line_blocks = (
        PORT_AWARE_LINE_BUFFER_BLOCKS_TILE_LE_320
        if output_tile_width <= 320
        else peak_line_buffers
    )
    port_aware_total = ram_total - peak_line_buffers + port_aware_line_blocks
    area_downscaler_total = (
        port_aware_total - AREA_DOWNSCALER_CANDIDATE_SAVING_BLOCKS
    )
    area_downscaler_control_logic_total = (
        area_downscaler_total - CNN_CONTROL_RAM_BLOCKS
    )

    # Frozen logical traffic: RAW capture/read -> 720p RGB A/B -> encoder F0;
    # each residual reads the main source with halo, rereads the core skip and
    # writes the destination; decoder reads F and writes RGB C/D; display reads C/D.
    traffic = {
        "camera_raw_write": RAW_1080_BYTES,
        "preprocess_raw_read": RAW_1080_BYTES,
        "rgb_input_write": RGB_720_BYTES,
        "encoder_rgb_read": encoder.source_read_bytes,
        "encoder_f0_write": BOTTLENECK_BYTES,
        "residual_main_reads": 3 * residual.source_read_bytes,
        "residual_skip_reads": 3 * BOTTLENECK_BYTES,
        "residual_writes": 3 * BOTTLENECK_BYTES,
        "decoder_f_read": decoder.source_read_bytes,
        "rgb_output_write": RGB_720_BYTES,
        "display_rgb_read": RGB_720_BYTES,
    }
    traffic_per_frame = sum(traffic.values())
    traffic_per_second = traffic_per_frame * FPS
    # The bus transfers whole 128-bit beats, including RGB halo alignment
    # prefixes/suffixes discarded by read_keep. Keep the logical figures for
    # comparison, but use actual row-command bytes for the bandwidth gate.
    dma_streams = build_tile_dma_command_streams(output_tile_width)
    alignment_overhead = sum(
        command["beats"] * AXI_DATA_BYTES - command["payload_bytes"]
        for commands in dma_streams.values()
        for command in commands
    )
    bus_bytes_per_frame = traffic_per_frame + alignment_overhead
    bus_bytes_per_second = bus_bytes_per_frame * FPS
    required_measured_bandwidth = bus_bytes_per_second / DDR_ALLOWED_UTILIZATION

    overrides: dict[str, int] = {}
    overrides.update(encoder.operation_output_values)
    overrides.update(expand_residual_overrides(residual.operation_output_values))
    overrides.update(decoder.operation_output_values)
    q17 = next(candidate for candidate in throughput.CANDIDATES if candidate.name == "MUL18_Q17_112MAC_24RQ")
    cycle_result = throughput.evaluate(
        throughput.model_spec.build_model_b(), q17, output_value_overrides=overrides
    )
    layer_cycles = {
        layer["name"]: int(layer["estimated_cycles"])
        for layer in cycle_result["layers"]
    }
    group_cycle_names = {
        "encoder": tuple(encoder.operation_output_values),
        "residual": tuple(name.replace("resX", "res1") for name in residual.operation_output_values),
        "decoder": tuple(decoder.operation_output_values),
    }
    line_port_64bit = []
    for group in (encoder, residual, decoder):
        available_cycles = sum(layer_cycles[name] for name in group_cycle_names[group.name])
        worst_read_cycles = max(
            int(line_buffer["row_read_words_64bit"])
            for line_buffer in group.line_buffers
        )
        line_port_64bit.append(
            {
                "group": group.name,
                "available_group_compute_cycles": available_cycles,
                "worst_independent_row_bank_read_cycles": worst_read_cycles,
                "port_utilization_percent": 100.0 * worst_read_cycles / available_cycles,
                "prefetch_overlap_gate": worst_read_cycles <= available_cycles,
            }
        )

    return {
        "output_tile_width": output_tile_width,
        "cpu_profile": cpu_profile,
        "groups": [asdict(group) for group in (encoder, residual, decoder)],
        "peak_line_buffer_blocks": peak_line_buffers,
        "ram": {
            "v13_after_debug_removal": V13_RAM_BLOCKS - V13_DEBUG_RAM_BLOCKS,
            "q17_engine": Q17_ENGINE_RAM_BLOCKS,
            "cnn_dma_fifos": CNN_DMA_FIFO_RAM_BLOCKS,
            "cnn_control": CNN_CONTROL_RAM_BLOCKS,
            "cpu": cpu_ram_blocks,
            "line_buffer_peak": peak_line_buffers,
            "line_buffer_model": "10K_BIT_CAPACITY_LOWER_BOUND_ONLY",
            "total": ram_total,
            "percent": 100.0 * ram_total / DEVICE_RAM_BLOCKS,
            "target_85_blocks": RAM_TARGET_BLOCKS,
            "hard_limit_90_blocks": RAM_HARD_LIMIT_BLOCKS,
            "gate_85_percent": ram_total <= RAM_TARGET_BLOCKS,
            "gate_90_percent": ram_total <= RAM_HARD_LIMIT_BLOCKS,
            "scenarios": {
                "port_aware_current_scaler": {
                    "total": port_aware_total,
                    "percent": 100.0 * port_aware_total / DEVICE_RAM_BLOCKS,
                    "gate_85_percent": port_aware_total <= RAM_TARGET_BLOCKS,
                    "gate_90_percent": port_aware_total <= RAM_HARD_LIMIT_BLOCKS,
                    "evidence": "tile<=320 uses eight independent 64bitx512 row banks; Q17+line-buffer pressure P&R measured 52 RAM total.",
                },
                "port_aware_area_downscaler_candidate": {
                    "total": area_downscaler_total,
                    "percent": 100.0 * area_downscaler_total / DEVICE_RAM_BLOCKS,
                    "gate_85_percent": area_downscaler_total <= RAM_TARGET_BLOCKS,
                    "gate_90_percent": area_downscaler_total <= RAM_HARD_LIMIT_BLOCKS,
                    "evidence": "v14 full pipeline P&R measured 147 RAM versus v13 157 RAM; active rgb_area_downscaler_3to2 hierarchy uses six RAM blocks and passes strict STA.",
                },
                "area_downscaler_control_in_logic_candidate": {
                    "total": area_downscaler_control_logic_total,
                    "percent": 100.0 * area_downscaler_control_logic_total / DEVICE_RAM_BLOCKS,
                    "gate_85_percent": area_downscaler_control_logic_total <= RAM_TARGET_BLOCKS,
                    "gate_90_percent": area_downscaler_control_logic_total <= RAM_HARD_LIMIT_BLOCKS,
                    "evidence": "v17 integrated pressure mapping confirmed that moving the two-block small control table out of BRAM gives 212 RAM and 48096 XLR with the six-block net DMA reserve retained; effort-5 multi-seed P&R passed 3/4 seeds. v19 added the real 20-entry tile descriptor sequencer at 48058 XLR/212 RAM. v20 added the three-stream row-DMA address adapter at 48468 XLR/212 RAM; seed 1 and seed 2 both passed the +0.3ns internal gate at +0.383ns and +0.454ns. The RAM saving and address-controller coexistence are proven, but every frozen netlist still requires fresh multi-seed timing selection and path-directed physical review.",
                },
            },
        },
        "ddr": {
            "traffic_bytes_per_frame": traffic_per_frame,
            "traffic_breakdown": traffic,
            "payload_bytes_per_second_at_30fps": traffic_per_second,
            "axi_alignment_overhead_bytes_per_frame": alignment_overhead,
            "axi_bus_bytes_per_frame_lower_bound": bus_bytes_per_frame,
            "axi_bus_bytes_per_second_at_30fps_lower_bound": bus_bytes_per_second,
            "axi_bus_utilization_percent_of_peak": 100.0 * bus_bytes_per_second / DDR_PEAK_BYTES_PER_SECOND,
            "peak_bytes_per_second": DDR_PEAK_BYTES_PER_SECOND,
            "payload_utilization_percent_of_peak": 100.0 * traffic_per_second / DDR_PEAK_BYTES_PER_SECOND,
            "headroom_percent_of_peak": 100.0 * (1.0 - traffic_per_second / DDR_PEAK_BYTES_PER_SECOND),
            "required_measured_sustained_bytes_per_second_for_15pct_margin": required_measured_bandwidth,
            "required_measured_efficiency_percent_of_peak": 100.0
            * required_measured_bandwidth
            / DDR_PEAK_BYTES_PER_SECOND,
            "theoretical_gate_15pct_margin": bus_bytes_per_second
            <= DDR_PEAK_BYTES_PER_SECOND * DDR_ALLOWED_UTILIZATION,
            "warning": "仅为现有DMA总线载荷下界，含128bit对齐补字节；CPU、手势及最终同帧对比显示的附加流量尚未计入。Q027仍须板上混合读写微基准证明实际持续带宽、最坏服务间隔和FIFO零错误。",
        },
        "cycles": {
            "estimated_cycles": cycle_result["total_cycles"],
            "estimated_ms_at_150mhz": cycle_result["estimated_ms_at_150mhz"],
            "estimated_fps": cycle_result["estimated_fps"],
            "tile_mac_overhead_percent": cycle_result["tile_mac_overhead_percent"],
            "q133_30ms_gate": cycle_result["q133_early_gate_30ms"],
            "frame_30fps_gate": cycle_result["cycle_gate_30fps"],
            "warning": "仍未计DDR stall、层切换、行首尾气泡、残差相加和异常恢复。",
        },
        "line_port_64bit": {
            "groups": line_port_64bit,
            "all_groups_prefetch_overlap_gate": all(
                group["prefetch_overlap_gate"] for group in line_port_64bit
            ),
            "warning": "只证明每个活动行缓存拥有独立64bit读口、读预取可与同融合组MAC调度重叠时，端口拍数不超过该组理想计算拍数；尚未证明真实波前控制、FIFO深度或无空泡调度。",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path)
    parser.add_argument("--tile-manifest-json", type=Path)
    parser.add_argument("--tile-vectors", type=Path)
    parser.add_argument("--tile-dma-vectors-dir", type=Path)
    args = parser.parse_args()

    results = []
    for tile_width in (1280, 640, 320, 256):
        for cpu_profile in (
            "onchip32k_plus_8k_caches_estimator",
            "sapphire_sc431_standard_measured",
            "sapphire_lite_4k_i4_measured",
            "sapphire_lite_4k_i1_measured",
            "sapphire_lite_4k_i1_local_balanced_measured",
            "sapphire_lite_4k_i1_local_ram_saver_measured",
            "sapphire_lite_2k_nocache_measured",
        ):
            result = evaluate_candidate(tile_width, cpu_profile)
            results.append(result)
            print(
                f"tile={tile_width:4d} cpu={cpu_profile:31s} "
                f"RAM={result['ram']['total']:3d}/256 ({result['ram']['percent']:.2f}%) "
                f"ram85={'PASS' if result['ram']['gate_85_percent'] else 'FAIL'} "
                f"ram90={'PASS' if result['ram']['gate_90_percent'] else 'FAIL'} "
                f"DDR-bus-min={result['ddr']['axi_bus_utilization_percent_of_peak']:.2f}% "
                f"need={result['ddr']['required_measured_efficiency_percent_of_peak']:.2f}%peak "
                f"cycle={result['cycles']['estimated_ms_at_150mhz']:.3f}ms "
                f"30ms={'PASS' if result['cycles']['q133_30ms_gate'] else 'FAIL'}"
            )

    document = {
        "facts": {
            "ram_block_bits": RAM_BLOCK_BITS,
            "device_ram_blocks": DEVICE_RAM_BLOCKS,
            "ddr_peak_bytes_per_second": DDR_PEAK_BYTES_PER_SECOND,
            "ddr_allowed_utilization": DDR_ALLOWED_UTILIZATION,
        },
        "results": results,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.tile_manifest_json:
        args.tile_manifest_json.parent.mkdir(parents=True, exist_ok=True)
        args.tile_manifest_json.write_text(
            json.dumps(build_tile_manifest(320), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.tile_vectors:
        write_tile_descriptor_vectors(args.tile_vectors)
    if args.tile_dma_vectors_dir:
        write_tile_dma_vectors(args.tile_dma_vectors_dir)


if __name__ == "__main__":
    main()
