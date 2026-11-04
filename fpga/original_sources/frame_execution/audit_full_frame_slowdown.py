"""Explain the measured frame using frozen commands, RTL and exclusive counters.

This is an evidence audit, not a new simulation or a throughput prediction.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit():
    run = ROOT / "tmp/frame-execution/fourway-full-native3"
    vec = ROOT / "tmp/frame-execution/fourway-full-vectors2"
    old_path = ROOT / "output/toolchain/frame_optimization_analysis_2026-09-30.json"
    result, inputs, manifest, old = map(read, (
        run / "result.json", run / "inputs.json", vec / "frame_manifest.json", old_path))
    assert result["inputs_sha256"] == sha(run / "inputs.json")
    assert result["log_sha256"] == sha(run / "simulation.log")
    assert inputs["vector_manifest_sha256"] == sha(vec / "frame_manifest.json")
    for name in ("program.json", "program.hex"):
        assert manifest["files"][name] == sha(vec / name)
    for source in inputs["inputs"]:
        assert sha(Path(source["snapshot"])) == source["sha256"]
    sources = {Path(s["snapshot"]).name: Path(s["snapshot"]) for s in inputs["inputs"]}
    assert "localparam integer R=o/8,P=o%8;" in sources["pw_column_dispatch32.sv"].read_text()
    assert "One store read may be in flight" in sources["frame_arena_row_dma.sv"].read_text()
    tb = sources["tb_fourway_full.sv"].read_text()
    assert "if(field(0)==2)run_compute();else run_transfer();" in tb
    assert tb.index("system_result=$system(verify_command)") < tb.index("mbegin=1;")
    program = read(vec / "program.json")
    packed = [int(line, 16) for line in (vec / "program.hex").read_text().splitlines()]
    assert len(program) == len(packed) == result["commands"]
    node_info = {n["id"]: n for n in manifest["nodes"]}
    rows, dma = {}, defaultdict(Counter)
    previous_header = None
    header_counts, unique_headers = Counter(), {}
    header_words = 0
    for command, encoded in zip(program, packed):
        field = lambda index: (encoded >> (32 * index)) & 0xffffffff
        if command["kind"] != "compute":
            row = command["row"]
            part = dma[f'{command["group"]}_{command["kind"]}']
            part["commands"] += 1
            part["arena_words"] += row["width"] * row["groups"]
            part["payload_bytes"] += row["width"] * (3 if command["channels"] == 3 else row["groups"] * 8)
            continue
        assert field(0) == 2 and field(2) == command["pixels"] and field(3) == command["rows"]
        assert field(5) == command["output_groups"]
        name = command["node"]
        stat = rows.setdefault(name, dict(commands=0, rows_histogram=Counter(), width_histogram=Counter(),
                                         output_pixels=0, output_words=0, four_row_slots=0, header_reloads=0))
        stat["commands"] += 1
        stat["rows_histogram"][command["rows"]] += 1
        stat["width_histogram"][command["pixels"]] += 1
        stat["output_pixels"] += command["pixels"] * command["rows"]
        stat["output_words"] += command["pixels"] * command["rows"] * command["output_groups"]
        stat["four_row_slots"] += 4 * command["pixels"]
        if field(17) != previous_header:
            previous_header = field(17)
            header_counts[command["output_groups"]] += 1
            words = 8 * command["output_groups"] + ((field(19) % 8 + command["output_groups"] + 7) // 8)
            header_words += words
            unique_headers[previous_header] = words
            stat["header_reloads"] += 1
    for name, stat in rows.items():
        shape = node_info[name]["shape"]
        stat["logical_pixels"] = shape[0] * shape[1]
        stat["halo_pixel_overhead_percent"] = 100 * (stat["output_pixels"] / stat["logical_pixels"] - 1)
        stat["row_slot_utilization_percent"] = 100 * stat["output_pixels"] / stat["four_row_slots"]
        stat["row_slot_scope"] = ("geometry only; NOT measured MAC duty cycle; stride2 uses its own 18-owner layout")
    nodes = {n["id"]: n for n in result["nodes"]}
    assert sum(s["output_words"] for s in rows.values()) == sum(n["words"] for n in nodes.values()) == 8231400
    old_by_name = {g["name"]: g["measured_partition_cycles"] for g in old["groups"]}
    partitions = [
        ("残差PW及加法", [6, 9, 11, 14, 16, 19], ["残差块PW", "残差相加/重标定"]),
        ("残差DW", [5, 7, 10, 12, 15, 17], ["残差块DW"]),
        ("解码DW", [20, 23, 24, 27], ["解码DW"]),
        ("其他PW", [2, 4, 21, 25], ["其他PW"]),
        ("首层/下采样/RGB输出", [0, 1, 3, 28], ["RGB首层", "两层下采样DW", "RGB输出"]),
    ]
    bridge = []
    for name, new_nodes, old_groups in partitions:
        before = sum(old_by_name[g] for g in old_groups)
        after = sum(nodes[n]["cycles_with_header_control"] for n in new_nodes)
        bridge.append(dict(name=name, old_partition_cycles=before, full_cycles=after,
                           delta_cycles=after-before, old_ms=before/150000, full_ms=after/150000,
                           delta_ms=(after-before)/150000))
    remainder = result["cycles"] - sum(b["full_cycles"] for b in bridge)
    bridge.append(dict(name="DMA及帧收尾（旧账未计）", old_partition_cycles=0, full_cycles=remainder,
                       delta_cycles=remainder, old_ms=0, full_ms=remainder/150000, delta_ms=remainder/150000))
    assert sum(b["full_cycles"] for b in bridge) == result["cycles"]
    assert sum(b["old_partition_cycles"] for b in bridge) == 5175749
    reasons = {r["id"]: r["cycles"] for r in result["reasons"]}
    assert sum(reasons.values()) == result["cycles"]
    arena_words = sum(p["arena_words"] for p in dma.values())
    assert arena_words == 2814480
    execution = sum(reasons[i] for i in (1, 2, 3))
    header_reload_count = sum(header_counts.values())
    compute_count = sum(s["commands"] for s in rows.values())
    # Frozen profile0: sequential ISSUE/WAIT reads take three clocks per word;
    # two reload handshake clocks and one pool-config clock per compute command.
    assert 3 * header_words + 2 * header_reload_count + compute_count == reasons[6]
    return dict(
        status="FROZEN_FRAME_SLOWDOWN_ACCOUNTING_PASS", measured_cycles=result["cycles"],
        frame_ms_at_target_150mhz=result["measured_frame_ms"],
        old_mixed_partition_ms=5175749/150000, difference_ms=(result["cycles"]-5175749)/150000,
        bridge=bridge, nodes=rows, dma=dict(dma),
        dma_analysis=dict(measured_cycles=reasons[5], measured_ms=reasons[5]/150000,
                          words=arena_words, one_word_issue_floor_ms=arena_words/150000,
                          floor_share_percent=100*arena_words/reasons[5],
                          cycles_above_issue_floor=reasons[5]-arena_words,
                          ideal_axi_beat_service_ms=(int(result["line"][5])+int(result["line"][6]))/150000,
                          scope="one current arena request/cycle; AXI bound assumes same 150MHz and one beat/cycle, NOT physical DDR performance"),
        header_analysis=dict(reloads=header_reload_count, reloads_by_groups=dict(header_counts),
                             resident_words_read=header_words, unique_headers=len(unique_headers),
                             unique_header_words=sum(unique_headers.values()),
                             exact_cycles_equation=dict(word_reads=3*header_words,
                                                        reload_handshakes=2*header_reload_count,
                                                        pool_config=compute_count,
                                                        sum=reasons[6]),
                             measured_header_and_pool_config_ms=reasons[6]/150000,
                             overlap=False),
        arithmetic_stage_including_feed_and_drain_ms=execution/150000,
        arithmetic_stage_excess_over_30fps_ms=execution/150000-1000/30,
        full_required_speedup=result["cycles"]/5000000,
        evidence=[dict(path=str(p),sha256=sha(p)) for p in (
            run/"result.json",run/"inputs.json",vec/"program.json",vec/"program.hex",old_path,
            sources["tb_fourway_full.sv"],sources["pw_column_dispatch32.sv"],
            sources["frame_arena_row_dma.sv"],sources["next_header_prefetch.sv"],Path(__file__))],
        limitations=[
            "Old vs new is a scope/schedule comparison, NOT matched full-frame A/B slowdown.",
            "Per-node costs already include headers/config; never add the 5.304480ms again to the bridge.",
            "Row capacity is a geometric bound; port and weight stalls not separately attributed in the frozen run.",
            "No new RTL, simulation, map, P&R or board test in this analysis.",
            "One diagnostic CNN frame, not full baseline or measured steady-state AI fps.",
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    result = audit()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status","difference_ms","bridge","dma_analysis","header_analysis")}, ensure_ascii=False))
