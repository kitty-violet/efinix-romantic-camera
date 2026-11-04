"""Read-only coverage audit of the unified review, never a hardware release.

Keep the historical issue IDs and known diagnostic gaps visible during the
video-service rewrite. No simulator, USB, process launch or production edits.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
REPORTS = ROOT / "output/toolchain"
BOARD = ROOT / "tmp/frame-execution/connected-board-continuous-srl-register1"
HERE = Path(__file__).resolve().parent
OLD = REPORTS / "连续运行黑屏与闪烁_全链路排查_2026-10-15.md"
NEW = REPORTS / "旧疑点与结构重构统一处理_2026-10-15.md"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(text, pattern):
    result = {}
    for line in text.splitlines():
        match = re.match(pattern, line)
        if match:
            key = match[1]
            if key in result:
                raise ValueError(f"Duplicate review identifier: {key}")
            fields = [x.strip() for x in line.strip().strip("|").split("|")]
            result[key] = fields
    return result


def audit():
    old = OLD.read_text(encoding="utf-8")
    new = NEW.read_text(encoding="utf-8")
    old_rows = rows(old, r"^\| (\d{2}) \|")
    new_rows = rows(new, r"^\| (\d{2}) \|")
    required = {f"{i:02d}" for i in range(1, 33)}
    if old_rows.keys() != required or new_rows.keys() != required:
        raise ValueError("The unified review must retain every one of the original 32 IDs")
    if any(len(row) != 4 or any(not field for field in row) for row in new_rows.values()):
        raise ValueError("Each issue needs a disposition and evidence/observation requirement")
    observations = rows(new, r"^\| (O\d{2}) ")
    if observations.keys() != {f"O{i:02d}" for i in range(1, 11)}:
        raise ValueError("Missing permanent diagnostic capability group")
    modes = ("TX_PATTERN", "DDR_PATTERN", "INPUT_PREVIEW", "FROZEN_A", "LIVE_A")
    if any(f"| {mode} |" not in new for mode in modes):
        raise ValueError("A required same-bitstream isolation mode is missing")

    # Bind actual historical evidence, rather than inferring HDL or board PASS
    # from the existence of a design-review paragraph.
    short_path = ROOT / "tmp/board-test/demo30-architecture-comparison1/short2/result.json"
    visual_path = ROOT / "tmp/board-test/demo10-1080p30-1/visual-result1.json"
    adjacent_path = ROOT / "tmp/board-test/video-global-audit-summary1.json"
    for path in (short_path, visual_path, adjacent_path):
        json.loads(path.read_text(encoding="utf-8-sig"))
    decoder_path = HERE / "read_cnn_observer_trim.py"
    decoder = decoder_path.read_text(encoding="utf-8")
    retired = ("relative_clock_counts_and_ticks", "reset_age_and_edges", "startup_trace")
    if "unavailable_diagnostics" not in decoder or any(x not in decoder for x in retired):
        raise ValueError("Re-audit the changed legacy diagnostic profile before trimming")
    apb_path = BOARD / "assembly/cnn_apb_control.sv"
    scheduler_path = BOARD / "assembly/hdmi_frame_scheduler.sv"
    ingress_path = BOARD / "assembly/raw_stream_ingress.sv"
    apb = apb_path.read_text(encoding="utf-8")
    scheduler = scheduler_path.read_text(encoding="utf-8")
    counters = ("displayed_frame_count", "blanked_frame_count", "underflow_count", "marker_error_count")
    if not all(re.search(r"output reg\s+\[31:0\]\s+" + name, scheduler) for name in counters):
        raise ValueError("Display source changed; review the real event definitions again")
    missing_apb_names = [name for name in counters if name not in apb]
    # This is a source-interface finding, NOT a proof of every possible alias
    # through the mapped design. Prior mapped/board audit remains separate.
    unconnected_raw_occupancy = ".wr_count()" in ingress_path.read_text(encoding="utf-8")

    files = [Path(__file__), OLD, NEW, short_path, visual_path, adjacent_path,
             decoder_path, apb_path, scheduler_path, ingress_path]
    return {
        "status": "REVIEW_COVERAGE_PASS_IMPLEMENTATION_PENDING",
        "issue_count": len(new_rows),
        "issues": [{"id": i, "historical_hypothesis": old_rows[i][1],
                    "current_disposition": new_rows[i][2],
                    "required_evidence": new_rows[i][3]} for i in sorted(required)],
        "permanent_observation_groups": observations,
        "runtime_isolation_modes_required_not_implemented": modes,
        "current_known_gaps": {
            "retired_diagnostics_not_zero_measurements": retired,
            "display_counters_without_direct_apb_named_connection": missing_apb_names,
            "raw_ingress_write_occupancy_output_unconnected": unconnected_raw_occupancy,
            "gap_scope": "Source interfaces plus linked historical audit; no new mapped connectivity proof",
        },
        "bindings": {str(path.relative_to(ROOT)).replace("\\", "/"): digest(path) for path in files},
        "production_rtl_modified": False,
        "diagnostics_implemented": False,
        "hardware_access": False,
        "board_release": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = args.report.resolve()
    if not report.is_relative_to(ROOT / "tmp") or report.exists():
        raise ValueError("Use a new report path under the project tmp directory")
    result = audit()
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ("status", "issue_count", "current_known_gaps",
                                           "production_rtl_modified", "hardware_access", "board_release")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
