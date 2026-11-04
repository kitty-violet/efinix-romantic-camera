"""Reject a research graph against the actual frozen board's B1-only ABI."""
import json
from pathlib import Path
from .runtime import configure_workspace, file_hash, write_json


def main():
    out=configure_workspace()/'runs/sampling-hardware-audit-20261003'
    frozen=json.loads((out/'inputs.json').read_text())
    board=Path(frozen['selected_board']);src=board/'assembly'
    allocation=json.loads((out/'fullres-c8-probe/allocation.json').read_text())
    paths=[src/n for n in ('cnn_production_frame.sv','frame_program_fetch.sv',
                          'frame_program_issue.sv','coefficient_bank_store_fast.sv')]
    expected={r['path']:r['sha256'] for r in frozen['all_frozen_sources']}
    records=[]
    for p in paths:
        assert file_hash(p)==expected[str(p)]
        records.append(dict(path=str(p),sha256=file_hash(p)))
    frame,fetch,issue,coeff=[p.read_text() for p in paths]
    requirements=[
        (frame,".compute_records(14'd8789)", 'Top-level compute record count fixed at 8789; candidate requires 14400.'),
        (frame,".dma_records(14'd10440)", 'Top-level DMA count fixed at 10440; candidate requires 11520.'),
        (fetch,'MAX_COMPUTE=8789,MAX_DMA=10440', 'Fetch limits reject candidate counts.'),
        (issue,"nodes_completed==29'h1fffffff", 'Frame publication requires all 29 B1 nodes; candidate has 10 actual nodes.'),
        (issue,'d2!=256', 'Final RGB stores require 256-pixel stripes; candidate finite allocation uses 160.'),
        (issue,"rgb_x<=rgb_x+11'd256", 'Frame completion tracks fixed 256-pixel stripes.'),
        (issue,'cdata[28:14]>=19229', 'Global command-index guard rejects part of candidate 25920-command stream.'),
        (coeff,"begin_image_bytes != 32'd17792", 'Coefficient image length is fixed: use a validated compatible padded image or version the loader.')]
    for body,needle,_ in requirements:assert needle in body,needle
    compute=allocation['commands']['compute'];dma=allocation['commands']['load']+allocation['commands']['store']
    assert compute==14400 and dma==11520
    result=dict(status='CURRENT_FROZEN_BOARD_ABI_REJECTS_CANDIDATE',
                new_arithmetic_operation_required=False,arithmetic_compatibility='operation types only; trained INT8 not validated',
                pure_weight_replacement=False,pure_command_replacement=False,
                required_control_adaptations=[r[2] for r in requirements],
                source_records=records,candidate_compute=compute,candidate_dma=dma,
                descriptor_count_bits_fit=compute<2**14 and dma<2**14,
                ordinal_15bit_fit=compute+dma<2**15,
                compressed_descriptor_bytes_if_existing_abi_reused=compute*64+dma*16,
                coefficient_format_note='Padding bytes is not a production export; new offsets, quantization and validation still required.',
                decision='Do not edit hardware for this visually unaccepted candidate. No resource/timing guarantee.',
                program_fp32_image_quality_pass=False,resource_fit=False,physical_timing_pass=False)
    write_json(out/'fullres-c8-probe/board_abi.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
