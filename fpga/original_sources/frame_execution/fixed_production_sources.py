"""One source selection for fixed-window production simulation and board map."""
from pathlib import Path
import os
from run_units import HERE
from run_units import ROOT
from fixed_variant_overlay import resolve_overlay

def sources():
    folder=HERE/'variants/fixed_window_dot8x32'
    names=('fixed_video_compute','fixed_production_mac','fixed_source_columns6',
           'fixed_window_dot8x32','fixed_ring_weights9','fixed_partial32',
           'fixed_rq32_mixed','fixed_skip_vectors4','fixed_column_store4','fixed_weight_loader')
    files=[HERE/'variants/fixed_production_cluster/row_pair_compute_cluster.sv',
            HERE/'variants/row_pair_taps_fixed/row_pair_aux_router.sv',
            *[folder/(name+'.sv') for name in names],
            ROOT/'tmp/frame-execution/row-pair-taps-fixed-production-short1/source/requantize_i32_mul18_q17_pipeline.sv']
    if os.environ.get('FPGA_FIXED_OVERLAY'):
        replacements=resolve_overlay(os.environ['FPGA_FIXED_OVERLAY'])
        pipeline={'cnn_frame_datapath.sv','cnn_production_frame.sv','coefficient_bank_store_fast.sv','row_pair_dma.sv','frame_program_fetch.sv','frame_program_issue.sv','row_pair_arena6.sv'}
        assert replacements and set(replacements)<=set(f.name for f in files)|pipeline, 'Unknown fixed production override'
        files=[replacements.get(f.name,f) for f in files]
        files.extend(replacements[n] for n in sorted(pipeline & set(replacements)))
    return files

def instrument(body):
    # Replace only obsolete architecture counters. All actual RAM-write tensor,
    # descriptor, DDR-response, final RGB and CRC assertions remain unchanged.
    start=body.index('  if(mactive && system.core.dut.state==2 && hbusy)')
    stop=body.index('  if(!reset && !flush)for(integer p=0;p<6;p=p+1)',start)
    profile=(HERE/'variants/fixed_production_cluster/production_profile.svh').read_text()
    return body[:start]+profile+body[stop:]
