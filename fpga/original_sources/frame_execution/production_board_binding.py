"""Strict source/proof binding for the narrow production CNN partition."""
import json
from pathlib import Path
from run_units import sha


def validate(board, metadata, require_vendor=True):
    assert metadata['native_kind']=='production_cnn'
    native=Path(metadata['native']);nm=json.loads((native/'inputs.json').read_text())
    assert sha(native/'inputs.json')==metadata['native_inputs_sha256']
    assert sha(native/'production.dll')==metadata['native_dll_sha256']==nm['dll_sha256']
    assert sha(board/'cnn_production_frame_dpi.sv')==nm['wrapper_sha256']
    assert sha(native/'production.cpp')==nm['bridge_sha256']
    assert nm['hardware_changed'] is False and nm['lookahead'] is False
    byname={Path(r['snapshot']).name:r for r in metadata['inputs']}
    visibility='wire merge_busy,dma_end;wire compute_begin /*verilator public_flat_rd*/;wire compute_end /*verilator public_flat_rd*/;wire [14:0] debug_program_index /*verilator public_flat_rd*/;'
    original='wire merge_busy,compute_begin,compute_end,dma_end;wire [14:0] debug_program_index;'
    for row in nm['inputs']:
        f=Path(row['snapshot']);assert sha(f)==row['sha256']
        target=byname[f.name];assert sha(target['snapshot'])==target['sha256']==row['board_source_sha256']
        if f.name=='cnn_production_frame.sv':
            text=f.read_text();assert text.count(visibility)==1
            assert text.replace(visibility,original)==Path(target['snapshot']).read_text()
        else:assert row['sha256']==row['board_source_sha256']
    for header in (native/'source').glob('*.vh'):
        matches=[Path(r['snapshot']) for r in metadata['inputs'] if Path(r['snapshot']).name==header.name]
        assert matches and any(sha(path)==sha(header) for path in matches),header.name
    compiled=[Path(x).name for x in metadata['source_files']]
    assert 'cnn_production_frame.sv' not in compiled and compiled.count('cnn_production_frame_dpi.sv')==1
    required={
        'boundary':'PRODUCTION_DPI_EXTERNAL_BOUNDARY_EQUIVALENCE_PASS',
        'active':'PRODUCTION_ACTIVE_DPI_EXTERNAL_EQUIVALENCE_PASS',
        'unknown':'PRODUCTION_DPI_UNKNOWN_ACTIVE_R_REJECTION_PASS'}
    if require_vendor:required['vendor']='PRODUCTION_DPI_REAL_DDR_FOUR_MASTER_EQUIVALENCE_PASS'
    for name,status in required.items():
        item=metadata['partition_proofs'][name];path=Path(item['path'])
        assert sha(path)==item['sha256'];data=json.loads(path.read_text())
        assert data['status']==status and data['native_inputs_sha256']==sha(native/'inputs.json')
        assert sha(path.parent/'simulation.log')==data['log_sha256']
    return dict(kind='production_cnn',native_inputs_sha256=sha(native/'inputs.json'),
                native_dll_sha256=sha(native/'production.dll'),hardware_unchanged=True,
                four_state_periphery=True,two_state_cnn=True,lookahead=False,
                vendor_partition_regression=require_vendor)
