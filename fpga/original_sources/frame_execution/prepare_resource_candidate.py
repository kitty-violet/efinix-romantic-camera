"""Reproduce the selected measured board; never silently lose adopted overrides.

Preparing sources is not map, physical approval or a download operation.
"""
import argparse,json,subprocess,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_connected_system import manifest

def check_file(item):
    path=(ROOT/item['path']).resolve()
    assert path.is_relative_to(ROOT) and path.is_file(),path
    assert sha(path)==item['sha256'],f'Frozen evidence/source changed: {path}'
    return path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--profile',type=Path,default=HERE/'board_resource_candidate.json')
    p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.profile.read_text(encoding='utf8'))
    assert not data['board_download_allowed'] and not data['resource_fit']
    check_file(data['contract'])
    check_file(data['base_inputs']);check_file(data['selected_inputs'])
    result=json.loads(check_file(data['selected_result']).read_text())
    selected=(ROOT/data['selected_board']).resolve()
    for file,key in [('inputs.json','inputs_sha256'),('outflow/ti60f225_oob.map.v','map_sha256'),
                     ('outflow/ti60f225_oob.map.rpt','report_sha256'),('outflow/ti60f225_oob.warn.log','warning_sha256')]:
        assert sha(selected/file)==result[key],file
    if data.get('source_binding'):
        binding=json.loads(check_file(data['source_binding']).read_text())
        if binding['status']=='NO_AUDIO_SOURCE_BINDING_PASS':
            from audit_no_audio import verify_no_audio_binding
            verify_no_audio_binding(binding)
        elif binding['status']=='SHARED_WORD_ROUTE_SOURCE_CYCLE_BINDING_PASS':
            from audit_shared_word_route import verify_shared_word_binding
            verify_shared_word_binding(binding)
        elif binding['status']=='UNUSED_READER_SOURCE_CYCLE_BINDING_PASS':
            from audit_unused_reader import verify_unused_reader_binding
            verify_unused_reader_binding(binding)
        elif binding['status']=='COMPACT_RESULT_PATH_SOURCE_BINDING_PASS':
            from audit_compact_result_path import verify_compact_binding
            verify_compact_binding(binding)
        elif binding['status'] in ('RADIX3_DISPATCH_SOURCE_CYCLE_BINDING_PASS','DISPATCH_STORAGE_SOURCE_CYCLE_BINDING_PASS'):
            from audit_radix3_dispatch import verify_radix3_binding
            verify_radix3_binding(binding)
        elif binding['status']=='HEADER_RAM_RESOURCE_BINDING_PASS':
            from audit_header_ram import verify_header_ram
            verify_header_ram(binding)
        elif binding['status']=='FIXED_STORAGE_RESOURCE_BINDING_PASS':
            from audit_fixed_storage import verify_fixed_storage
            verify_fixed_storage(binding)
        elif binding['status']=='VALID_PAYLOAD_RESOURCE_BINDING_PASS':
            from audit_valid_payload_refactor import verify_valid_payload
            verify_valid_payload(binding)
        elif binding['status']=='FIXED_DATAPATH_RESOURCE_BINDING_PASS':
            from audit_fixed_datapath_resources import verify_fixed_datapath
            verify_fixed_datapath(binding)
        elif binding['status']=='BANK_PREFIX_SOURCE_CYCLE_BINDING_PASS':
            from audit_bank_prefix_history import verify_bank_prefix_binding
            verify_bank_prefix_binding(binding)
        elif binding['status']=='DIRECT_SRAM_SOURCE_CYCLE_BINDING_PASS':
            from audit_direct_sram_response import verify_direct_binding
            verify_direct_binding(binding)
        else:
            assert binding['status']=='CONTROL_PAYLOAD_SOURCE_CYCLE_BINDING_PASS'
        assert binding['inputs_sha256']==result['inputs_sha256']
        assert binding['map_sha256']==result['map_sha256']
        for gate in binding['gates']:
            assert sha(Path(gate['path'])/'result.json')==gate['result_sha256']
    overrides=[check_file(x) for x in data['overrides']]
    for gate in data['gates']:
        evidence=json.loads(check_file(gate).read_text())
        assert evidence['status']==gate['status'] and (gate['status'].endswith('_PASS') or gate['status']=='PASS_REAL_BOOT_COMPLETE')
        inputs=gate.get('input_manifest',gate['path'].replace('result.json','inputs.json'))
        assert sha(ROOT/inputs)==evidence['inputs_sha256']
    _,expected=manifest(selected)
    base=(ROOT/data['base']).resolve()
    _,base_names=manifest(base)
    merged=dict(base_names)
    for path in overrides:
        assert path.name in merged,path.name
        merged[path.name]=sha(path)
    assert merged==expected,'Profile does not reconstruct the measured input set'
    cmd=[sys.executable,str(HERE/'prepare_board_variant.py'),'--base',str(base),'--root',str(a.root)]
    for path in overrides:cmd.extend(['--override',str(path)])
    subprocess.run(cmd,check=True)
    _,actual=manifest(a.root.resolve())
    assert actual==expected
    output=dict(status='SELECTED_BOARD_SOURCE_REPRODUCTION_PASS',source_names=len(actual),
                profile_sha256=sha(a.profile),measured_board=str(selected),resource_fit=False,
                full_frame=False,board_download_allowed=False)
    (a.root/'reproduction.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':main()
