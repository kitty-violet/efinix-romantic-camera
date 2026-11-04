"""Bind optional reader refactor and actual CPU peripheral removal to whole map."""
import argparse, json, re
from pathlib import Path
from run_units import ROOT, HERE, sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_shared_word_route import verify_shared_word_binding
from prune_unused_sapphire import prune


def record(root, role):
    result=json.loads((root/'result.json').read_text())
    source=root/('native_inputs.json' if role in ('cluster','reference_cluster') else 'inputs.json')
    assert sha(source)==result['inputs_sha256']
    return dict(role=role,path=str(root),result_sha256=sha(root/'result.json'),inputs_sha256=sha(source))


def verify_unused_reader_binding(binding):
    assert binding['status']=='UNUSED_READER_SOURCE_CYCLE_BINDING_PASS'
    before=read_board(Path(binding['reference_board']));after=read_board(Path(binding['board']))
    assert before['names'].keys()==after['names'].keys() and len(after['names'])==125
    changed={n for n in after['names'] if after['names'][n]!=before['names'][n]}
    assert changed in ({'EfxSapphireSoc.v'}, {'EfxSapphireSoc.v','unified_column_reader.sv'})
    assert sorted(changed)==binding['changed_sources']
    assert not after['result']['new_undriven_signatures']
    for key in ('inputs_sha256','map_sha256'):assert binding[key]==after['result'][key]
    inherited=binding['inherited_binding'];assert sha(inherited['path'])==inherited['sha256']
    prev=json.loads(Path(inherited['path']).read_text());verify_shared_word_binding(prev)
    assert Path(prev['board'])==Path(binding['reference_board'])
    old_cpu=Path(binding['reference_board'])/'assembly/EfxSapphireSoc.v'
    new_cpu=Path(binding['board'])/'assembly/EfxSapphireSoc.v'
    assert new_cpu.read_text()==prune(old_cpu.read_text())
    assert sha(HERE/'prune_unused_sapphire.py')==binding['transform_sha256']
    roles={'cpu','decoder'}
    if 'unified_column_reader.sv' in changed:roles|={'reader','cluster','reference_cluster','guard'}
    assert {g['role'] for g in binding['gates']}==roles
    roots={}
    for g in binding['gates']:
        role=g['role'];root=Path(g['path']);roots[role]=root
        assert record(root,role)==g
        result=json.loads((root/'result.json').read_text())
        assert sha(root/'simulation.log')==result['log_sha256']
        if role=='cpu':
            assert result['status']=='PASS_REAL_BOOT_COMPLETE'
            data=json.loads((root/'inputs.json').read_text());q=data['cpu_qualification']
            assert q['board_sha256']==after['names']['EfxSapphireSoc.v']
            assert q['reference_sha256']==before['names']['EfxSapphireSoc.v']
            assert q['transform_sha256']==binding['transform_sha256']
            assert q['public_wrapper_equal'] and q['encrypted_cpu_text_unchanged']
            assert sha(root/'tb_real_boot.sv')==data['tb_sha256']
            assert sha(root/'boot_vectors.vh')==data['fixture_sha256']
            for x in data['inputs']:
                f=Path(x['snapshot']);assert sha(f)==x['sha256']
                if f.name=='EfxSapphireSoc.v':
                    public=lambda t:t.split('module EfxCPUSp1')[0].replace('(* syn_ramstyle = "registers" *) ','')
                    assert public(f.read_text())==public(new_cpu.read_text())
                elif f.name in after['names']:assert x['sha256']==after['names'][f.name]
            assert data['image']['sha256']=='34d9c2b09e37cfc9d34219302be4fb058478b334518ad1ded0602b71335a6085'
            continue
        names,_=gate(root,role in ('cluster','reference_cluster'))
        expected=before['names'] if role=='reference_cluster' else after['names']
        common=names.keys() & expected.keys();assert common
        assert all(names[n]==expected[n] for n in common),(role,common)
        if role=='decoder':
            assert names['EfxSapphireSoc.v']==after['names']['EfxSapphireSoc.v']
            module=re.search(r'(?ms)^module BmbDecoder_3 \(.*?^endmodule',new_cpu.read_text()).group()
            assert (root/'decoder.v').read_text()==module+'\n'
            assert result['status']=='UNUSED_PERIPHERAL_DECODER_FOUR_STATE_PASS'
        else:
            assert 'unified_column_reader.sv' in common
            if role=='reader':
                assert result['status']=='READER_SELECT_FOUR_STATE_CYCLE_PASS'
                assert result['reference_sha256']==before['names']['unified_column_reader.sv']
                assert result['candidate_sha256']==after['names']['unified_column_reader.sv']
    if 'cluster' in roots:
        blocks=lambda r:re.findall(r'^BLOCK .*$',(r/'simulation.log').read_text(),re.M)
        assert len(blocks(roots['cluster']))==24
        assert blocks(roots['cluster'])==blocks(roots['reference_cluster'])
        vectors=[]
        for role in ('cluster','reference_cluster'):
            source=json.loads((roots[role]/'frozen/inputs.json').read_text());vectors.append(source['vectors'])
            for name,digest in source['vectors'].items():assert sha(roots[role]/'frozen/vectors'/name)==digest
        assert vectors[0]==vectors[1]
    return before,after


def main():
    p=argparse.ArgumentParser()
    for n in ('board','reference-board','cpu','decoder'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('reader','cluster','reference-cluster','guard'):p.add_argument('--'+n,type=Path)
    a=p.parse_args();board=a.board.resolve();reference=a.reference_board.resolve()
    out=board/'gate_binding.json';assert board.is_relative_to(ROOT/'tmp') and not out.exists()
    before,after=read_board(reference),read_board(board)
    changed=sorted(n for n in after['names'] if after['names'][n]!=before['names'][n])
    gates=[record(getattr(a,role).resolve(),role) for role in ('cpu','decoder','reader','cluster','reference_cluster','guard') if getattr(a,role)]
    binding=dict(status='UNUSED_READER_SOURCE_CYCLE_BINDING_PASS',board=str(board),reference_board=str(reference),
        inputs_sha256=after['result']['inputs_sha256'],map_sha256=after['result']['map_sha256'],
        source_names=125,changed_sources=changed,unchanged_sources=125-len(changed),
        inherited_binding=dict(path=str(reference/'gate_binding.json'),sha256=sha(reference/'gate_binding.json')),
        transform_sha256=sha(HERE/'prune_unused_sapphire.py'),gates=gates,
        frame_ms=None,physical_timing_pass=False,resource_fit=False,board_download_allowed=False)
    verify_unused_reader_binding(binding)
    binding['resource_delta']={k:v-before['result']['resources'].get(k,0) for k,v in after['result']['resources'].items()}
    out.write_text(json.dumps(binding,indent=2)+'\n');print(json.dumps(binding['resource_delta']))


if __name__=='__main__':main()
