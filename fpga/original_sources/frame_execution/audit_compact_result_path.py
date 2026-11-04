"""Bind compact result-path trials to a complete, same-source board map."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from audit_resource_scope import read_board
from audit_control_payload_refactor import gate
from audit_unused_reader import verify_unused_reader_binding
from prepare_compact_writer import transform


def verify_compact_binding(binding):
    assert binding['status']=='COMPACT_RESULT_PATH_SOURCE_BINDING_PASS'
    before=read_board(Path(binding['reference_board']));after=read_board(Path(binding['board']))
    assert before['names'].keys()==after['names'].keys() and len(after['names'])==125
    changed={n for n in after['names'] if after['names'][n]!=before['names'][n]}
    assert changed in ({'column_write8.sv','compute_column_cluster.sv'},
                       {'column_write8.sv','compute_column_cluster.sv','pw_skip_fuse4.sv'})
    assert sorted(changed)==binding['changed_sources']
    assert not after['result']['new_undriven_signatures']
    for key in ('inputs_sha256','map_sha256'):assert binding[key]==after['result'][key]
    inherited=binding['inherited_binding'];assert sha(inherited['path'])==inherited['sha256']
    parent=json.loads(Path(inherited['path']).read_text());verify_unused_reader_binding(parent)
    assert Path(parent['board'])==Path(binding['reference_board'])
    assert (Path(binding['board'])/'assembly/compute_column_cluster.sv').read_text()==transform(
        (Path(binding['reference_board'])/'assembly/compute_column_cluster.sv').read_text())
    assert sha(HERE/'prepare_compact_writer.py')==binding['transform_sha256']
    roles={'writer','cluster','reference_cluster','guard','fourstate'}
    if 'pw_skip_fuse4.sv' in changed:
        roles|={'abort_native','abort_fourstate'}
        from prepare_shared_residual_return import transform as shared_return
        old_skip=Path(binding['reference_board'])/'assembly/pw_skip_fuse4.sv'
        new_skip=Path(binding['board'])/'assembly/pw_skip_fuse4.sv'
        assert new_skip.read_text()==shared_return(old_skip.read_text())
        assert sha(HERE/'prepare_shared_residual_return.py')==binding['skip_transform_sha256']
    assert {r['role'] for r in binding['gates']}==roles
    roots={}
    for record in binding['gates']:
        role=record['role'];root=Path(record['path']);roots[role]=root
        names,actual=gate(root,role in ('cluster','reference_cluster','abort_native'))
        assert record==dict(actual,role=role)
        expected=before['names'] if role=='reference_cluster' else after['names']
        common=names.keys() & expected.keys()
        assert common
        if role=='writer':
            # The four-state unit runner writes text snapshots on Windows.
            # Verify ORIGINAL bytes against the board, then permit only the
            # exact universal-newline round trip in these two snapshots.
            meta=json.loads((root/'inputs.json').read_text())
            for row in meta['inputs']:
                name=Path(row['snapshot']).name
                if name in common:
                    assert row['source_sha256']==expected[name]
                    assert sha(row['source'])==row['source_sha256']
                    assert Path(row['snapshot']).read_text()==Path(row['source']).read_text()
        else:
            assert all(names[n]==expected[n] for n in common),(role,common)
        if role in ('cluster','guard','fourstate','abort_native','abort_fourstate'):assert changed<=common
        if role in ('abort_native','abort_fourstate'):
            assert 'PASS RESIDUAL_ABORT_PROBE stages=5 tag_fault=1' in (root/'simulation.log').read_text()
            log=(root/'simulation.log').read_text()
            if role=='abort_native':
                assert re.search(r'PASS COMPUTE_CLUSTER .*blocks=24 .*mac_held=[1-9]\d* ',log)
            else:
                assert re.search(r'RESIDUAL_RETURN_HOLD checks=[1-9]\d*',log)
        if role=='writer':
            result=json.loads((root/'result.json').read_text())
            assert result['status']=='COMPACT_WRITER_FOUR_STATE_CYCLE_EQUIVALENCE_PASS'
            assert len(result['results'])==3
            for run in result['results']:assert sha(root/f"seed{run['seed']}.log")==run['log_sha256']
            original=Path(binding['reference_board'])/'assembly/column_write8.sv'
            assert (root/'column_write8_reference.sv').read_text()==original.read_text().replace(
                'module column_write8 #','module column_write8_reference #',1)
    blocks=lambda root:re.findall(r'^BLOCK .*$',(root/'simulation.log').read_text(),re.M)
    b0,b1=blocks(roots['reference_cluster']),blocks(roots['cluster']);assert len(b0)==len(b1)==24
    deltas=[]
    for old,new in zip(b0,b1):
        oa=dict(re.findall(r'(\w+)=(\d+)',old));na=dict(re.findall(r'(\w+)=(\d+)',new))
        assert all(oa[k]==na[k] for k in oa if k not in ('cycles','last_command'))
        deltas.append(dict(case=int(na['case']),random=int(na['random']),before=int(oa['cycles']),after=int(na['cycles']),
                           delta=int(na['cycles'])-int(oa['cycles'])))
    assert deltas==binding['block_cycles']
    if 'pw_skip_fuse4.sv' not in changed:assert b0==b1
    vectors=[]
    for role in ('cluster','reference_cluster'):
        meta=json.loads((roots[role]/'frozen/inputs.json').read_text());vectors.append(meta['vectors'])
        for name,digest in meta['vectors'].items():assert sha(roots[role]/'frozen/vectors'/name)==digest
    assert vectors[0]==vectors[1]
    return before,after


def main():
    p=argparse.ArgumentParser()
    for n in ('board','reference-board','writer','cluster','reference-cluster','guard','fourstate'):
        p.add_argument('--'+n,type=Path,required=True)
    for n in ('abort-native','abort-fourstate'):p.add_argument('--'+n,type=Path)
    a=p.parse_args();board=a.board.resolve();reference=a.reference_board.resolve()
    assert board.is_relative_to(ROOT/'tmp') and not (board/'gate_binding.json').exists()
    before,after=read_board(reference),read_board(board)
    evidence=[]
    for role in ('writer','cluster','reference_cluster','guard','fourstate','abort_native','abort_fourstate'):
        if getattr(a,role) is None:continue
        _,record=gate(getattr(a,role).resolve(),role in ('cluster','reference_cluster','abort_native'));evidence.append(dict(record,role=role))
    blocks=lambda root:[dict(re.findall(r'(\w+)=(\d+)',s)) for s in re.findall(r'^BLOCK .*$',(root/'simulation.log').read_text(),re.M)]
    pairs=zip(blocks(a.reference_cluster),blocks(a.cluster))
    cycles=[dict(case=int(n['case']),random=int(n['random']),before=int(o['cycles']),after=int(n['cycles']),delta=int(n['cycles'])-int(o['cycles'])) for o,n in pairs]
    binding=dict(status='COMPACT_RESULT_PATH_SOURCE_BINDING_PASS',board=str(board),reference_board=str(reference),
        inputs_sha256=after['result']['inputs_sha256'],map_sha256=after['result']['map_sha256'],source_names=125,
        changed_sources=sorted(n for n in after['names'] if after['names'][n]!=before['names'][n]),
        inherited_binding=dict(path=str(reference/'gate_binding.json'),sha256=sha(reference/'gate_binding.json')),
        transform_sha256=sha(HERE/'prepare_compact_writer.py'),gates=evidence,block_cycles=cycles,
        frame_ms=None,resource_fit=False,physical_timing_pass=False,board_download_allowed=False)
    if 'pw_skip_fuse4.sv' in binding['changed_sources']:
        binding['skip_transform_sha256']=sha(HERE/'prepare_shared_residual_return.py')
    verify_compact_binding(binding)
    binding['resource_delta']={k:v-before['result']['resources'].get(k,0) for k,v in after['result']['resources'].items()}
    (board/'gate_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
    print(json.dumps(dict(status=binding['status'],delta=binding['resource_delta'],block_cycles=cycles)))


if __name__=='__main__':main()
