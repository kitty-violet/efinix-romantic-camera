"""Map and audit the bounded-run parent plus exact writer/data timing changes."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from cnn_run_budget_board import BOARD as PARENT,audit_delta as parent_audit
from cnn_timing_closure_native import OUT as TRANSFORM
from native_local_drivers import graph,CELL
from complete_native_logic_split import resource_counts,protected_blocks
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary
BOARD=ROOT/'tmp/frame-execution/connected-board-cnn-timing-closure1'

def transform_check():
    r=json.loads((TRANSFORM/'result.json').read_text());assert r['status']=='CNN_TIMING_CLOSURE_JOINT_NATIVE_GRAPH_PASS'
    assert r['latency_cycles_added']==0 and r['clones']==307 and r['writer_luts']==156
    assert r['source_sha256']==sha(TRANSFORM/'native_source.v') and r['replications_sha256']==sha(TRANSFORM/'replications.json')
    for f,h in r['inputs'].items():assert sha(f)==h,f
    return r

def audit_delta(board):
    parent_audit(PARENT);transform_check();board=Path(board)
    m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
    assert r['status']=='CNN_TIMING_CLOSURE_FULL_BOARD_MAP_PASS'
    assert m['timing_parent_sha256']==sha(PARENT/'inputs.json') and m['timing_transform_sha256']==sha(TRANSFORM/'result.json')
    assert m['timing_builder_sha256']==sha(__file__) and r['inputs_sha256']==sha(board/'inputs.json')
    for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
    for f,h in r['evidence'].items():assert sha(f)==h,f
    assert sha(board/'outflow/ti60f225_oob.map.v')==r['map_sha256'] and sha(board/'outflow/ti60f225_oob.vdb')==r['vdb_sha256']
    return dict(status='CNN_TIMING_CLOSURE_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--map-existing',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    proof=transform_check();parent_audit(PARENT);b=BOARD
    if not(a.map_existing or a.audit_only):
        assert not b.exists();b.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
        for row in m['inputs']:
            src=Path(row['snapshot']);dest=b/src.relative_to(PARENT);dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(TRANSFORM/'native_source.v' if dest.name=='native_source.v' else src,dest)
            row.update(snapshot=str(dest),sha256=sha(dest))
        m.update(timing_parent_sha256=sha(PARENT/'inputs.json'),timing_transform_sha256=sha(TRANSFORM/'result.json'),timing_builder_sha256=sha(__file__),hardware_access=False)
        (b/'inputs.json').write_text(json.dumps(m,indent=2))
        for n in ('map.cmd','camera_graft.json','recovery_graft.json','capture_burst_graft.json'):shutil.copy2(PARENT/n,b/n)
        shutil.copy2(TRANSFORM/'replications.json',b/'replications.json')
    else:m=json.loads((b/'inputs.json').read_text())
    if a.prepare_only:print('PREPARED',b);return
    for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
    if a.audit_only:body=(b/'map.log').read_text();seconds=None
    else:
        assert not (b/'map.log').exists();body,seconds=run(['cmd.exe','/d','/c','map.cmd'],b,'map.log',480)
    assert re.search(r'map\s*:\s*PASS',body)
    source=(b/'native_source.v').read_text();mapped=(b/'outflow/ti60f225_oob.map.v').read_text()
    sg,mg=graph(source),graph(mapped);missing=set(sg)-set(mg)
    assert all(sg[n][0]=='EFX_LUT4' and not n.startswith('protected_inst/') for n in missing),sorted(missing)
    dead={t for n in missing for t in sg[n][1]['O']}
    assert not [(n,p,t) for n,(k,ps) in sg.items() if n not in missing for p,ts in ps.items() for t in ts if t in dead]
    clean=CELL.sub(lambda z:'' if z[2].lstrip('\\') in missing else z[0],source)
    conn=verify(flattened_public_ports(clean),flattened_public_ports(mapped));assert conn['added_protected']==0
    lex=lexical_roundtrip(clean,mapped);boundary=compare_boundary(source,mapped,conn['renamed_nets'])
    (b/'public_connectivity_audit.json').write_text(json.dumps(conn,indent=2))
    old=resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt');counts=resource_counts(b/'outflow/ti60f225_oob.map.rpt');expected=dict(old)
    for k,v in proof['total_added_kinds'].items():expected[k]=expected.get(k,0)+v
    expected['EFX_LUT4']-=len(missing);assert expected==counts,(expected,counts)
    assert protected_blocks(source)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
    assert counts['EFX_RAM10']+counts['EFX_DPRAM10']==235 and counts.get('EFX_DSP48',0)+counts.get('EFX_DSP24',0)//2==160
    assert '[EFX-0654' not in (b/'outflow/ti60f225_oob.warn.log').read_text()
    r=dict(status='CNN_TIMING_CLOSURE_FULL_BOARD_MAP_PASS',seconds=seconds,counts=counts,lexical_audit=lex,protected_boundary=boundary,
        dead_generated_luts=sorted(missing),added_cells=sorted(set(mg)-set(graph((PARENT/'outflow/ti60f225_oob.map.v').read_text()))),
        map_sha256=sha(b/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(b/'outflow/ti60f225_oob.vdb'),inputs_sha256=sha(b/'inputs.json'),
        evidence={str(f):sha(f) for f in (Path(__file__),TRANSFORM/'result.json',b/'public_connectivity_audit.json',b/'replications.json')},
        hardware_access=False,physical_pass=False,board_release=False)
    assert not (b/'result.json').exists();(b/'result.json').write_text(json.dumps(r,indent=2));print(r['status'],counts,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if BOARD.exists():(BOARD/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
