"""Map only the verified continuous-control delta into the released board."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from continuous_control_native import PARENT,DEST as TRANSFORM,VARIANT,connections
from cnn_sparse_control_board import audit_delta as parent_audit
from native_local_drivers import graph,CELL
from complete_native_logic_split import resource_counts,protected_blocks
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary

BOARD=ROOT/'tmp/frame-execution/connected-board-continuous-control1'
BOUNDARY=ROOT/'tmp/frame-execution/preview-boundary-short7'
UNIT=ROOT/'tmp/frame-execution/continuous-control-short2'

def transform_check():
    t=json.loads((TRANSFORM/'result.json').read_text())
    assert t['status']=='PREVIEW_NATIVE_CANDIDATE_BOUNDARY_EXPORTED_NOT_VERIFIED'
    assert t['source_sha256']==sha(TRANSFORM/'native_source.v')
    assert t['boundary_sha256']==sha(TRANSFORM/'boundary.json')
    for f,h in t['inputs'].items():assert sha(f)==h,f
    for root,status in [(BOUNDARY,'ORIGINAL_MAPPED_CONTROLLER_EXPORT_MEANINGS_PASS'),
                        (UNIT,'CONTINUOUS_PREVIEW_CONTROL_SHORT_PASS')]:
        r=json.loads((root/'result.json').read_text());assert r['status']==status
        assert r['inputs_sha256']==sha(root/'inputs.json')
        for f,h in json.loads((root/'inputs.json').read_text()).items():assert sha(f)==h,f
    r=json.loads((BOUNDARY/'result.json').read_text())
    assert r['inputs_sha256']==sha(BOUNDARY/'inputs.json')
    for f,h in json.loads((BOUNDARY/'inputs.json').read_text()).items():assert sha(f)==h,f
    cut=json.loads((TRANSFORM/'expanded_cut_audit.json').read_text())
    assert cut['status']=='EXPANDED_NATIVE_CUT_PASS' and not cut['missing']
    for f,h in cut['inputs'].items():assert sha(f)==h,f
    return t

def audit_delta(board):
    parent_audit(PARENT);transform_check();board=Path(board)
    m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
    assert r['status']=='CONTINUOUS_CONTROL_FULL_BOARD_MAP_PASS'
    assert m['continuous_parent_sha256']==sha(PARENT/'inputs.json')
    assert m['continuous_transform_sha256']==sha(TRANSFORM/'result.json')
    assert m['continuous_builder_sha256']==sha(__file__)
    assert r['inputs_sha256']==sha(board/'inputs.json')
    for row in m['inputs']:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
    for f,h in r['evidence'].items():assert sha(f)==h,f
    assert sha(board/'outflow/ti60f225_oob.map.v')==r['map_sha256']
    assert sha(board/'outflow/ti60f225_oob.vdb')==r['vdb_sha256']
    return dict(status='CONTINUOUS_CONTROL_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-only',action='store_true');p.add_argument('--map-existing',action='store_true');p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    transform_check();parent_audit(PARENT);b=BOARD
    if not(a.map_existing or a.audit_only):
        assert not b.exists();b.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
        for row in m['inputs']:
            src=Path(row['snapshot']);dst=b/src.relative_to(PARENT);dst.parent.mkdir(parents=True,exist_ok=True)
            replacement=TRANSFORM/'native_source.v' if dst.name=='native_source.v' else VARIANT/'cnn_frame_admission.sv' if dst.name=='cnn_frame_admission.sv' else src
            shutil.copy2(replacement,dst);row.update(snapshot=str(dst),sha256=sha(dst))
        module=TRANSFORM/'cnn_preview_boot.sv';shutil.copy2(module,b/module.name)
        m['inputs'].append(dict(source=str(module),snapshot=str(b/module.name),sha256=sha(module)))
        xml=b/'ti60f225_oob.xml';s=xml.read_text();anchor='<efx:design_file name="native_source.v" version="default" library="default" />';assert s.count(anchor)==1
        xml.write_text(s.replace(anchor,anchor+'\n<efx:design_file name="cnn_preview_boot.sv" version="default" library="default" />'),encoding='utf8')
        for row in m['inputs']:row['sha256']=sha(row['snapshot'])
        m.update(continuous_parent_sha256=sha(PARENT/'inputs.json'),continuous_transform_sha256=sha(TRANSFORM/'result.json'),continuous_builder_sha256=sha(__file__),hardware_access=False)
        (b/'inputs.json').write_text(json.dumps(m,indent=2))
        for n in ['map.cmd','camera_graft.json','recovery_graft.json','capture_burst_graft.json']:shutil.copy2(PARENT/n,b/n)
    else:m=json.loads((b/'inputs.json').read_text())
    if a.prepare_only:print('PREPARED',b,flush=True);return
    for row in m['inputs']:assert sha(row['snapshot'])==row['sha256']
    if a.audit_only:log=(b/'map.log').read_text();sec=None
    else:
        assert not (b/'map.log').exists();log,sec=run(['cmd.exe','/d','/c','map.cmd'],b,'map.log',480)
    assert re.search(r'map\s*:\s*PASS',log)
    source=(b/'native_source.v').read_text();mapped=(b/'outflow/ti60f225_oob.map.v').read_text()
    sg,mg=graph(source),graph(mapped);missing=set(sg)-set(mg)
    assert not missing,sorted(missing)
    added=set(mg)-set(sg)
    filtered=CELL.sub(lambda z:z[0] if z[2].lstrip('\\') in sg else '',mapped)
    conn=verify(flattened_public_ports(source),flattened_public_ports(filtered))
    lex=lexical_roundtrip(source,filtered)
    boundary=compare_boundary(source,mapped,conn['renamed_nets'])
    (b/'public_connectivity_audit.json').write_text(json.dumps(conn,indent=2))
    assert protected_blocks(source)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
    counts=resource_counts(b/'outflow/ti60f225_oob.map.rpt')
    assert counts['EFX_RAM10']+counts['EFX_DPRAM10']==235
    assert counts.get('EFX_DSP48',0)+counts.get('EFX_DSP24',0)//2==160
    assert '[EFX-0654' not in (b/'outflow/ti60f225_oob.warn.log').read_text()
    evidence={str(f):sha(f) for f in [Path(__file__),TRANSFORM/'result.json',TRANSFORM/'cnn_preview_boot.sv',
        BOUNDARY/'result.json',UNIT/'result.json',TRANSFORM/'expanded_cut_audit.json',
        b/'public_connectivity_audit.json',VARIANT/'cnn_frame_admission.sv']}
    r=dict(status='CONTINUOUS_CONTROL_FULL_BOARD_MAP_PASS',seconds=sec,counts=counts,added_cells=sorted(added),
        lexical_audit=lex,protected_boundary=boundary,inputs_sha256=sha(b/'inputs.json'),
        map_sha256=sha(b/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(b/'outflow/ti60f225_oob.vdb'),
        evidence=evidence,physical_pass=False,board_release=False,hardware_access=False)
    assert not (b/'result.json').exists();(b/'result.json').write_text(json.dumps(r,indent=2));print(r['status'],counts,flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        if BOARD.exists():(BOARD/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2))
        raise
