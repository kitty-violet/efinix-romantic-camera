"""Full native remap and exact-delta audits for the local payload owner change."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from camera_payload_owner import BASE as PARENT
from camera_checked_board import audit_delta as parent_audit
from complete_native_logic_split import resource_counts,protected_blocks
from map_native_local_drivers import flattened_public_ports,lexical_roundtrip
from audit_native_roundtrip import verify
from probe_native_open_ports import compare_boundary

BOARD=ROOT/'tmp/frame-execution/connected-board-camera-payload1'

def audit_delta(board):
    parent_audit(PARENT)
    m=json.loads((board/'inputs.json').read_text());r=json.loads((board/'result.json').read_text())
    assert m['payload_parent']==str(PARENT) and m['payload_parent_sha256']==sha(PARENT/'inputs.json')
    assert r['auditor_sha256']==sha(__file__) and r['original_builder_sha256']==m['payload_builder_sha256']
    for item in m['inputs']:assert sha(item['snapshot'])==item['sha256'],item['snapshot']
    for f,h in r['evidence'].items():assert sha(f)==h,f
    assert r['status']=='PAYLOAD_CAMERA_FULL_BOARD_MAP_PASS'
    assert r['inputs_sha256']==sha(board/'inputs.json')
    assert r['map_sha256']==sha(board/'outflow/ti60f225_oob.map.v') and r['vdb_sha256']==sha(board/'outflow/ti60f225_oob.vdb')
    return dict(status='PAYLOAD_CAMERA_FULL_VIDEO_A_MAP_BOUND',map_sha256=r['map_sha256'],hardware_access=False)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--shorts',type=Path,required=True)
    p.add_argument('--root',type=Path,default=BOARD);p.add_argument('--audit-only',action='store_true');a=p.parse_args();src=a.source.resolve();shorts=a.shorts.resolve();board=a.root.resolve()
    assert board.is_relative_to(ROOT/'tmp') and (board.exists() if a.audit_only else not board.exists());parent_audit(PARENT)
    proof=json.loads((src/'result.json').read_text());assert proof['status']=='PAYLOAD_SPECIALIZATION_GRAPH_PASS'
    test=json.loads((shorts/'result.json').read_text());assert test['status']=='PAYLOAD_PAIRED_OFFICIAL_RAM_FOURSTATE_PASS'
    assert json.loads((shorts/'inputs.json').read_text())['source_result_sha256']==sha(src/'result.json')
    for f,h in proof['inputs'].items():assert sha(f)==h
    for f,h in proof['files'].items():assert sha(src/f)==h
    if a.audit_only:
        assert not (board/'result.json').exists()
        m=json.loads((board/'inputs.json').read_text())
        for item in m['inputs']:assert sha(item['snapshot'])==item['sha256']
        assert sha(src/'native_source.v')==sha(board/'native_source.v')
        assert sha(ROOT/'tmp/board-test/camera_payload_board_initial.py')==m['payload_builder_sha256']
    else:
        board.mkdir();m=json.loads((PARENT/'inputs.json').read_text())
        for item in m['inputs']:
            old=Path(item['snapshot']);dst=board/old.relative_to(PARENT);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(old,dst)
            item['snapshot']=str(dst)
        shutil.copy2(src/'native_source.v',board/'native_source.v')
    # Camera RTL has already been incorporated in this complete map. Do not
    # instantiate it a second time or let hierarchy synthesis alter old logic.
    xml=board/'ti60f225_oob.xml';s=xml.read_text()
    names=re.findall(r'<efx:design_file name="([^"]+)"[^>]*/>',s)
    assert names==['native_source.v'] if a.audit_only else len(names)==6 and 'native_source.v' in names,names
    s=re.sub(r'<efx:design_file name="([^"]+)"[^>]*/>',lambda q:q[0] if q[1]=='native_source.v' else '',s)
    if a.audit_only:log=(board/'map.log').read_text();secs=None
    else:
        xml.write_text(s,encoding='utf8')
        for item in m['inputs']:item['sha256']=sha(item['snapshot'])
        m.update(payload_parent=str(PARENT),payload_parent_sha256=sha(PARENT/'inputs.json'),payload_builder_sha256=sha(__file__),hardware_access=False)
        (board/'inputs.json').write_text(json.dumps(m,indent=2))
        shutil.copy2(PARENT/'map.cmd',board/'map.cmd')
        log,secs=run(['cmd.exe','/d','/c','map.cmd'],board,'map.log',timeout=480)
    assert re.search(r'map\s*:\s*PASS',log)
    before=(board/'native_source.v').read_text();net=board/'outflow/ti60f225_oob.map.v';after=net.read_text()
    connectivity=verify(flattened_public_ports(before),flattened_public_ports(after));assert connectivity['added_protected']==0
    lexical=lexical_roundtrip(before,after);boundary=compare_boundary(before,after,connectivity['renamed_nets'])
    # The tool re-encrypts protected output; ciphertext identity is required
    # for its input only. Do not claim independent proof inside encryption.
    assert protected_blocks(before)==protected_blocks((PARENT/'outflow/ti60f225_oob.map.v').read_text())
    assert len(protected_blocks(before))==len(protected_blocks(after))
    (board/'public_connectivity_audit.json').write_text(json.dumps(connectivity,indent=2))
    counts=resource_counts(board/'outflow/ti60f225_oob.map.rpt');old=resource_counts(PARENT/'outflow/ti60f225_oob.map.rpt')
    expected=dict(old);expected['EFX_FF']=expected.get('EFX_FF',0)+proof['clones']
    assert counts==expected,(counts,expected)
    warn=(board/'outflow/ti60f225_oob.warn.log').read_text();assert '[EFX-0654' not in warn
    evidence={str(f):sha(f) for f in [src/'result.json',src/'groups.json',src/'ram_enable_proof.json',shorts/'result.json',shorts/'inputs.json',Path(__file__)]}
    for row in test['results']:evidence[str(shorts/f"seed{row['seed']}.log")]=row['log_sha256']
    result=dict(status='PAYLOAD_CAMERA_FULL_BOARD_MAP_PASS',seconds=secs,counts=counts,
        map_sha256=sha(net),vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),inputs_sha256=sha(board/'inputs.json'),
        lexical_audit=lexical,protected_boundary=boundary,evidence=evidence,
        auditor_sha256=sha(__file__),original_builder_sha256=m['payload_builder_sha256'],audit_only=a.audit_only,
        protected_scope='Original encrypted input bytes unchanged; official remap re-encrypts output, public boundary/parameters checked. Internal encrypted equivalence not independently asserted.',
        hardware_access=False,physical_pass=False,board_release=False)
    (board/'result.json').write_text(json.dumps(result,indent=2));audit_delta(board);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
