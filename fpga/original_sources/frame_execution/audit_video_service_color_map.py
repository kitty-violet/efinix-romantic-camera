"""Compare the new official map against the released64 map at every public cell."""
import argparse,json,pickle,time
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import graph
from probe_full_native_split_map import params
from audit_video_service_color_exposure import P
from continuous_control_native import net

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native']);parent=Path(meta['parent'])
    proof=json.loads((native/'color-exposure-independent-audit.json').read_text())
    assert proof['status']=='COLOR_EXPOSURE_INDEPENDENT_EXACT_DELTA_PASS'
    assert proof['source_sha256']==sha(board/'native_source.v')
    for f,h in proof['evidence'].items():assert sha(f)==h
    result=json.loads((board/'result.json').read_text());assert result['status']=='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS'
    assert result['inputs_sha256']==sha(board/'inputs.json')
    for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
    oldpath=parent/'outflow/ti60f225_oob.map.v';newpath=board/'outflow/ti60f225_oob.map.v';start=time.monotonic()
    key,_,old,_,op=pickle.loads((parent/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(oldpath)
    text=newpath.read_text();new=graph(text);np=params(text)
    (board/'color-map-parse.pkl').write_bytes(pickle.dumps((sha(newpath),new,np),protocol=4))
    expected=set(proof['changed_connections']);unknown=[];changed=[]
    assert old.keys()==new.keys(),('changed public cell set',sorted(old.keys()-new.keys())[:20],sorted(new.keys()-old.keys())[:20])
    for name in old:
        if old[name]==new[name] and op.get(name)==np.get(name):continue
        kind,pins=old[name];nk,nps=new[name]
        if name in expected:
            wanted={k:list(v) for k,v in pins.items()}
            token=wanted['O'][0]
            assert token[1] in (P+'w_r_00_00_h0',P+'w_b_00_00_h0')
            other='b' if token[1]==P+'w_r_00_00_h0' else 'r'
            wanted['O']=[net(P+'w_'+other+'_00_00_h0',token[2])]
            if nk==kind=='EFX_ADD' and wanted==nps and op[name]==np[name]:changed.append(name);continue
        if name=='camera_checked/config_rom/ROM__D$2' and kind==nk=='EFX_RAM10' and pins==nps:
            delta=[k for k in op[name] if op[name][k]!=np[name][k]]
            if set(delta)=={'INIT_8','INIT_9'}:
                nativekey,ng,nparams=pickle.loads((native/'color-source-parse.pkl').read_bytes())
                assert nativekey==sha(native/'native_source.v') and all(np[name][k]==nparams[name][k] for k in delta)
                changed.append(name);continue
        unknown.append(dict(cell=name,old=old[name],new=new[name],old_params=op.get(name),new_params=np.get(name)))
    inspection=dict(status='COLOR_MAP_INSPECTION',public_cells=len(new),recognized_changes=changed,
                    unknown_changes=unknown,seconds=time.monotonic()-start)
    (board/'color-map-inspection.json').write_text(json.dumps(inspection,indent=2))
    assert not unknown,('Unexpected map differences',len(unknown),[x['cell'] for x in unknown[:12]])
    assert set(changed)==expected|{'camera_checked/config_rom/ROM__D$2'}
    prior=json.loads((parent/'retained-map-audit.json').read_text())
    assert prior['status']=='VIDEO_SERVICE_RETAINED_MAPPED_GRAPH_PASS' and prior['map_sha256']==sha(oldpath)
    report=dict(status='COLOR_EXPOSURE_CURRENT_MAP_COMPOSITION_PASS',public_cells=len(new),changed_public_cells=17,
        all_other_public_cells_and_parameters_identical=True,retained_primitives=prior['retained'],
        source_sha256=sha(board/'native_source.v'),map_sha256=sha(newpath),vdb_sha256=sha(board/'outflow/ti60f225_oob.vdb'),
        evidence={str(f):sha(f) for f in [Path(__file__),oldpath,newpath,native/'color-exposure-independent-audit.json',parent/'retained-map-audit.json',board/'inputs.json',board/'result.json']},
        protected_internal_equivalence='Official mapper; public graph comparison does not inspect encrypted internals',
        hardware_access=False,physical_pass=False,board_release=False,seconds=time.monotonic()-start)
    (board/'color-map-audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
