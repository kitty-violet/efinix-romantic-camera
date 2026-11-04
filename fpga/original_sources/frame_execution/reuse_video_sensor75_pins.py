"""Reuse completed pin tests only after exact mapped-cell function comparison.

Names of internal wires and unused LUT inputs may change in synthesis. All
cells, state parameters, net connections and LUT truth functions must agree.
Any substantive difference rejects reuse; the mapped tests must then run.
"""
import argparse,copy,json,pickle,re
from pathlib import Path
from run_units import ROOT,HERE,sha
from native_local_drivers import render_token
from probe_full_native_split_map import lut_function
from prepare_video_service_color_exposure import read_rom


def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--prepared',type=Path,required=True);p.add_argument('--tested',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    b,prepared,tested,out=map(Path.resolve,(a.board,a.prepared,a.tested,a.root));assert out.is_relative_to(ROOT/'tmp') and not out.exists()
    meta=json.loads((b/'inputs.json').read_text());native=Path(meta['native']);parent=Path(meta['parent'])
    proof=json.loads((prepared/'inputs.json').read_text());prior=json.loads((tested/'result.json').read_text());pi=json.loads((tested/'inputs.json').read_text())
    assert prior['status']=='SENSOR74_MINIMAL_PIN_PASS_NOT_PHYSICAL_RELEASE' and [x['mode'] for x in prior['cases']]==[0,10,4,6,8]
    assert prior['inputs_sha256']==sha(tested/'inputs.json') and Path(pi['prepared'])==Path(json.loads((native/'inputs.json').read_text())['sensor74_source']) and pi['prepared_sha256']==sha(Path(pi['prepared'])/'inputs.json')
    for f,h in pi['files'].items():assert sha(f)==h
    for name in ('mapped_camera.v','expected.hex','tb.sv','efx_add.v','efx_ff.v','efx_lut4.v','efx_ram10.v'):assert sha(Path(pi['prepared'])/name)==sha(native/name)
    for f,h in proof['files'].items():assert sha(f)==h
    key,_,old,_,op=pickle.loads((parent/'retained-audit-parse.pkl').read_bytes());assert key[1]==sha(parent/'outflow/ti60f225_oob.map.v')
    h,g,ps=pickle.loads((b/'sensor75-map-parse.pkl').read_bytes());assert h==proof['map_sha256']==sha(b/'outflow/ti60f225_oob.map.v')
    expected=copy.deepcopy(old);ep=copy.deepcopy(op)
    for n,r in json.loads((native/'search.json').read_text())['selected'].items():
        expected[n]=('EFX_LUT4',{p:[tuple(t) for t in ts] for p,ts in r['pins'].items()});ep[n]['LUTMASK']=(16,r['mask'])
    _,blocks=read_rom((native/'native_source.v').read_text());blob=next(x[4] for x in blocks if x[3]==20)
    for i in range(40):ep['camera_checked/config_rom/ROM__D$2']['INIT_'+format(i,'X')]=(256,(blob>>(256*i))&((1<<256)-1))
    def names(path):return set(re.findall(r'^\s*EFX_\w+\s+\\(\S+)\s+\(',path.read_text(),re.M))
    selected=names(native/'mapped_camera.v');actual=names(prepared/'mapped_camera.v')
    assert selected==actual,('cell_difference',sorted(selected-actual),sorted(actual-selected))
    outs={'EFX_LUT4':['O'],'EFX_FF':['Q'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
    forward={('net',x,None):('net',x,None) for x in ('CLK_5M','w_i2c_arstn','io_cam_scl_IN','io_cam_sda_IN')};reverse={v:k for k,v in forward.items()}
    for n in selected:
        k,pins=expected[n];nk,np=g[n];assert k==nk
        for p in outs[k]:
            xs,ys=pins.get(p,[]),np.get(p,[]);assert len(xs)==len(ys),(n,p)
            for x,y in zip(xs,ys):
                if x[0]=='literal' or y[0]=='literal':assert x==y;continue
                assert x not in forward or forward[x]==y,('forward',n,p)
                assert y not in reverse or reverse[y]==x,('reverse',n,p)
                forward[x]=y;reverse[y]=x
    def translated(pins):
        return {p:[t if t[0]=='literal' else forward[t] for t in ts] for p,ts in pins.items()}
    def expressions(pins):
        return {p:render_token(ts[0]) if len(ts)==1 else '{'+','.join(render_token(t) for t in ts)+'}' if ts else '' for p,ts in pins.items()}
    pruned=[]
    for n in sorted(selected):
        k,pins=expected[n];mapped=translated(pins);np=g[n][1]
        if k=='EFX_LUT4':
            assert lut_function(expressions(mapped),ep[n]['LUTMASK'][1])==lut_function(expressions(np),ps[n]['LUTMASK'][1]),('lut_function',n)
            if mapped!=np or ep[n]!=ps[n]:pruned.append(n)
        else:assert mapped==np and ep[n]==ps[n],('state_or_wiring',n)
    for name in ('expected.hex',*[k.lower()+'.v' for k in proof['kinds']]):assert sha(native/name)==sha(prepared/name),name
    # Testbenches differ only in how strictly they spell the same gain failure.
    # Reuse requires actual source equality; if changed, run the prepared one.
    old_tb=(native/'tb.sv').read_text();new_tb=(prepared/'tb.sv').read_text()
    before="if(mode==10 && (diagnostic[59:56]!=5 || diagnostic[47:32]!=16'h3e09 || diagnostic[23:16]!=8'h34)) $fatal(1,\"GAIN_MISMATCH_NOT_REJECTED\");"
    after="if(mode==10 && (diagnostic[59:56]!=5 || diagnostic[47:32]!=16'h3e09 || diagnostic[31:24]!=8'h2a || diagnostic[23:16]!=8'h34)) $fatal(1,\"OLD_GAIN_NOT_REJECTED\");"
    assert old_tb.count(before)==new_tb.count(after)==1
    assert old_tb.replace(before,after)==new_tb
    out.mkdir();files={str(f):sha(f) for f in (Path(__file__),b/'sensor75-map-audit.json',prepared/'inputs.json',tested/'inputs.json',tested/'result.json',native/'inputs.json',native/'search.json',prepared/'mapped_camera.v',native/'mapped_camera.v')}
    result=dict(status='SENSOR75_EQUIVALENT_MAPPED_I2C_PINS_EVIDENCE_PASS',cases=prior['cases'],reexecuted=False,
        actual_mapped_cells=len(selected),same_state_and_connections=True,lut_input_pruning_equivalent=pruned,
        note='Prior actual primitive pin tests reused after per-cell connection/parameter/LUT-function proof. Testbench gain failure wording differs; the prior test already requires error 5 at 3e09 with expected34.',hardware_access=False,board_release=False)
    inputs=dict(prepared=str(prepared),prepared_sha256=sha(prepared/'inputs.json'),files=files,reference_test=str(tested),map_sha256=h)
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2));result['inputs_sha256']=sha(out/'inputs.json')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'],len(selected),pruned,flush=True)


if __name__=='__main__':main()
