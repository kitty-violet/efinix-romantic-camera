"""Require the mapped wide feedback mux to depend on registered RQ kind only."""
import argparse,json,re
from pathlib import Path
from run_units import sha,EF
from rq_feedback_select_board import audit_delta
from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm

def audit(board):
    delta=audit_delta(board);net=board/'outflow/ti60f225_oob.map.v';body=net.read_text();cc=cells(body)
    aliases={norm(a):norm(b) for a,b in re.findall(r'^\s*assign\s+(.*?)\s*=\s*([^;]+);',body,re.M)}
    outputs={'EFX_FF':('Q',),'EFX_LUT4':('O',),'EFX_ADD':('O','CO'),'EFX_SRL8':('Q',),'EFX_COMB4':('O','P','CO','FCO')}
    drivers={p[pin]:(n,k,p,pin) for n,(k,p,_) in cc.items() for pin in outputs.get(k,()) if pin in p}
    params={}
    for name,key,value in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',body,re.M):
        params.setdefault(norm(name),{})[key]=norm(value)
    prefix='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
    selector=prefix+'skip/feedback_selector';kind,ports,_=cc[selector];assert kind=='EFX_LUT4'
    expected={prefix+f'rq/selected_kind[{i}]~FF' for i in range(2)}
    visited=set();leafs=set()
    def walk(signal,stack=()):
        if signal in ("1'b0","1'b1"):return
        assert signal not in stack,('Combinational loop',signal)
        if signal in visited:return
        visited.add(signal)
        if signal in drivers:
            name,kind,p,pin_out=drivers[signal]
            if kind=='EFX_FF':
                assert p['CLK']=='process_clk_150';leafs.add(name);return
            if kind=='EFX_COMB4':
                assert pin_out=='P' and params[name]['MODE']=='"LOGIC"'
                # Official P=get_lut_value(1,{1,0,I1,I0}); unlike O it
                # depends only on I0/I1, even when the other LUT output packs
                # a different function. This is not an assumed LUT4 alias.
                assert p['I0']==prefix+'rq/selected_kind[0]' and p['I1']==prefix+'rq/selected_kind[1]'
                mask=int(params[name]['LUTMASK'].split("'h")[1],16)
                assert [(mask>>(8+i))&1 for i in range(4)]==[0,0,1,0]
                walk(p['I0'],stack+(signal,));walk(p['I1'],stack+(signal,));return
            assert kind in ('EFX_LUT4','EFX_ADD'),('Unexpected operand control boundary',name,kind)
            for pin,value in p.items():
                if pin not in outputs[kind]:walk(value,stack+(signal,))
            return
        assert signal in aliases,('Unknown source',signal)
        walk(aliases[signal],stack+(signal,))
    walk(ports['I1'])
    assert leafs and leafs<=expected,sorted(leafs)
    result=json.loads((board/'result.json').read_text());assert result['map_sha256']==sha(net)
    return dict(status='FEEDBACK_SELECTOR_REGISTERED_KIND_NO_READY_CONE_PASS',selector=selector,
        select_input=ports['I1'],source_ff=sorted(leafs),combinational_signals=len(visited),delta=delta,
        map_sha256=sha(net),inputs_sha256=sha(board/'inputs.json'),
        comb4_official_model_sha256=sha(EF/'sim_models/verilog/efx_comb4.v'),
        scope='Actual public feedback XOR control cone terminates only at registered RQ kind. Not a full STA substitute.',
        hardware_access=False,board_release=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();out=board/'feedback_selection_map_audit.json';assert not out.exists()
    result=audit(board);out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
