"""Check the actual full-map public RQ payload is driven by clocked heads."""
import argparse,json,re
from pathlib import Path
from run_units import sha
from rq_cached_head_board import audit_delta,BASE
from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm

def audit(board):
    delta=audit_delta(board);net=board/'outflow/ti60f225_oob.map.v'
    result=json.loads((board/'result.json').read_text());assert sha(net)==result['map_sha256']
    text=net.read_text();cc=cells(text)
    aliases={norm(a):norm(b) for a,b in re.findall(r'^\s*assign\s+(.*?)\s*=\s*([^;]+);',text,re.M)}
    drivers={p['Q']:n for n,(kind,p,_) in cc.items() if kind=='EFX_FF'}
    base='video_processing_pipeline/pipeline/cnn/core/dut/engine/'
    checked={}
    for bus,width in [('od',768),('ox',27),('ob',3)]:
        for bit in range(width):
            original=signal=base+f'{bus}[{bit}]';seen=set()
            while signal not in drivers and signal in aliases:
                assert signal not in seen;seen.add(signal);signal=aliases[signal]
            assert signal in drivers,('RQ payload not directly clocked',original,signal)
            name=drivers[signal];kind,ports,_=cc[name]
            assert ports['CLK']=='process_clk_150',(name,ports)
            assert ports['SR']=="1'b0",(name,'Unexpected wide reset')
            checked[original]=name
    assert len(set(checked.values()))==798
    srl={n:p for n,(k,p,_) in cc.items() if k=='EFX_SRL8' and n.startswith(base+'rq/QUEUE[')}
    assert len(srl)==798,len(srl)
    previous=json.loads((BASE/'result.json').read_text())['resources']
    assert result['resources']['DSP48']==previous['DSP48']
    assert result['resources']['RAM10']==previous['RAM10']
    return dict(status='RQ_798_CLOCKED_HEADS_798_SRL_MAP_PASS',head_bits=798,queue_srl=798,
        head_drivers=checked,head_clock='process_clk_150',wide_head_reset=False,
        resource_delta={k:result['resources'].get(k,0)-previous.get(k,0) for k in set(previous)|set(result['resources'])},
        map_sha256=sha(net),inputs_sha256=sha(board/'inputs.json'),delta=delta,
        physical_timing_proven=False,hardware_access=False,board_release=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();result=audit(board);out=board/'cached_head_map_audit.json';assert not out.exists()
    out.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='head_drivers'}))
