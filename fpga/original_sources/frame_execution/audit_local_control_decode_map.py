"""Actual map structure and truth table, independent of RTL attributes' intent."""
import argparse,json,re
from pathlib import Path
from run_units import sha
from local_control_decode import audit_delta
from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm

def audit(board):
    delta=audit_delta(board);net=board/'outflow/ti60f225_oob.map.v'
    result=json.loads((board/'result.json').read_text());assert result['map_sha256']==sha(net)
    body=net.read_text();cc=cells(body)
    prefix='cpu/ctrl_decode/'
    selected={n:(k,p) for n,(k,p,_) in cc.items() if n.startswith(prefix)}
    expected={prefix+f'{g}_{i}' for g,w in [('lo',4),('mi',3),('hi',3)] for i in range(w+1)}|{prefix+f'eq_{i}' for i in range(10)}
    assert set(selected)==expected,(sorted(selected),sorted(expected))
    assert all(k=='EFX_LUT4' for k,p in selected.values()),selected
    params={}
    for n,param,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',body,re.M):
        params.setdefault(norm(n),{})[param]=v.strip()
    aliases={norm(a):norm(b) for a,b in re.findall(r'^\s*assign\s+(.*?)\s*=\s*([^;]+);',body,re.M)}
    drivers={p['O']:n for n,(k,p) in selected.items()}
    def evaluate(signal,pattern,stack=()):
        if signal in ("1'b0","1'b1"):return int(signal[-1])
        m=re.fullmatch(r'cpu/state_oh\[(\d+)\]',signal)
        if m:return pattern>>int(m[1])&1
        assert signal not in stack
        if signal in drivers:
            n=drivers[signal];pins=selected[n][1];mask=int(params[n]['LUTMASK'].split("'h")[1],16)
            index=sum(evaluate(pins.get('I'+str(i),"1'b0"),pattern,stack+(signal,))<<i for i in range(4))
            return mask>>index&1
        assert signal in aliases,signal
        return evaluate(aliases[signal],pattern,stack+(signal,))
    for pattern in range(1024):
        for i in range(10):
            output=selected[prefix+f'eq_{i}'][1]['O']
            assert evaluate(output,pattern)==int(pattern==1<<i),(pattern,i)
    # Preserve the two real command CDC flops and the independent DDR event CDC.
    from audit_command_toggle_sync import audit as sync
    sync_result=sync(body);assert not sync_result['errors']
    pairs=[]
    for i in range(2):
        a=f'cpu/shift_meta[{i}]~FF';b=f'cpu/shift_sync[{i}]~FF'
        for n in (a,b):
            k,p,attrs=cc[n];assert k=='EFX_FF' and p['CLK']=='process_clk_150' and 'async_reg=' in attrs
            assert p['CE']=="1'b1" and p['SR']=="1'b0"
        assert cc[b][1]['D']==cc[a][1]['Q'];pairs.append([a,b])
    return dict(status='LOCAL_STATE_DECODE_23_SINGLE_OUTPUT_LUTS_MAP_PASS',lut_count=len(selected),truth_table_checks=10240,
        local_comb4=0,whole_board_comb4=sum(k=='EFX_COMB4' for k,p,a in cc.values()),
        command_sync=sync_result,ddr_event_sync_pairs=pairs,delta=delta,
        inputs_sha256=sha(board/'inputs.json'),map_sha256=sha(net),hardware_access=False,board_release=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    board=a.board.resolve();path=board/'local_decode_map_audit.json';assert not path.exists()
    result=audit(board);path.write_text(json.dumps(result,indent=2));print(json.dumps(result))
