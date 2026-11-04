"""Verify reset and observation chains in the actual full-board mapped netlist."""
import argparse,itertools,json,re
from pathlib import Path
from run_units import ROOT,sha
from prepare_local_reset_board import audit_delta,audit_reset_periphery
from audit_measured_pinout import audit as audit_pins


def norm(s):
    s=re.sub(r'/\*.*?\*/','',s,flags=re.S)
    return re.sub(r'\s+','',s).replace('\\','')


def audit(board,physical=None):
    delta=audit_delta(board);periphery=audit_reset_periphery(board)
    meta=json.loads((board/'result.json').read_text());net=board/'outflow/ti60f225_oob.map.v'
    assert sha(net)==meta['map_sha256']
    body=net.read_text()
    cells={norm(name):(kind,{k:norm(v) for k,v in re.findall(r'\.(\w+)\(([^()]*)\)',ports)},attrs)
       for kind,name,ports,attrs in re.findall(r'^\s*(EFX_\w+)\s+(\S+)\s*\((.*?)\) /\*([^\n]*)',body,re.M|re.S)}
    params={}
    for name,param,value in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',body,re.M):
        params.setdefault(norm(name),{})[param]=value.strip()
    ffs={n:p for n,(k,p,_) in cells.items() if n.startswith('startup_reset/') and k=='EFX_FF'}
    expected={'startup_reset/released~FF',*[f'startup_reset/hold_count[{i}]~FF' for i in range(6)],
              *[f'startup_reset/lock_pipe[{i}]~FF' for i in range(2)]}
    assert set(ffs)==expected,set(ffs)
    for n,p in ffs.items():
        assert p['CLK']=='process_clk_150',(n,p)
        for key in ('SR_SYNC','SR_VALUE','SR_POLARITY'):assert params[n][key]=="1'b0",(n,key)
    assert ffs['startup_reset/lock_pipe[0]~FF']['D']=="1'b1"
    assert ffs['startup_reset/lock_pipe[1]~FF']['D']=='startup_reset/lock_pipe[0]'
    assert all('async_reg="true"' in cells[f'startup_reset/lock_pipe[{i}]~FF'][2] for i in range(2))
    # Exhaustively check the real flattened LUT/alias cone, including the new
    # peripheral LOCKED signal. No behavioral replacement or source-only check.
    drivers={p['O']:(n,p) for n,(k,p,_) in cells.items() if k=='EFX_LUT4'}
    aliases={norm(a):norm(b) for a,b in re.findall(r'^\s*assign\s+(.*?)\s*=\s*([^;]+);',body,re.M)}
    visited=set()
    def evaluate(signal,values,stack=()):
        if signal in values:return values[signal]
        if signal in ("1'b0","1'b1"):return int(signal[-1])
        assert signal not in stack,('Combinational cycle',signal)
        if signal in aliases:return evaluate(aliases[signal],values,stack+(signal,))
        assert signal in drivers,('Unknown cone driver',signal)
        name,ports=drivers[signal];visited.add(name)
        mask=int(params[name]['LUTMASK'].split("'h")[1],16)
        index=0
        for i in range(4):
            pin=f'I{i}'
            if pin in ports:index|=evaluate(ports[pin],values,stack+(signal,))<<i
            else:assert all((mask>>v&1)==(mask>>(v^(1<<i))&1) for v in range(16))
        return mask>>index&1
    names=('i_arstn','i_pll_locked','pll_locked','user_pll_locked','process_pll_locked','startup_reset/released')
    for bits in itertools.product((0,1),repeat=6):
        values=dict(zip(names,bits))
        assert evaluate('startup_reset/sources_ready',values)==int(all(bits[:5])),bits
        assert evaluate('w_arstn',values)==int(all(bits)),bits
    slow_ffs={n:p for n,(k,p,_) in cells.items() if n.startswith('startup_reset_5m/') and k=='EFX_FF'}
    assert set(slow_ffs)=={n.replace('startup_reset/','startup_reset_5m/') for n in expected}
    for n,p in slow_ffs.items():
        assert p['CLK']=='CLK_5M'
        for key in ('SR_SYNC','SR_VALUE','SR_POLARITY'):assert params[n][key]=="1'b0",(n,key)
    for bits in itertools.product((0,1),repeat=7):
        values=dict(zip(names+('startup_reset_5m/released',),bits))
        assert evaluate('w_arstn',values)==int(all(bits[:6])),bits
        assert evaluate('w_i2c_arstn',values)==int(all(bits[:5]) and bits[6]),bits
        for ports in list(ffs.values())+list(slow_ffs.values()):
            assert evaluate(ports['SR'],values)==int(all(bits[:5])),bits
    counters={n:p for n,(k,p,_) in cells.items() if n.startswith('i2c_rst_cnt[') and k=='EFX_FF'}
    assert len(counters)==13
    assert all(p['CLK']=='CLK_5M' and params[n]['SR_POLARITY']=="1'b0" for n,p in counters.items())
    for bits in itertools.product((0,1),repeat=7):
        values=dict(zip(names+('startup_reset_5m/released',),bits))
        for ports in counters.values():
            assert evaluate(ports['SR'],values)==int(all(bits[:5]) and bits[6]),bits
    obs=[]
    for i in range(32):
        first=f'bringup_ddr_meta[{i}]~FF';second=f'bringup_ddr_sync[{i}]~FF'
        for n in (first,second):
            kind,ports,attrs=cells[n]
            assert kind=='EFX_FF' and ports['CLK']=='process_clk_150' and 'async_reg="true"' in attrs
        assert cells[second][1]['D']==cells[first][1]['Q']==f'bringup_ddr_meta[{i}]'
        obs.extend((first,second))
    if physical is None:
        return dict(status='FULL_MAPPED_LOCAL_RESETS_PASS_PENDING_PERIPHERY',board=str(board),
                    reset_ff=18,lock_sync_ff=4,observation_ff=64,reset_cone_truth_patterns=128,
                    i2c_reset_only_local_released=True,map_sha256=sha(net),delta=delta,
                    inputs_sha256=sha(board/'inputs.json'),checker_sha256=sha(__file__),board_release=False)
    interface=physical/'outflow/ti60f225_oob.interface.csv'
    report=physical/'outflow/ti60f225_oob.pt.rpt'
    template=physical/'outflow/ti60f225_oob_template.v'
    text=report.read_text();start=text.index('Locked Pin Name               : process_pll_locked')
    assert 'Resource                      : PLL_BR0' in text[max(0,start-120):start]
    assert len(re.findall(r'^input,.*\bprocess_pll_locked\s*$',interface.read_text(),re.M))==1
    assert re.search(r'input process_pll_locked\b',template.read_text())
    pins=audit_pins(physical)
    evidence={str(p):sha(p) for p in (board/'inputs.json',board/'result.json',net,interface,report,template,Path(__file__))}
    evidence.update(pins['inputs'])
    return dict(status='FULL_MAPPED_FOURPLL_RESET_AND_PERIPHERY_PASS',board=str(board),physical=str(physical),
        reset_ff=18,lock_sync_ff=4,observation_ff=64,reset_cone_truth_patterns=128,
        reset_cone_luts=sorted(visited),delta=delta,periphery=periphery,pins=pins,
        required_evidence=evidence,analogue_recovery_removal_proven=False,board_release=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
    p.add_argument('--physical',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert not a.out.exists();r=audit(a.board.resolve(),a.physical.resolve() if a.physical else None)
    a.out.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='required_evidence'}))


