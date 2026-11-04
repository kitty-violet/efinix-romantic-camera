"""Check generated physical pin assignments, including all vacated pins."""
import argparse,csv,io,json
from pathlib import Path
from run_units import ROOT,sha

EXPECTED={
 'P14':('i_arstn','Input','3.3 V'),
 'P13':('led[0]','Output','3.3 V'),
 'N13':('led[1]','Output','3.3 V'),
 'G3':('led[2].P','Output','1.8 V'),
 'N2':('led[3].P','Output','1.8 V'),
 'C4':('','Input','3.3 V'),'N14':('','Input','3.3 V'),'A3':('','Input','3.3 V'),
 'R14':('','Input','3.3 V'),'P15':('','Input','3.3 V')}

def read(p):
    text=p.read_text(encoding='utf-8');text=text[text.index('Pin Number,'):]
    return {r['Pin Number']:r for r in csv.DictReader(io.StringIO(text))}

def audit(physical):
    p=physical/'outflow/ti60f225_oob.pinout.csv';rows=read(p)
    for ball,(signal,direction,voltage) in EXPECTED.items():
        r=rows[ball]
        assert (r['Signal Name'],r['Direction'],r['I/O Bank Voltage'])==(signal,direction,voltage),(ball,r)
        if direction=='Input':assert r['Pull Type']=='weak pullup'
    old=ROOT/'tmp/frame-execution/bringup-status2-physical-batch1/pnr-seed1/outflow/ti60f225_oob.pinout.csv'
    prev=read(old);assert set(prev)==set(rows)
    different=[ball for ball in rows if rows[ball]!=prev[ball]]
    assert set(different)=={'P14','P13','N13','G3','N2','N14','R14','P15'},different
    # Every other package pin, direction, voltage and physical signal is unchanged.
    return dict(status='GENERATED_PACKAGE_PIN_CORRECTIONS_PASS',physical=str(physical),
                changed_balls=different,checked_package_rows=len(rows),
                inputs={str(p):sha(p),str(old):sha(old)},
                sampled_buttons_as_outputs=False,board_tested=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--physical',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();assert not a.out.exists();r=audit(a.physical.resolve())
    a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r))
