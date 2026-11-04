"""Check ten real mapped state FFs and their CE/clock/reset polarity."""
import argparse,hashlib,json,re
from pathlib import Path
def audit(net,prefix='',clock='io_systemClk'):
    net=Path(net);raw=net.read_text()
    text=re.sub(r'/\*.*?\*/|//[^\n]*','',raw,flags=re.S)
    params={}
    for n,k,v in re.findall(r'defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',text):
        params.setdefault(n.lstrip('\\'),{})[k]=v.strip()
    cells=[]
    for m in re.finditer(r'EFX_FF\s+(\S+)\s*\((.*?)\)\s*;',text,re.S):
        name=m[1].lstrip('\\')
        if name.startswith(prefix+'state_oh['):
            ports={k:v.strip() for k,v in re.findall(r'\.(\w+)\(([^()]*)\)',m[2])}
            cells.append(dict(name=name,ports=ports,parameters=params[name]))
    assert {x['name'] for x in cells}=={prefix+f'state_oh[{i}]~FF' for i in range(10)},cells
    for cell in cells:
        p=cell['ports'];v=cell['parameters']
        assert p['CE']=="1'b1" and p['CLK']==clock,cell
        assert v['CE_POLARITY']==v['CLK_POLARITY']==v['SR_SYNC']==v['SR_SYNC_PRIORITY']=="1'b1",cell
        assert v['SR_VALUE']==("1'b1" if cell['name']==prefix+'state_oh[0]~FF' else "1'b0"),cell
    return dict(status='ONEHOT_STATE_MAP_STRUCTURE_PASS',state_ffs=cells,
                net=str(net),map_sha256=hashlib.sha256(net.read_bytes()).hexdigest(),
                hardware_access=False,physical_pass=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--net',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--prefix',default='');p.add_argument('--clock',default='io_systemClk');a=p.parse_args()
    assert not a.out.exists();result=audit(a.net,a.prefix,a.clock);a.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='state_ffs'}))
