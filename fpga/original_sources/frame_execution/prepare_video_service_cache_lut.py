"""Collapse the measured two-level cache-address AND, retaining shared logic."""
import argparse,itertools,json,pickle,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token
from continuous_control_native import connections

def lut(mask,vals):
 indices=[0]
 for i,v in enumerate(vals):
  indices=[x+(b<<i) for x in indices for b in ([v] if v in (0,1) else [0,1])]
 result={mask>>x&1 for x in indices}
 return result.pop() if len(result)==1 else 'x'

def main():
 p=argparse.ArgumentParser()
 for n in ('base','board','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();base=a.base.resolve();b=a.board.resolve();out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 proof=json.loads((base/'native-graph-audit.json').read_text())
 for f,h in proof['inputs'].items():assert sha(f)==h,f
 key,g,mg,ps,mp=pickle.loads((b/'retained-audit-parse.pkl').read_bytes())
 assert key[0]==sha(base/'native_source.v')==sha(b/'native_source.v')
 d,_=connections(g);n='LUT__139908';up=d[g[n][1]['I1'][0]]
 assert g[n][0]==g[up][0]=='EFX_LUT4'
 assert ps[n]==ps[up]=={'LUTMASK':(16,0x4444)}
 assert set(g[n][1])==set(g[up][1])=={'I0','I1','O'}
 pins={'I0':g[n][1]['I0'],'I1':g[up][1]['I0'],'I2':g[up][1]['I1'],'O':g[n][1]['O']}
 cases=[]
 for vals in itertools.product((0,1,'x','z'),repeat=3):
  old=lut(0x4444,[vals[0],lut(0x4444,[vals[1],vals[2],0,0]),0,0])
  new=lut(0x1010,[*vals,0]);assert old==new,(vals,old,new)
  cases.append(dict(inputs=vals,old=old,new=new))
 text=(base/'native_source.v').read_text()
 blocks=[m[0] for m in CELL.finditer(text) if m[2].lstrip('\\')==n];assert len(blocks)==1
 before=blocks[0]
 after='    EFX_LUT4 '+n+' ('+', '.join('.'+p+'('+render_token(ts[0])+')' for p,ts in pins.items())+') /* collapsed shared-address AND */ ;'
 assert graph(after)[n]==('EFX_LUT4',pins)
 pat=r'\bdefparam\s+'+re.escape(n)+r'\.LUTMASK\s*=\s*[^;]+;'
 matches=list(re.finditer(pat,text));assert len(matches)==1
 param_before=matches[0][0];param_after="defparam "+n+".LUTMASK = 16'h1010;"
 changed=text.replace(before,after,1).replace(param_before,param_after,1)
 assert changed.replace(after,before,1).replace(param_after,param_before,1)==text
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name)
 (out/'native_source.v').write_text(changed,encoding='utf8')
 receipt=out/'cache-lut-edit.json'
 receipt.write_text(json.dumps(dict(base=str(base),base_source_sha256=sha(base/'native_source.v'),cell=n,upstream=up,
  before=before,after=after,param_before=param_before,param_after=param_after,new_pins=pins,cases=cases,
  lut_count_delta=0,ff_count_delta=0,latency_added=0,critical_logic_levels_removed=1,
  scope='Shared upstream LUT remains for all its other consumers. Only this successor becomes a 3-input AND with the same inversions.'),indent=2))
 graft=json.loads((base/'graft.json').read_text());graft['cache_lut']=dict(parent=str(base),receipt=str(receipt),sha256=sha(receipt))
 graft['inputs'][str(Path(__file__).resolve())]=sha(__file__);(out/'graft.json').write_text(json.dumps(graft,indent=2))
 inputs={**proof['inputs'],**{str(f):sha(f) for f in (base/'native-graph-audit.json',receipt,Path(__file__).resolve(),out/'native_source.v',out/'graft.json')}}
 result={**proof,'cache_lut_parent':str(base),'cache_lut_four_state_cases':64,'cache_lut_reversible_text':True,'inputs':inputs,'hardware_access':False,'board_release':False}
 (out/'native-graph-audit.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(dict(status=result['status'],source_sha256=sha(out/'native_source.v'),retained=result['retained_primitives'],four_state_cases=64,lut_count_delta=0)))
if __name__=='__main__':main()
