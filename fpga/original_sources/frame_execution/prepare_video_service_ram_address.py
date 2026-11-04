"""Flatten 45 measured storage-address gates without extra state or cycles."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph,render_token

def main():
 p=argparse.ArgumentParser()
 for n in ('base','survey','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();base=a.base.resolve();out=a.root.resolve();survey=a.survey.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 r=json.loads(survey.read_text());proof=json.loads((base/'native-graph-audit.json').read_text())
 assert r['source_sha256']==sha(base/'native_source.v') and len(r['candidates'])==45
 for f,h in proof['inputs'].items():assert sha(f)==h,f
 text=(base/'native_source.v').read_text();rows={v['child']:v for v in r['candidates']}
 assert len(rows)==45 and all(v['parent_remaining_loads'] for v in rows.values())
 records=[]
 for m in CELL.finditer(text):
  n=m[2].lstrip('\\')
  if n not in rows:continue
  v=rows[n];old=m[0];pins={f'I{i}':[tuple(t)] for i,t in enumerate(v['inputs'])};pins['O']=[tuple(t) for t in v['child_pins']['O']]
  assert not n.startswith('protected_inst/') and graph(old)[n][0]=='EFX_LUT4'
  assert graph(old)[n][1]=={k:[tuple(t) for t in ts] for k,ts in v['child_pins'].items()}
  new='    EFX_LUT4 '+n+' ('+', '.join('.'+k+'('+render_token(ts[0])+')' for k,ts in pins.items())+') /* exact shared RAM-address gate composition */ ;'
  assert graph(new)[n]==('EFX_LUT4',pins)
  matches=list(re.finditer(r'\bdefparam\s+'+re.escape(n)+r'\.LUTMASK\s*=\s*[^;]+;',text));assert len(matches)==1
  records.append(dict(child=n,parent=v['parent'],before=old,after=new,param_before=matches[0][0],param_after=f"defparam {n}.LUTMASK = 16'h{v['new_mask']:04x};"))
 assert len(records)==45
 changed=text
 for v in records:
  assert changed.count(v['before'])==changed.count(v['param_before'])==1
  changed=changed.replace(v['before'],v['after'],1).replace(v['param_before'],v['param_after'],1)
 restored=changed
 for v in reversed(records):restored=restored.replace(v['after'],v['before'],1).replace(v['param_after'],v['param_before'],1)
 assert restored==text
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name)
 shutil.copy2(base/'cache-lut-edit.json',out/'cache-lut-edit.json')
 (out/'native_source.v').write_text(changed,encoding='utf8')
 receipt=out/'ram-address-edit.json';receipt.write_text(json.dumps(dict(base=str(base),base_source_sha256=sha(base/'native_source.v'),survey=str(survey),survey_sha256=sha(survey),
  records=records,lut_count_delta=0,ff_count_delta=0,latency_added=0,changed_address_bits=45,scope='Five existing store banks, nine address bits per bank. Shared parent gates and their state consumers remain unchanged.'),indent=2))
 graft=json.loads((base/'graft.json').read_text());graft['ram_address']=dict(parent=str(base),receipt=str(receipt),sha256=sha(receipt));graft['inputs'][str(Path(__file__).resolve())]=sha(__file__)
 (out/'graft.json').write_text(json.dumps(graft,indent=2))
 inputs={**proof['inputs'],**{str(f):sha(f) for f in (base/'native-graph-audit.json',receipt,survey,Path(__file__).resolve(),out/'native_source.v',out/'graft.json',out/'cache-lut-edit.json')}}
 proof={**proof,'ram_address_parent':str(base),'ram_address_bits':45,'ram_address_reversible_text':True,'inputs':inputs,'hardware_access':False,'board_release':False}
 (out/'native-graph-audit.json').write_text(json.dumps(proof,indent=2))
 print(json.dumps(dict(status=proof['status'],source_sha256=sha(out/'native_source.v'),changed_luts=45,new_state=0,new_cycles=0)))
if __name__=='__main__':main()
