"""Remove only data clearing whose public validity requires a fresh capture.

Invalid payload contents are deliberately unspecified. Valid bits, error bits,
queue credits, tags, clocks, capture enables and arithmetic remain unchanged.
This is valid-qualified equivalence, not unqualified all-output equivalence.
"""
import argparse,itertools,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from native_local_drivers import CELL,graph
from probe_full_native_split_map import params
from audit_video_service_rq_fast import lut

P='video_processing_pipeline/pipeline/cnn/core/dut/engine/'

def main():
 p=argparse.ArgumentParser()
 for n in ('base','short','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();base=a.base.resolve();out=a.root.resolve();short=a.short.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 parentfile=base/'native-graph-audit.json';parent=json.loads(parentfile.read_text())
 assert parent['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
 for f,h in parent['inputs'].items():assert sha(f)==h,f
 test=json.loads((short/'result.json').read_text());assert test['status']=='DATA_ONLY_RESET_PROTOCOL_AND_NUMERIC_SHORT_PASS_NOT_NATIVE_PROOF'
 assert sha(short/'inputs.json')==test['inputs_sha256']
 for f,h in json.loads((short/'inputs.json').read_text()).items():assert sha(f)==h,f
 text=(base/'native_source.v').read_text();ps=params(text)
 targets={P+stem+'['+str(i)+']~FF' for stem in ('rq/rq_data','skip/skip_data') for i in range(256)}
 control={P+'rq/rq_valid[31]~FF',P+'skip/skip_held~FF'}
 fragments={m[2].lstrip('\\'):m[0] for m in CELL.finditer(text) if m[2].lstrip('\\') in targets|control}
 assert set(fragments)==targets|control
 g=graph('\n'.join(fragments.values()));sr=[('net',P+'wsr',5)]
 clk=[('net','process_clk_150',None)];rq_ce=[('net',P+'rq/LANE[31].rq/v2',None)]
 skip_ces={tuple(g[n][1]['CE']) for n in targets if '/skip/' in n}
 # Prior physical work already cloned the capture decode geographically.
 # Check the exact current LUT functions, not one guessed enable-net name.
 enable_tokens={t[0] for t in skip_ces};enable_cells={}
 for m in CELL.finditer(text):
  if m[1]!='EFX_LUT4':continue
  if not any(t[1] in m[0] for t in enable_tokens):continue
  n=m[2].lstrip('\\');cg=graph(m[0]);pins=cg[n][1]
  if len(pins.get('O',[]))==1 and pins['O'][0] in enable_tokens:enable_cells[pins['O'][0]]=(n,pins)
 assert set(enable_cells)==enable_tokens
 refname,refpins=enable_cells[('net',P+'skip/n7085',None)]
 refins=[refpins['I'+str(i)][0] for i in range(4)]
 ce_checks=0
 for n,pins in enable_cells.values():
  ins=[pins['I'+str(i)][0] for i in range(4)]
  leaves=list(dict.fromkeys(t for t in ins+refins if t[0]=='net'));assert len(leaves)<=4
  for bits in itertools.product((0,1,'x','z'),repeat=len(leaves)):
   values=dict(zip(leaves,bits));assert lut(ps[n]['LUTMASK'][1],ins,values)==lut(ps[refname]['LUTMASK'][1],refins,values);ce_checks+=1
 assert g[P+'rq/rq_valid[31]~FF'][1]['D']==rq_ce
 assert all(g[n][1]['SR']==sr and g[n][1]['CLK']==clk for n in control)
 records=[]
 for n in sorted(targets):
  k,pins=g[n];q=ps[n]
  assert k=='EFX_FF' and pins['CLK']==clk and pins['SR']==sr
  assert q['CLK_POLARITY']==q['CE_POLARITY']==q['D_POLARITY']==q['SR_SYNC']==q['SR_SYNC_PRIORITY']==(1,1)
  assert q['SR_VALUE']==(1,0) and q['SR_POLARITY']==(1,0)
  if '/rq/' in n:assert pins['CE']==rq_ce
  else:assert tuple(pins['CE']) in skip_ces
  # SR_POLARITY=0: tying SR LOW would hold this register in reset forever.
  old=fragments[n];m=re.search(r'\.SR\([^()]*\)',old);assert m
  new=old[:m.start()]+".SR(1'b1)"+old[m.end():]
  expected={**pins,'SR':[('literal',"1'b1",None)]}
  assert graph(new)[n]==('EFX_FF',expected)
  records.append(dict(cell=n,before=old,after=new,inactive_SR=1,parameters=q))
 # One-step induction for a valid-qualified data register. Data may contain
 # X while invalid. Every transition creating valid captures the same D;
 # a held valid skip sample preserves equality, and reset invalidates it.
 checks=0
 for mode,reset,valid,capture,release,oldq,newq,data in itertools.product(range(2),range(2),range(2),range(2),range(2),(0,1,'x'),(0,1,'x'),(0,1,'x')):
  if valid and oldq!=newq:continue
  if mode==1 and reset and capture:continue  # skip capture includes !reset.
  oldn=0 if reset else data if capture else oldq
  newn=data if capture else newq
  vn=(not reset) and (capture if mode==0 else capture or (valid and not release))
  if vn:assert oldn==newn
  checks+=1
 changes={r['cell']:r for r in records}
 changed=CELL.sub(lambda m:changes.get(m[2].lstrip('\\'),{}).get('after',m[0]),text)
 inverse={r['cell']:r for r in records}
 restored=CELL.sub(lambda m:inverse.get(m[2].lstrip('\\'),{}).get('before',m[0]),changed)
 assert restored==text and all(fragments[n] in changed for n in control)
 out.mkdir()
 for f in base.glob('*.sv'):shutil.copy2(f,out/f.name);assert sha(f)==sha(out/f.name)
 (out/'native_source.v').write_text(changed,encoding='utf8')
 rp=out/'data-reset-edit.json'
 rp.write_text(json.dumps(dict(base=str(base),base_sha256=sha(base/'native_source.v'),records=records,
  data_bits=512,control_bits_changed=0,latency_added=0,active_low_SR_inactive_value=1,qualified_induction_cases=checks,
  skip_capture_enable_cells=[n for n,p in enable_cells.values()],skip_capture_four_state_checks=ce_checks,
  scope='Only valid-qualified data equality; invalid data can retain previous values. Native D/CE/CLK and all control/reset FFs are unchanged.',
  short=str(short),short_result_sha256=sha(short/'result.json'),ordinary_placement_anchors_added=0),indent=2))
 graft=json.loads((base/'graft.json').read_text());graft['data_reset']=dict(base=str(base),manifest=str(rp),manifest_sha256=sha(rp));graft['inputs'][str(Path(__file__).resolve())]=sha(__file__)
 (out/'graft.json').write_text(json.dumps(graft,indent=2))
 inputs={**parent['inputs'],**{str(f):sha(f) for f in (parentfile,Path(__file__).resolve(),rp,out/'graft.json',out/'native_source.v',short/'result.json',short/'inputs.json')}}
 inputs.update(json.loads((short/'inputs.json').read_text()))
 proof={**parent,'data_reset_parent':str(base),'data_reset_bits':512,'data_reset_reversible_text':True,
  'data_reset_semantics':'valid-qualified; invalid data intentionally unspecified',
  'data_reset_induction_cases':checks,'data_reset_short':str(short),'inputs':inputs,'hardware_access':False,'board_release':False}
 (out/'native-graph-audit.json').write_text(json.dumps(proof,indent=2))
 print(json.dumps(dict(status='DATA_ONLY_RESET_NATIVE_EDIT_AND_QUALIFIED_SHORT_BOUND',bits=512,induction_cases=checks,source_sha256=sha(out/'native_source.v'))))
if __name__=='__main__':main()
