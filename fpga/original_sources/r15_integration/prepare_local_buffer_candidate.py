"""Build one isolated combinational fanout experiment after the C3 route fails.

The selected net groups come from the reviewed actual connectivity and geometry.
No source in production, inputs, or the prior implementation is overwritten.
"""
import argparse,collections,faulthandler,json,re,shutil
from pathlib import Path
from audit_production_map import ROOT,sha,parse,topology
from primitive119 import CELL,raw_cells
from run_probe import EFX,run

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--build',default='production-r14')
 ap.add_argument('--groups',choices=['both','mode','enable'],default='both');args=ap.parse_args()
 assert re.fullmatch(r'production-r[1-9][0-9]*',args.build)
 debug=(ROOT/'evidence'/f'{args.build}-preparation-stack.log').open('w',encoding='utf-8')
 faulthandler.dump_traceback_later(30,repeat=True,file=debug)
 previous=ROOT/'build/production-r13';out=ROOT/'build'/args.build
 failure=ROOT/'evidence/production-r13-route.json';fr=json.loads(failure.read_text())
 assert fr['status']=='ROUTE_FAILED','Wait for the existing bounded route; do not duplicate full runs'
 plan_path=ROOT/'evidence/local-control-buffer-plan.json';plan=json.loads(plan_path.read_text())
 if args.groups!='both':
  prior_capacity=ROOT/'evidence/production-r14-placement.json';cap=json.loads(prior_capacity.read_text())
  assert cap['status']=='PLACEMENT_NOT_LEGAL' and cap['resources']['XLRs']['used']>60800
  plan['groups']=[g for g in plan['groups'] if g['label']==args.groups];assert len(plan['groups'])==1
 source=Path(plan['source']);assert sha(source)==plan['source_sha256']==sha(previous/'native222.v')
 assert sha(previous/'outflow/ti60f225_oob.out.placeloc')==plan['geometry_sha256']
 probe=ROOT/'evidence/preserved-buffer-map-probe.json';pr=json.loads(probe.read_text())
 assert pr['status']=='FOUR_PRESERVED_COMBINATIONAL_BUFFERS_MAPPED'
 for p,h in pr['sources'].items():assert sha(Path(p))==h
 out.mkdir(exist_ok=False)
 for p in previous.iterdir():
  if p.is_file() and p.suffix in ('.v','.sv','.sdc','.xml','.bin','.hex','.placeloc'):
   shutil.copy2(p,out/p.name)
 assert not(out/'efx_pnr_settings.ini').exists()
 print('SOURCE_COPIED',flush=True)
 original=source.read_text();raw=raw_cells(original);print('RAW_CELLS_PARSED',len(raw),flush=True)
 old=parse(source);og,od,ol=topology(old);print('SOURCE_GRAPH_PARSED',flush=True)
 edits=collections.defaultdict(list);aliases={};declarations=[];removed=set();observed=collections.Counter()
 for group in plan['groups']:
  oldtoken=tuple(group['net']);driver=group['original_driver'];removed.add(driver)
  assert raw[driver][0]=='EFX_LUT4' and og[driver][1]['I0']==[tuple(t) for t in group['original_input']]
  upstream=raw[driver][1]['I0']
  for branch in group['branches']:
   name=branch['name'];aliases[('net',name,None)]=oldtoken
   declarations += [f'    (* syn_keep="true", syn_preserve="true" *) wire {name};',
    f'    EFX_LUT4 {name}_driver (.I0({upstream}), .I1(1\'b0), .I2(1\'b0), .I3(1\'b0), .O({name}));',
    f'    defparam {name}_driver.LUTMASK = 16\'haaaa;']
   for row in branch['consumers']:
    edits[row['cell']].append((oldtoken,name,row['pins']))
 def change(match):
  name=match[2].lstrip('\\')
  if name in removed:return ''
  if name not in edits:return match[0]
  body=match[3]
  for token,newname,pins in edits[name]:
   expected=collections.Counter(p for p,i in pins)
   for pin,count in expected.items():
    assert pin not in ('CLK','WCLK','RCLK','CLKA','CLKB','SR','RST','CASCIN')
    pattern=r'\\'+re.escape(token[1])+r'\s*\['+str(token[2])+r'\]'
    port=re.search(r'\.'+re.escape(pin)+r'\(([^()]*)\)',body);assert port
    replacement,n=re.subn(pattern,newname,port[1]);assert n==count,(name,pin,n,count)
    body=body[:port.start(1)]+replacement+body[port.end(1):];observed[newname]+=n
  return match[0][:match.start(3)-match.start()]+body+match[0][match.end(3)-match.start():]
 print('BRANCH_EDITS_PREPARED',flush=True)
 modified=CELL.sub(change,original);print('CONSUMER_PORTS_REWRITTEN',flush=True)
 for name in removed:
  # The flat source contains long blank-line runs. Cross-line \s* at every
  # line anchor creates quadratic backtracking; this declaration is one line.
  modified,n=re.subn(r'^[ \t]*defparam[ \t]+\\'+re.escape(name)+r'[ \t]*\.LUTMASK[ \t]*=[ \t]*[^;\r\n]+;','',modified,flags=re.M)
  assert n==1,(name,n)
 for group in plan['groups']:
  for branch in group['branches']:assert observed[branch['name']]==branch['fanout']
 assert modified.count('endmodule')==original.count('endmodule')
 first_end=modified.index('endmodule');modified=modified[:first_end]+'\n'+'\n'.join(declarations)+'\n'+modified[first_end:]
 target=out/'native222.v';target.write_text(modified,encoding='utf-8')
 print('SOURCE_WRITTEN',flush=True)
 new=parse(target);ng,nd,nl=topology(new)
 added={name+'_driver' for name in observed}
 assert set(ng)==set(og)-removed|added
 # Independently collapse each new buffer output back onto the old net and
 # compare every remaining primitive port and parameter, including all state.
 for n in set(og)-removed:
  k,ports=ng[n];collapsed={p:[aliases.get(t,t) for t in ts] for p,ts in ports.items()}
  assert (k,collapsed)==og[n] and new[2].get(n,{})==old[2].get(n,{}),n
 for group in plan['groups']:
  for branch in group['branches']:
   n=branch['name']+'_driver';kind,ports=ng[n];mask=new[2][n]['LUTMASK']
   assert kind=='EFX_LUT4' and mask==(16,43690) and ports['I0']==[tuple(t) for t in group['original_input']]
   assert ports['O']==[('net',branch['name'],None)] and len(nl[ports['O'][0]])==branch['fanout']
 overlay=dict(status='EXACT_COMBINATIONAL_BUFFER_DUPLICATION_SOURCE_PASS',before=sha(source),after=sha(target),
  source=str(source),target=str(target),plan_sha256=sha(plan_path),compiler_probe_sha256=sha(probe),
  selected_groups=args.groups,
  removed_buffer_cells=sorted(removed),added_buffer_cells=sorted(added),
  unchanged_state_and_other_primitives=len(set(og)-removed),changed_input_pin_occurrences=sum(observed.values()),
  additional_LUTs=len(added)-len(removed),latency_cycles_added=0,placement_locks=0,hardware_access=False)
 overlay_path=ROOT/'evidence'/f'{out.name}-source-topology.json';overlay_path.write_text(json.dumps(overlay,indent=2),encoding='utf-8')
 faulthandler.cancel_dump_traceback_later();debug.close()
 original_sdc=ROOT/'inputs/native8h-source/native8h-source/source/ti60f225_oob.sdc'
 shutil.copy2(original_sdc,out/original_sdc.name)
 interface=json.loads((ROOT/'evidence/production-r2-interface-restoration.json').read_text())
 (out/'outflow').mkdir()
 for name,digest in interface['copied_equal_bytes'].items():
  p=Path(interface['source'])/'outflow'/name;assert sha(p)==digest;shutil.copy2(p,out/'outflow'/name)
 # r11-r13 are same-map physical clones; the executed RTL evidence belongs
 # to the original r10 logical staging, not to a new synthesis in r13.
 prior=json.loads((ROOT/'evidence/production-r10-staging.json').read_text())
 functional=prior['functional_evidence']
 for p,h in functional.items():
  assert sha(Path(p))==h
  receipt=json.loads(Path(p).read_text())
  for q,digest in receipt['sources'].items():
   source_file=Path(q);assert sha(source_file)==digest
   if source_file.parent.name=='rtl':assert sha(out/source_file.name)==digest
 record=dict(stage='BUFFER_TOPOLOGY_MAP_PACK',source=str(previous),control_variant='compact4',
  placement_policy='free',optimization_level='CONGESTION_2',functional_evidence=functional,
  topology_overlay={str(overlay_path):sha(overlay_path)},prior_failed_route_sha256=sha(failure),
  selected_groups=args.groups,
  inputs={p.name:sha(p) for p in out.iterdir() if p.is_file()},interface=interface['copied_equal_bytes'],
  route_started=False,hardware_access=False)
 (ROOT/'evidence'/f'{out.name}-staging.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 cmd='\n'.join(['@echo off',f'call "{EFX}/bin/setup.bat"','if errorlevel 1 exit /b 1',
  'call efx_run.bat ti60f225_oob.xml --flow map --timeout 600','if errorlevel 1 exit /b 1',
  'call efx_run.bat ti60f225_oob.xml --flow pnr --pnr_opts pack optimization_level=CONGESTION_2 seed=1 max_threads=8 --timeout 240',
  'exit /b %errorlevel%',''])
 (out/'map-pack.cmd').write_text(cmd,encoding='ascii');print(overlay['status'],len(added),'buffers, added',overlay['additional_LUTs'],flush=True)
 receipt=run(['cmd.exe','/d','/c',out/'map-pack.cmd'],out,'map-pack',900)
 report=(out/'outflow/ti60f225_oob.pnr.hier_util.rpt').read_text()
 line=next(x for x in report.splitlines() if '| +ti60f225_oob_top' in x)
 nums=[int(re.match(r'\s*(\d+)',part)[1]) for part in line.split('|')[2:5]]
 record.update(stage='BUFFER_TOPOLOGY_MAP_PACK_ONLY',resources=line,receipt=receipt,
  xlr=nums[0],ram=nums[1],dsp=nums[2],fits_absolute_capacity=nums[0]<=60800 and nums[1]<=256 and nums[2]<=160,
  placed=False,routed=False,sta_pass=False)
 (ROOT/'evidence'/f'{out.name}-map-pack.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
 print(line,flush=True)

if __name__=='__main__':main()
