"""Reuse an intact placed snapshot after a host timeout; route offline only."""
import argparse,json,re,shutil,subprocess,sys,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from windows_process_tree import stop_launched_tree
from prepare_ddr_shift_guard_board import audit_delta,proof as calibration_proof
from fixed_cnn_ram_locations import verify_placed
from audit_canonical_multiseed import audit

def main():
 p=argparse.ArgumentParser()
 for n in ('source','root'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--recipe',choices=('CONGESTION_1','CONGESTION_3'),required=True)
 p.add_argument('--budget',type=int,default=900)
 p.add_argument('--route-only',action='store_true',help='Allow parallel routing; run STA reports serially afterwards')
 a=p.parse_args();source=a.source.resolve();r=a.root.resolve();old=source/'pnr-seed1';physical=r/'pnr-seed1'
 assert source.is_relative_to(ROOT/'tmp') and r.is_relative_to(ROOT/'tmp') and not r.exists()
 assert 600<=a.budget<=1200
 calibration_proof()
 if a.recipe:
  prior=json.loads((old/'result.json').read_text());previous=audit(source/'canonical-seed1')
  assert not previous['current_timing_pass'] and previous['checks']==dict.fromkeys(previous['checks'],0)
  assert previous['whs_ns']>=0 and previous['wns_ns']<0
  source_gate={str(source/'canonical-seed1/result.json'):sha(source/'canonical-seed1/result.json'),
               str(source/'selected.json'):sha(source/'selected.json'),str(old/'result.json'):sha(old/'result.json')}
 else:
  prior=json.loads((old/'failure.json').read_text());assert 'TimeoutExpired' in prior['error']
  json.loads((old/'pnr_timeout_cleanup.json').read_text())
  source_gate={str(old/'failure.json'):sha(old/'failure.json'),str(old/'pnr_timeout_cleanup.json'):sha(old/'pnr_timeout_cleanup.json')}
 meta=json.loads((old/'physical_inputs.json').read_text());board=Path(meta['base']);audit_delta(board)
 assert sha(board/'inputs.json')==meta['base_inputs_sha256']
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 # Copy preserves the timed-out source and its last legal placement verbatim.
 r.mkdir();(r/'runner.py').write_bytes(Path(__file__).read_bytes())
 shutil.copytree(old,physical)
 for n in ('failure.json','result.json','stages.json'):
  q=physical/n
  if q.exists():q.rename(physical/('source_'+n))
 for n in ('pnr-seed1-ram-probe.py','pnr-seed1-ram-probe.json'):
  shutil.copy2(source/n,r/n)
 recipe_change=None
 if a.recipe:
  tree=ET.parse(physical/'ti60f225_oob.xml');nodes=[n for n in tree.iter() if n.get('name')=='optimization_level']
  assert len(nodes)==1 and nodes[0].get('value')==meta['optimization_change']['value']
  recipe_change=dict(before=nodes[0].get('value'),after=a.recipe)
  nodes[0].set('value',a.recipe);tree.write(physical/'ti60f225_oob.xml',encoding='utf-8',xml_declaration=True)
  meta['optimization_change'].update(value=a.recipe,before_sha256=sha(old/'ti60f225_oob.xml'),after_sha256=sha(physical/'ti60f225_oob.xml'),scope='Route recipe only; original placed seed preserved')
  meta['seed_change'].update(before_sha256=sha(physical/'ti60f225_oob.xml'),after_sha256=sha(physical/'ti60f225_oob.xml'),scope='Retain the source placement seed; no new placement run')
 for row in meta['inputs']:
  if a.recipe and Path(row['snapshot']).name=='ti60f225_oob.xml':row['sha256']=sha(physical/'ti60f225_oob.xml')
  row['snapshot']=str(physical/Path(row['snapshot']).relative_to(old));assert sha(row['snapshot'])==row['sha256']
 (physical/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
 shutil.copytree(source/'startup',r/'startup')
 shutil.copy2(source/'reused_startup.json',r/'reused_startup.json')
 (r/'inputs.json').write_text(json.dumps(dict(board=str(board),source_batch=str(source),source_inputs_sha256=sha(source/'inputs.json'),
  board_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__),hardware_access=False),indent=2))
 names=['outflow/ti60f225_oob'+s for s in ('.place','.gplace','.out.placeloc','.vdb','.map.v','.interface.csv')]
 names+=['ti60f225_oob.xml','ti60f225_oob.sdc','efx_pnr_settings.ini','fixed_dsp.placeloc',
  'work_pnr/ti60f225_oob.net_proto','work_pnr/ti60f225_oob.primplace','work_pnr/ti60f225_oob.io_place']
 original={str(old/n):sha(old/n) for n in names}
 for n in names:
  if not (a.recipe and n=='ti60f225_oob.xml'):assert sha(old/n)==sha(physical/n)
 args=[str(EF/'bin/efx_pnr.exe'),'--circuit','ti60f225_oob','--family','Titanium','--device','Ti60F225',
  '--operating_conditions','I3','--vdb_file','outflow/ti60f225_oob.vdb','--use_vdb_file','on',
  '--prj',str(physical/'ti60f225_oob.xml'),'--output_dir','outflow','--work_dir','work_pnr',
  '--place_file','outflow/ti60f225_oob.place','--route_file','outflow/ti60f225_oob.route',
  '--settings_file','efx_pnr_settings.ini','--sync_file','outflow/ti60f225_oob.interface.csv',
  '--generate_prevpr_netlist','off','--sdc_file=ti60f225_oob.sdc',*(['--max_router_iterations','100'] if a.recipe else []),'--route']
 (physical/'route_only.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\n'+subprocess.list2cmdline(args)+'\nexit /b %errorlevel%\n')
 source_gate[str(old/'physical_inputs.json')]=sha(old/'physical_inputs.json')
 proof=dict(source=str(old),source_kind='legal_route_timing_fail' if a.recipe else 'host_timeout',source_gate=source_gate,optimization_change=recipe_change,
  copied_placement_and_inputs=original,command=args,command_sha256=sha(physical/'route_only.cmd'),budget_s=a.budget,
  rtl_changed=False,map_changed=False,constraints_changed=False,hardware_access=False)
 (r/'route_continuation.json').write_text(json.dumps(proof,indent=2));stages=[]
 def run(name,command,cwd,budget):
  log=r/(name+'.log');start=time.monotonic()
  with log.open('x') as f:
   child=subprocess.Popen(command,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
   row=dict(stage=name,pid=child.pid,status='running',budget_s=budget);stages.append(row)
   (r/'stages.json').write_text(json.dumps(stages,indent=2))
   try:code=child.wait(timeout=budget)
   except subprocess.TimeoutExpired:
    (r/(name+'_cleanup.json')).write_text(json.dumps(stop_launched_tree(child),indent=2));raise
  row.update(status='finished',exit_code=code,seconds=time.monotonic()-start,log_sha256=sha(log))
  (r/'stages.json').write_text(json.dumps(stages,indent=2));assert code==0,(name,str(log))
  return row
 try:
  route=run('route_only',['cmd.exe','/d','/c',str(physical/'route_only.cmd')],physical,a.budget)
  for q,h in original.items():assert sha(q)==h,('Source changed',q)
  for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.vdb','ti60f225_oob.sdc','efx_pnr_settings.ini','fixed_dsp.placeloc'):
   assert sha(physical/name)==sha(old/name)
  route_text=(physical/'outflow/ti60f225_oob.route.out').read_text()
  overlap=re.findall(r'Iter\s+\d+:\s*overlap=(\d+)',route_text);assert overlap and int(overlap[-1])==0
  assert 'Routing took' in route_text and not re.search(r'^\s*(?:ERROR|FATAL)\s*:',route_text,re.M)
  (physical/'cnn_ram_actual_locations.json').write_text(json.dumps(verify_placed(physical),indent=2))
  from fixed_cnn_feedback_locations import verify_placed as verify_feedback
  (physical/'cnn_feedback_actual_locations.json').write_text(json.dumps(verify_feedback(physical),indent=2))
  (physical/'result.json').write_text(json.dumps(dict(status='PLACEMENT_REUSED_ROUTE_COMPLETED_NOT_RELEASE',
   stages=[*[s for s in prior['stages'] if s['flow']=='interface'],dict(flow='pnr',exit_code=0,seconds=route['seconds'],route_only=True,log_sha256=route['log_sha256'])],
   continuation_sha256=sha(r/'route_continuation.json'),board_release=False),indent=2))
  import audit_control_state_implementation as impl
  impl.audit_delta=audit_delta
  (r/'implementation.json').write_text(json.dumps(impl.audit(board,physical),indent=2))
  if a.route_only:
   (r/'route_result.json').write_text(json.dumps(dict(status='SHIFT_GUARD_ROUTE_FINISHED_REQUIRE_STA',physical=str(physical),route_continuation_sha256=sha(r/'route_continuation.json'),hardware_access=False),indent=2))
   print('SHIFT_GUARD_ROUTE_FINISHED_REQUIRE_STA',flush=True);return
  canonical=r/'canonical-seed1';gray=r/'gray-seed1'
  run('canonical',[sys.executable,str(HERE/'canonical_snapshot_sta.py'),'--base',str(physical),'--root',str(canonical)],ROOT,270)
  selected=audit(canonical);(r/'selected.json').write_text(json.dumps(selected,indent=2));assert selected['current_timing_pass'],(selected['wns_ns'],selected['whs_ns'])
  run('gray',[sys.executable,str(HERE/'audit_snapshot_routed_gray.py'),'--base',str(physical),'--root',str(gray)],ROOT,370)
  result=dict(status='SHIFT_GUARD_PHYSICAL_PASS_PENDING_VENDOR_REPLAY_AND_BUNDLE',selected=selected,gray=str(gray),
   route_continuation_sha256=sha(r/'route_continuation.json'),hardware_access=False,board_download_allowed=False)
  (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as exc:
  (r/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False),indent=2));raise

if __name__=='__main__':main()
