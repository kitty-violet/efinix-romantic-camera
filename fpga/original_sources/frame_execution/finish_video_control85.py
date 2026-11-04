"""Sequential offline STA/CDC/PLL/bitstream gates; never access the board."""
import argparse,json,subprocess,sys,time,traceback
from pathlib import Path
from run_units import ROOT,sha
from audit_canonical_multiseed import audit
from video_control85_physical import BOARD

def main():
 p=argparse.ArgumentParser()
 for n in ('physical','route','canonical','gray','pll','bits','output','root'):p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();v={n:Path(x).resolve() for n,x in vars(a).items()};root=v['root'];h=Path(__file__).resolve().parent
 assert all(x.is_relative_to(ROOT/'tmp/frame-execution') for n,x in v.items() if n!='output')
 assert v['output'].is_relative_to(ROOT/'output/deployment')
 assert not any(v[n].exists() for n in ('canonical','gray','pll','bits','output','root'))
 rr=json.loads((v['route']/'result.json').read_text());assert rr['status']=='VIDEO_SERVICE_LEGAL_ROUTE_PASS_NOT_RELEASE'
 ri=json.loads((v['route']/'inputs.json').read_text());assert Path(ri['physical'])==v['physical']
 assert rr['trace_sha256']==sha(v['physical']/'outflow/ti60f225_oob.troutingtraces')
 root.mkdir();stages=[]
 def save(n,d):(root/n).write_text(json.dumps(d,indent=2),encoding='utf8')
 save('inputs.json',dict(paths={n:str(x) for n,x in v.items()},runner_sha256=sha(__file__),
  route_result_sha256=sha(v['route']/'result.json'),hardware_access=False,automatic_download_allowed=False))
 def run(label,script,*args):
  command=[sys.executable,str(h/script),*map(str,args)];log=root/(label+'.log');t=time.monotonic()
  s=dict(name=label,status='running',command=command);stages.append(s);save('stages.json',stages)
  # Each called tool wrapper owns its timeout and safe process-tree cleanup.
  with log.open('x',encoding='utf8') as f:proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
  s['pid']=proc.pid;save('stages.json',stages);code=proc.wait()
  s.update(status='finished',exit_code=code,seconds=time.monotonic()-t,log_sha256=sha(log));save('stages.json',stages)
  print(json.dumps(s),flush=True);assert code==0,(label,code)
 try:
  run('canonical','video_sensor72_sta.py','--base',v['physical'],'--root',v['canonical'])
  t=audit(v['canonical']);assert t['current_timing_pass'] and Path(t['physical'])==v['physical'] and t['seed']==1,{k:t[k] for k in ('current_timing_pass','physical','seed','wns_ns','whs_ns','checks')}
  run('gray','video_service_narrow_gray.py','--base',v['physical'],'--root',v['gray'],'--inventory',BOARD/'service-cdc-inventory-expanded.json')
  run('pll_measure','survey_pll_shift_routed_rcf.py','--base',v['canonical'],'--root',v['pll'])
  run('pll_audit','video_control85_pll.py','--board',BOARD,'--root',v['pll'])
  run('bits','prepare_routed_bitstream.py','--canonical',v['canonical'],'--root',v['bits'])
  args=[]
  for n in ('route','canonical','gray','pll','bits','output'):args.extend(['--'+n,v[n]])
  run('package','package_video_control85.py',*args)
  validation=json.loads((v['bits']/'offline85-validation.json').read_text())
  save('offline-validation.json',validation)
  save('result.json',dict(status='OFFLINE_RELEASE_READY_WAITING_USER_SUPERVISION',bundle=str(v['output']),
   manifest_sha256=sha(v['output']/'manifest.json'),validation_sha256=sha(root/'offline-validation.json'),hardware_access=False,downloaded=False))
  print('OFFLINE_RELEASE_READY_WAITING_USER_SUPERVISION',v['output'],flush=True)
 except Exception as exc:
  save('failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),hardware_access=False));raise
if __name__=='__main__':main()
