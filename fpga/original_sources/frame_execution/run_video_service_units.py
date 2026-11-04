"""Bounded source-snapshotted RTL tests; no hardware access or release claim."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,MS,EF,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--tests',nargs='+');a=p.parse_args()
 out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir(parents=True)
 variant=HERE/'variants/video_service'
 # This is a generated fixture template, not a standalone RTL test. The
 # actual old/new native serializer regressions bind PROBE_BITS separately.
 templates={'tb_video_snapshot_serializer.sv'}
 sources=[p for p in sorted(variant.glob('*.sv')) if p.name not in templates]
 rows=[]
 for src in sources:
  dst=out/src.name;shutil.copy2(src,dst);rows.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 src=EF/'sim_models/verilog/efx_ram10.v';dst=out/src.name;shutil.copy2(src,dst)
 rows.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 env=os.environ.copy();env['TEMP']=env['TMP']=str(out)
 record=dict(status='RUNNING',sources=rows,runner_sha256=sha(__file__),stages=[],hardware_access=False,board_release=False)
 def save():(out/'result.json').write_text(json.dumps(record,indent=2),encoding='utf8')
 start=time.monotonic();save()
 commands=[('vlib',['vlib.exe','work']),('compile',['vlog.exe','-sv','-timescale','1ns/1ps',*[Path(r['snapshot']).name for r in rows]])]
 tests=[p for p in sorted(variant.glob('tb_*.sv')) if p.name not in templates]
 if a.tests:
  assert set(a.tests)<={p.stem for p in tests},a.tests
  tests=[tb for tb in tests if tb.stem in a.tests]
 for tb in tests:
  commands.append((tb.stem,['vsim.exe','-c','work.'+tb.stem,'-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0']))
 try:
  for name,cmd in commands:
   stage=time.monotonic()
   proc=subprocess.run([str(MS/cmd[0]),*cmd[1:]],cwd=out,env=env,capture_output=True,text=True,timeout=45)
   log=proc.stdout+proc.stderr;(out/(name+'.log')).write_text(log,encoding='utf8')
   record['stages'].append(dict(name=name,exit_code=proc.returncode,seconds=round(time.monotonic()-stage,3),log_sha256=sha(out/(name+'.log'))));save()
   if proc.returncode or re.search(r'\*\* (?:Error|Fatal)',log):raise RuntimeError(name+'\n'+log[-3000:])
   if name.startswith('tb_') and 'PASS ' not in log:raise RuntimeError('No positive completion: '+name)
  assert all(sha(r['source'])==r['sha256'] for r in rows)
  record['status']='VIDEO_SERVICE_UNITS_PASS_NOT_INTEGRATED'
 except Exception as exc:record['status']='FAIL';record['error']=str(exc);raise
 finally:
  record['seconds']=round(time.monotonic()-start,3);save();print(json.dumps({k:v for k,v in record.items() if k!='sources'},indent=2))
if __name__=='__main__':main()
