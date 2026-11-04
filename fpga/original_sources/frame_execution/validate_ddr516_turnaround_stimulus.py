"""Check 336-case driver/scoreboard on independent memory and injected bad data."""
import argparse,json,re,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--traffic',choices=('matrix','dense','combined'),default='matrix')
 p.add_argument('--layouts',type=int,choices=(1,3),default=1)
 p.add_argument('--serial-turns',action='store_true');a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 sources=[HERE/'tb_ddr516_turnaround.sv',HERE/'tb_ddr516_turnaround_behavior.sv']
 for src in sources:(root/src.name).write_bytes(src.read_bytes())
 stimulus=root/sources[0].name
 if a.traffic!='matrix':stimulus.write_text(stimulus.read_text().replace('MATRIX=1,DENSE=0','MATRIX='+str(int(a.traffic=='combined'))+',DENSE=1'))
 stimulus.write_text(stimulus.read_text().replace('LAYOUTS=1',f'LAYOUTS={a.layouts}'))
 if a.serial_turns:stimulus.write_text(stimulus.read_text().replace('SERIAL=0','SERIAL=1'))
 expected=(336 if a.traffic!='dense' else 0)+(32*a.layouts if a.traffic!='matrix' else 0)+256*a.serial_turns
 (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
 sim=Path('E:/altera/15.0/modelsim_ase/win32aloem');results=[]
 commands=[('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv',*[root/p.name for p in sources]]),
  ('normal',[sim/'vsim.exe','-c','work.tb_turn_behavior','-do','run.do']),
  ('corrupt',[sim/'vsim.exe','-c','work.tb_turn_behavior','-gCORRUPT=1','-do','run.do'])]
 for name,command in commands:
  start=time.monotonic()
  with (root/(name+'.log')).open('wb') as f:r=subprocess.run([str(c) for c in command],cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=90)
  body=(root/(name+'.log')).read_text(errors='replace');errors=bool(re.search(r'\*\* (?:Error|Fatal)',body))
  passed=(r.returncode==0 and not errors) if name!='corrupt' else ('TURN R mismatch' in body and errors and 'TURN_ALL_PASS' not in body)
  if name=='normal':passed=passed and f'PASS TURN_STIMULUS cases={expected} ' in body
  results.append(dict(stage=name,passed=passed,returncode=r.returncode,seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log'))))
  if not passed:break
 ok=len(results)==4 and all(r['passed'] for r in results)
 result=dict(status='TURN_STIMULUS_POSITIVE_NEGATIVE_PASS' if ok else 'TURN_STIMULUS_FAIL',traffic=a.traffic,layouts=a.layouts,cases=expected,stages=results,sources={str(p):sha(p) for p in sources},snapshot_stimulus_sha256=sha(stimulus),full_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 if not ok:raise SystemExit(1)
if __name__=='__main__':main()
