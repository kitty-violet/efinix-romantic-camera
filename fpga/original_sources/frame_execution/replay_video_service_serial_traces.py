"""Run each real observer separately, compare every sampled output and input.

Keeps each simulation below the installed Starter's instance limit. Does not
suppress that check or replace the hardware models. All original stimulus and
single-DUT protocol assertions remain. Exact trace equality supplies the
cross-DUT comparison, including X/Z, sequence, busy, and valid at every TCK.
"""
import argparse,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,sha
from windows_process_tree_safe import stop_launched_tree
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 src=a.source.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 meta=json.loads((src/'inputs.json').read_text())
 for f,h in meta['files'].items():assert sha(f)==h,f
 root.mkdir();started=time.monotonic();sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 results=[];inputs={str(src/'inputs.json'):sha(src/'inputs.json'),str(Path(__file__)):sha(__file__)}
 result=dict(status='RUNNING',hardware_access=False,board_release=False)
 try:
  for selected in ('old','new'):
   out=root/selected;out.mkdir();other='new' if selected=='old' else 'old'
   for f in src.glob('*.v'):shutil.copy2(f,out/f.name)
   shutil.copy2(src/(selected+'.sv'),out/(selected+'.sv'))
   tb=(src/'tb.sv').read_text()
   pattern=r'^ observer_'+other+r' '+other+r'_dut\([^\n]+\);\n'
   tb,count=re.subn(pattern,'',tb,flags=re.M);assert count==1
   tb=re.sub(r'^  '+other+r'_dut\.[^\n]+\n','',tb,flags=re.M)
   alias='\n'.join(f'assign {other}_{n}={selected}_{n};' for n in ('tdo','busy','valid','sequence'))
   anchor=' integer checked=0';assert tb.count(anchor)==1
   tb=tb.replace(anchor,alias+'\ninteger trace_handle;initial trace_handle=$fopen("trace.txt","w");\n'+anchor)
   old='task tick;begin #17;tck=1;#17;tck=0;#0.01;end endtask'
   new='task tick;begin #17;tck=1;#17;tck=0;#0.01;$fwrite(trace_handle,"%b_%b_%b_%b_%b%b%b_%h\\n",'+','.join(selected+'_'+n for n in ('tdo','busy','valid','sequence'))+',capture,poll,shift,probes);end endtask'
   assert tb.count(old)==1;tb=tb.replace(old,new).replace('$finish;','$fclose(trace_handle);$finish;')
   (out/'tb.sv').write_text(tb)
   files={str(f):sha(f) for f in out.iterdir()};inputs.update(files)
   for args,name,budget in [(['vlib.exe','work'],'vlib.log',20),(['vlog.exe','-sv','-timescale','1ns/1ps',*map(str,out.glob('*.v')),*map(str,out.glob('*.sv'))],'compile.log',60),(['vsim.exe','-c','work.tb_video_snapshot_serializer','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],'simulate.log',180)]:
    with (out/name).open('x') as log:
     child=subprocess.Popen([str(sim/args[0]),*args[1:]],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
     try:rc=child.wait(timeout=budget)
     except subprocess.TimeoutExpired:
      (out/'timeout-cleanup.json').write_text(json.dumps(stop_launched_tree(child),indent=2));raise
    body=(out/name).read_text();assert rc==0 and not re.search(r'^#?\s*\*\* (Error|Fatal)',body,re.M),(selected,name,body[-1600:])
   assert 'PASS VIDEO_SERVICE_INDEXED_SERIAL_NATIVE' in body
   trace=out/'trace.txt';results.append(dict(dut=selected,trace_sha256=sha(trace),samples=sum(1 for _ in trace.open()),markers=re.findall(r'^# PASS.*$',body,re.M)))
  assert results[0]['trace_sha256']==results[1]['trace_sha256'] and results[0]['samples']>47000,results
  for f,h in inputs.items():assert sha(f)==h,f
  result.update(status='ACTUAL_NATIVE_SERIAL_SEPARATE_EXACT_TRACE_PASS',results=results,trace_compares='Every TCK: all input probe bits, capture/poll/shift, TDO, busy, valid, sequence; exact including X/Z')
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result.update(seconds=time.monotonic()-started,inputs=inputs)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='inputs'},indent=2))
if __name__=='__main__':main()
