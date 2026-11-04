"""Bounded isolated maps to screen data-layout/router structures, not board fit."""
import argparse,concurrent.futures,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,EF,run,sha
from frozen_sources import frozen_paths

def one(root,src,mode,budget=240,ce_effort=None,min_ce_fanout=None):
 assert mode in ("speed","area","area2")
 root.mkdir(parents=True,exist_ok=False);top=src.stem;dst=root/src.name;shutil.copy2(src,dst)
 template=ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml'
 project=template.read_text().replace('frame_execution_meter',top).replace('name="mode" value="speed"', 'name="mode" value="'+mode+'"')
 for key,value in [('infer-clk-enable',ce_effort),('min-ce-fanout',min_ce_fanout)]:
  if value is not None:
   project,n=re.subn(r'(name="'+key+r'" value=")[^"]+',lambda m:m[1]+str(value),project)
   assert n<=1,key
   if n==0:
    value_type='e_option' if key=='infer-clk-enable' else 'e_integer'
    assert project.count('</efx:synthesis>')==1
    project=project.replace('</efx:synthesis>',f'<efx:param name="{key}" value="{value}" value_type="{value_type}" /></efx:synthesis>')
 (root/(top+'.xml')).write_text(project)
 (root/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
 (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout '+str(budget)+'\nexit /b %errorlevel%\n')
 (root/'inputs.json').write_text(json.dumps(dict(source=str(src),source_sha256=sha(src),runner_sha256=sha(__file__),files={f.name:sha(f) for f in root.iterdir() if f.is_file()}),indent=2))
 body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=budget+30);assert re.search(r'map\s*:\s*PASS',body)
 rpt=root/'outflow'/(top+'.map.rpt');warn=root/'outflow'/(top+'.warn.log')
 counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
 assert sha(src)==sha(dst) and 'EFX-0654' not in warn.read_text(errors='replace')
 result=dict(status='ISOLATED_MODE_SCREEN_NOT_BOARD',mode=mode,ce_effort=ce_effort,min_ce_fanout=min_ce_fanout,resources=counts,seconds=secs,source=str(src),source_sha256=sha(src),report_sha256=sha(rpt),warning_sha256=sha(warn))
 (root/'result.json').write_text(json.dumps(result,indent=2));return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--mode',choices=('speed','area','area2'),required=True);p.add_argument('--budget',type=int,default=240);p.add_argument('--ce-effort',type=int,choices=(0,1,2,3,4));p.add_argument('--min-ce-fanout',type=int);a=p.parse_args()
 assert 60<=a.budget<=900
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 assert a.min_ce_fanout is None or a.min_ce_fanout>=0
 print(json.dumps(one(root,a.source.resolve(),a.mode,a.budget,a.ce_effort,a.min_ce_fanout)),flush=True)
if __name__=='__main__':main()
