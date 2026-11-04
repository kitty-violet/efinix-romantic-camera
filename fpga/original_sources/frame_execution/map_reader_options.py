"""Small isolated maps to separate two reader expressions; not board savings."""
import argparse, concurrent.futures, json, re, shutil
from pathlib import Path
from run_units import ROOT, HERE, EF, run, sha

def one(root,src):
 root.mkdir(parents=True,exist_ok=False);dst=root/src.name;shutil.copy2(src,dst)
 top='unified_column_reader'
 template=ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml'
 (root/(top+'.xml')).write_text(template.read_text().replace('frame_execution_meter',top))
 (root/(top+'.sdc')).write_text('create_clock -name clk -period 6.667 [get_ports {clk}]\n')
 (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 120\nexit /b %errorlevel%\n')
 (root/'inputs.json').write_text(json.dumps(dict(source=str(src),source_sha256=sha(src),runner_sha256=sha(__file__),
  files={f.name:sha(f) for f in root.iterdir() if f.is_file()}),indent=2))
 body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log',timeout=150)
 assert re.search(r'map\s*:\s*PASS',body)
 rpt=root/'outflow'/(top+'.map.rpt')
 counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
 assert sha(src)==sha(dst)
 result=dict(status='ISOLATED_READER_MAP_NOT_BOARD',resources=counts,seconds=secs,source=str(src),source_sha256=sha(src),report_sha256=sha(rpt))
 (root/'result.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 choices={'reference':ROOT/'tmp/frame-execution/connected-board-fixed-route1/assembly/unified_column_reader.sv'}
 for name in ['reader_select','reader_odd_only','reader_decode_only']:choices[name]=HERE/'variants'/name/'unified_column_reader.sv'
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  futures={name:pool.submit(one,root/name,path) for name,path in choices.items()}
  for name,f in futures.items():print(name,json.dumps(f.result()),flush=True)
