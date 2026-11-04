"""Joint DW/PW resource: exactly one reader, resident stream, pool and writer."""
import argparse,collections,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha,run
from validate_residual_bypass import selected

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 try:
  top='shared_pool_banked_fifo';records=[]
  for f in selected():
   dst=root/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
  xml=xml.replace('<efx:design_file name="'+top+'.sv" version="default" library="default" />',
   ''.join('<efx:design_file name="'+Path(r['snapshot']).name+'" version="default" library="default" />' for r in records))
  (root/(top+'.xml')).write_text(xml);(root/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
  (root/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 900\nexit /b %errorlevel%\n')
  body,secs=run(['cmd.exe','/d','/c','run.cmd'],root,'map.log');assert re.search(r'map\s*:\s*PASS',body)
  net=root/('outflow/'+top+'.map.v');rpt=root/('outflow/'+top+'.map.rpt');warn=root/('outflow/'+top+'.warn.log')
  assert 'EFX-0201' not in warn.read_text()
  counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result=dict(status='RESIDUAL_BYPASS_POOL_MAP_REQUIRES_GATE',resources=counts,seconds=secs,inputs_sha256=sha(root/'inputs.json'),
   map_sha256=sha(net),warning_sha256=sha(warn),report_sha256=sha(rpt),one_pool=True,arena_external=True,full_frame=False,pnr=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
