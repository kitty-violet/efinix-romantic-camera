"""Parallel candidate-only component resource maps; never claims whole design fit."""
import argparse,collections,concurrent.futures,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha,run

def component(root,top):
 build=root/top;build.mkdir();f=HERE/(top+'.sv');dst=build/f.name;shutil.copy2(f,dst);h=sha(f)
 try:
  xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
  (build/(top+'.xml')).write_text(xml);(build/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
  (build/'run.cmd').write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 600\nexit /b %errorlevel%\n')
  manifest=dict(inputs=[dict(path=str(f),snapshot=str(dst),sha256=h)],files={x.name:sha(x) for x in build.iterdir() if x.is_file()})
  (build/'inputs.json').write_text(json.dumps(manifest,indent=2));body,seconds=run(['cmd.exe','/d','/c','run.cmd'],build,'map.log')
  assert re.search(r'map\s*:\s*PASS',body)
  net=build/('outflow/'+top+'.map.v');report=build/('outflow/'+top+'.map.rpt');warn=build/('outflow/'+top+'.warn.log')
  assert 'EFX-0201' not in warn.read_text(), 'non-driven rewiring rejected'
  counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',report.read_text(),re.M)==[str(v)]
  assert sha(f)==sha(dst)==h
  result=dict(status='CANDIDATE_COMPONENT_MAP_REQUIRES_GATE',top=top,resources=counts,source_sha256=h,map_sha256=sha(net),
   report_sha256=sha(report),warning_sha256=sha(warn),inputs_sha256=sha(build/'inputs.json'),seconds=seconds,full_frame=False,pnr=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));return result
 except Exception as e:
  fail=dict(status='FAIL',top=top,error=repr(e),traceback=traceback.format_exc());(build/'failure.json').write_text(json.dumps(fail,indent=2));return fail

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(parents=True,exist_ok=False)
 tops=['dw_block_plan32','dw_bank_read32','dw_block_write4','dw_coefficient_registers']
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  results=list(pool.map(lambda t:component(root,t),tops))
 result=dict(status='CANDIDATE_RESOURCE_SCREEN' if all(r['status']!='FAIL' for r in results) else 'HAS_FAILED_COMPONENT',components=results,runner_sha256=sha(__file__))
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 assert result['status']=='CANDIDATE_RESOURCE_SCREEN'
if __name__=='__main__':main()
