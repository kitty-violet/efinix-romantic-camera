"""Synthesize the actual frame meter or fixed-port arena; no P&R or download."""
import argparse,collections,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--kind',choices=['meter','arena'],required=True)
 p.add_argument('--reference',type=Path,required=True);a=p.parse_args()
 ref=json.loads((a.reference/'result.json').read_text())
 assert ref['status']==('LOCAL_METER_PASS' if a.kind=='meter' else 'LOCAL_OFFICIAL_PRIMITIVE_PASS')
 m=json.loads((a.reference/'inputs.json').read_text());assert sha(a.reference/'inputs.json')==ref['inputs_sha256']
 top='frame_execution_meter' if a.kind=='meter' else 'tensor_arena12'
 build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
 try:
  f=HERE/(top+'.sv');r=next(r for r in m['inputs'] if Path(r['path'])==f)
  assert sha(f)==sha(r['snapshot'])==r['sha256'];shutil.copy2(f,build/f.name)
  xml='''<efx:project xmlns:efx="http://www.efinixinc.com/enf_proj" name="TOP" sw_version="2026.1.132">
<efx:device_info><efx:family name="Titanium"/><efx:device name="Ti60F225"/><efx:timing_model name="I3"/></efx:device_info>
<efx:design_info def_veri_version="sv_09" def_vhdl_version="vhdl_2008" unified_flow="false">
<efx:top_module name="TOP"/><efx:design_file name="TOP.sv" version="default" library="default"/>
<efx:top_vhdl_arch name=""/></efx:design_info><efx:constraint_info><efx:sdc_file name="TOP.sdc"/><efx:inter_file name=""/></efx:constraint_info>
<efx:sim_info/><efx:misc_info/><efx:ooc_info/><efx:ip_info/>
<efx:synthesis tool_name="efx_map"><efx:param name="work_dir" value="work_syn" value_type="e_string"/>
<efx:param name="mode" value="speed" value_type="e_option"/><efx:param name="max_ram" value="-1" value_type="e_integer"/>
<efx:param name="max_mult" value="-1" value_type="e_integer"/></efx:synthesis>
<efx:place_and_route tool_name="efx_pnr"><efx:param name="work_dir" value="work_pnr" value_type="e_string"/></efx:place_and_route>
<efx:bitstream_generation tool_name="efx_pgm"/><efx:debugger><efx:param name="auto_instantiation" value="off" value_type="e_bool"/></efx:debugger></efx:project>'''.replace('TOP',top)
  (build/(top+'.xml')).write_text(xml)
  (build/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
  cmd='@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 600\nexit /b %errorlevel%\n'
  (build/'run.cmd').write_text(cmd)
  manifest=dict(top=top,source_sha256=sha(f),source=str(f),reference_sha256=sha(a.reference/'result.json'),
   reference=str((a.reference/'result.json').resolve()),runner_sha256=sha(__file__),
   files={n:sha(build/n) for n in (f.name,top+'.xml',top+'.sdc','run.cmd')})
  (build/'inputs.json').write_text(json.dumps(manifest,indent=2))
  t=time.monotonic()
  with (build/'map.log').open('w') as out:q=subprocess.run(['cmd.exe','/d','/c','run.cmd'],cwd=build,stdout=out,stderr=subprocess.STDOUT,timeout=660)
  body=(build/'map.log').read_text();assert q.returncode==0 and re.search(r'map\s*:\s*PASS',body)
  net=build/('outflow/'+top+'.map.v');rpt=build/('outflow/'+top+'.map.rpt')
  counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
  for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
  assert not any(counts.get(k,0) for k in ('DSP24','DSP48'))
  if a.kind=='arena':assert counts.get('DPRAM10',0)==84,counts
  assert sha(f)==manifest['source_sha256'] and sha(__file__)==manifest['runner_sha256']
  result=dict(status='LOCAL_MAP_PASS_NOT_PNR',top=top,resources=counts,map_sha256=sha(net),
   report_sha256=sha(rpt),inputs_sha256=sha(build/'inputs.json'),seconds=time.monotonic()-t,full_frame=False,pnr=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()

