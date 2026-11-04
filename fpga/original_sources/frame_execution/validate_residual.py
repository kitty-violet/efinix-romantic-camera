"""Four-state official primitive regression or real map of residual lookup."""
import argparse,collections,json,re,shutil,traceback
from pathlib import Path
from run_units import HERE,ROOT,EF,MS,sha,run
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--mode',choices=['source','map'],required=True)
 p.add_argument('--reference',type=Path);p.add_argument('--seeds',type=int,nargs='+',default=[71,391,1027]);a=p.parse_args()
 build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
 top='residual_lookup_align32';tables=ROOT/'tmp/frame-execution/residual-tables1'
 assert json.loads((tables/'result.json').read_text())['status']=='EXACT_UNEQUAL_SCALE_TABLE_REFERENCE_PASS'
 paths=[HERE/(top+'.sv'),HERE/('tb_'+top+'.sv'),EF/'sim_models/verilog/efx_dpram10.v',HERE/'residual_tables.py',Path(__file__),
  ROOT/'network_ab/contract_reference.py',ROOT/'network_ab/model_b_contract.json']
 records=[]
 try:
  for f in paths:
   dst=build/f.name;shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  for name in ('table.hex','crc.hex'):
   shutil.copy2(tables/name,build/name);records.append(dict(path=str(tables/name),snapshot=str(build/name),sha256=sha(tables/name)))
  manifest=dict(inputs=records,tables_reference_sha256=sha(tables/'result.json'),mode=a.mode,seeds=a.seeds)
  (build/'inputs.json').write_text(json.dumps(manifest,indent=2))
  if a.mode=='source':
   run([MS/'vlib.exe','work'],build,'vlib.log')
   body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',build/'efx_dpram10.v',build/(top+'.sv'),build/('tb_'+top+'.sv')],build,'compile.log')
   assert 'Errors: 0, Warnings: 0' in body
   rows=[]
   for seed in a.seeds:
    body,secs=run([MS/'vsim.exe','-c','work.tb_'+top,'-gSEED='+str(seed),'+VECTORS='+build.as_posix(),
      '-do','onerror {quit -code 1}; run -all; quit -code 0'],build,f'seed{seed}.log')
    lines=re.findall(r'^# (PASS RESIDUAL_LOOKUP.*)$',body,re.M)
    assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
    rows.append(dict(seed=seed,line=lines[0],seconds=secs,log_sha256=sha(build/f'seed{seed}.log')))
   result=dict(status='LOCAL_EXACT_RESIDUAL_PRIMITIVE_PASS',results=rows)
  else:
   assert a.reference
   ref=json.loads((a.reference/'result.json').read_text());assert ref['status']=='LOCAL_EXACT_RESIDUAL_PRIMITIVE_PASS'
   rm=json.loads((a.reference/'inputs.json').read_text());assert sha(a.reference/'inputs.json')==ref['inputs_sha256']
   for row in rm['inputs']:
    if Path(row['path']).suffix=='.sv':assert sha(row['path'])==row['sha256']
   xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
   (build/(top+'.xml')).write_text(xml);(build/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
   cmd='@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 600\nexit /b %errorlevel%\n'
   (build/'run.cmd').write_text(cmd);body,secs=run(['cmd.exe','/d','/c','run.cmd'],build,'map.log')
   assert re.search(r'map\s*:\s*PASS',body)
   net=build/('outflow/'+top+'.map.v');rpt=build/('outflow/'+top+'.map.rpt')
   counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
   for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
   assert counts.get('DPRAM10',0)==16 and counts.get('DSP48',0)==0 and counts.get('DSP24',0)==0
   result=dict(status='LOCAL_RESIDUAL_MAP_PASS_NOT_PNR',resources=counts,map_sha256=sha(net),report_sha256=sha(rpt),seconds=secs,
    source_reference=str(a.reference/'result.json'),source_reference_sha256=sha(a.reference/'result.json'))
  for row in records:assert sha(row['path'])==sha(row['snapshot'])==row['sha256']
  result.update(inputs_sha256=sha(build/'inputs.json'),full_frame=False,pnr=False,trained_scales=False,shared_RQ_connected=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
