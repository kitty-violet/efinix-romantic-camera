"""Exact one-DSP Q17 candidate: freeze inputs, run four-state reference tests and real map."""
import argparse,collections,json,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import HERE,ROOT,MS,EF,sha,run
from mapped_pin_audit import audit as audit_pins
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--mode',choices=['source','map','mapped'],required=True);p.add_argument('--reference',type=Path)
 p.add_argument('--seeds',nargs='+',type=int,default=[71,391,1027]);p.add_argument('--original',action='store_true');p.add_argument('--narrow',action='store_true');a=p.parse_args()
 build=a.root.resolve();assert build.is_relative_to(ROOT/'tmp');build.mkdir(parents=True,exist_ok=False)
 original=HERE.parent/'operator_core/requantize_i32_mul18_q17_pipeline.sv'
 candidate=HERE/('variants/narrow/rq_single_dsp.sv' if a.narrow else 'rq_single_dsp.sv');tb=HERE/'tb_rq_single_dsp.sv'
 top='requantize_i32_mul18_q17_pipeline' if a.original else 'rq_single_dsp'
 try:
  paths=[candidate,original,tb,Path(__file__),HERE/'mapped_pin_audit.py']
  ref=None
  if a.reference:
   ref=json.loads((a.reference/'result.json').read_text())
   assert ref['status'] in ('LOCAL_EXACT_SOURCE_PASS','LOCAL_MAP_PASS_NOT_PNR')
   paths+=[a.reference/'result.json',a.reference/'inputs.json']
   rm=json.loads((a.reference/'inputs.json').read_text())
   for item in rm['inputs']:
    if Path(item['path']) in (candidate,original,tb):assert sha(item['path'])==item['sha256']
  if a.mode=='mapped':
   assert ref and ref['status']=='LOCAL_MAP_PASS_NOT_PNR'
   net=a.reference/('outflow/'+top+'.map.v');assert sha(net)==ref['map_sha256'];paths.append(net)
   types=set(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M))
   paths += [EF/('sim_models/verilog/efx_'+x.lower()+'.v') for x in sorted(types)]
  records=[]
  for i,f in enumerate(paths):
   assert f.is_file(),f
   dst=build/('reference_'+f.name if a.reference and f in (a.reference/'result.json',a.reference/'inputs.json') else f.name)
   if dst.exists():dst=build/(str(i)+'_'+f.name)
   shutil.copy2(f,dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(f)))
  manifest=dict(inputs=records,mode=a.mode,seeds=a.seeds,top=top,original=a.original)
  (build/'inputs.json').write_text(json.dumps(manifest,indent=2))
  if a.mode in ('source','mapped'):
   run([MS/'vlib.exe','work'],build,'vlib.log')
   files=[build/original.name,build/tb.name]
   if a.mode=='source':files.insert(0,build/candidate.name)
   else:
    files.insert(0,build/(top+'.map.v'))
    files += [Path(r['snapshot']) for r in records if Path(r['path']).parent==EF/'sim_models/verilog']
   body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*files],build,'compile.log')
   assert 'Errors: 0, Warnings: 0' in body
   results=[]
   for seed in a.seeds:
    body,seconds=run([MS/'vsim.exe','-c','work.tb_rq_single_dsp','-gSEED='+str(seed),
      '-do','onerror {quit -code 1}; run -all; quit -code 0'],build,f'seed{seed}.log')
    lines=re.findall(r'^# (PASS RQ_SINGLE_DSP.*)$',body,re.M)
    assert len(lines)==1
    warnings=audit_pins(body,(build/(top+'.map.v')).read_text()) if a.mode=='mapped' else None
    if a.mode=='source':assert 'Errors: 0, Warnings: 0' in body
    results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(build/f'seed{seed}.log'),warnings=warnings))
   result=dict(status='LOCAL_EXACT_SOURCE_PASS' if a.mode=='source' else 'LOCAL_OFFICIAL_MAP_NUMERIC_PASS',results=results)
  else:
   assert ref and ref['status']=='LOCAL_EXACT_SOURCE_PASS'
   # Same existing Ti60/I3 project template; fixed150MHz, no P&R claim.
   xml=(ROOT/'tmp/frame-execution/meter-map1/frame_execution_meter.xml').read_text().replace('frame_execution_meter',top)
   (build/(top+'.xml')).write_text(xml)
   (build/(top+'.sdc')).write_text('create_clock -name clk -period 6.6667 [get_ports {clk}]\n')
   cmd='@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat '+top+'.xml --flow map --timeout 600\nexit /b %errorlevel%\n'
   (build/'run.cmd').write_text(cmd)
   body,seconds=run(['cmd.exe','/d','/c','run.cmd'],build,'map.log')
   assert re.search(r'map\s*:\s*PASS',body)
   net=build/('outflow/'+top+'.map.v');rpt=build/('outflow/'+top+'.map.rpt')
   counts=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
   for k,v in counts.items():assert re.findall(r'^EFX_'+k+r'\s*:\s*(\d+)\s*$',rpt.read_text(),re.M)==[str(v)]
   assert counts.get('DSP48',0)==(2 if a.original else 1),counts
   result=dict(status='LOCAL_MAP_PASS_NOT_PNR',resources=counts,map_sha256=sha(net),report_sha256=sha(rpt),seconds=seconds)
  for r in records:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  result.update(inputs_sha256=sha(build/'inputs.json'),full_frame=False,pnr=False)
  (build/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception as e:
  (build/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
