"""Private physical legality probe using an unchanged proven 150MHz interface.
No board project modifications, no bitstream, no timing exceptions.
"""
import argparse,json,re,shutil,subprocess,time,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--paired-from',type=Path);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
base=ROOT/'tmp/efinity-dw8-stream-physical-attempt3b-seed1';variant=HERE/'variants/fixed_dot9'
records=[]
for src,name in [(base/'dw8_stream_physical.peri.xml','dw8_stream_physical.peri.xml'),
                 (base/'dw8_stream_physical.sdc','dw8_stream_physical.sdc'),
                 (variant/'dot9_physical_probe.sv','dw8_stream_physical_probe.sv'),
                 (variant/'dsp_dot9_pair_chain.sv','dsp_dot9_pair_chain.sv')]:
 dst=r/name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(src)))
ns='{http://www.efinixinc.com/enf_proj}';ET.register_namespace('efx',ns[1:-1]);tree=ET.parse(base/'dw8_stream_physical.xml');di=tree.find(ns+'design_info')
for child in list(di):
 if child.tag==ns+'design_file':di.remove(child)
for index,name in enumerate(('dsp_dot9_pair_chain.sv','dw8_stream_physical_probe.sv')):
 di.insert(1+index,ET.Element(ns+'design_file',name=name,version='default',library='default'))
tree.write(r/'dw8_stream_physical.xml',encoding='utf-8',xml_declaration=False)
pair_proof=None
if a.paired_from:
 previous=a.paired_from.resolve();pr=json.loads((previous/'result.json').read_text())
 assert all(x['exit_code']==0 for x in pr['stages']) and len(pr['stages'])==3
 pm=json.loads((previous/'inputs.json').read_text())
 assert sha(previous/'inputs.json')==pr['inputs_sha256']
 for x in pm['inputs']:assert sha(x['snapshot'])==x['sha256']
 assert sha(previous/'dsp_dot9_pair_chain.sv')==sha(r/'dsp_dot9_pair_chain.sv')
 loc=previous/'outflow/dw8_stream_physical.out.placeloc';body=loc.read_text()
 placed=re.findall(r'^arithmetic/LANE\[0\]\.TAP\[(\d+)\]\.mac\s+(\d+)\s+(\d+)\s+0\s*$',body,re.M)
 assert {int(x[0]) for x in placed}==set(range(9))
 constraints=[]
 for tap,x,y in placed:
  for lane in (0,1):constraints.append(f'arithmetic/LANE[{lane}].TAP[{tap}].mac\t{x}\t{y}\t{lane}')
 (r/'paired_dsp.placeloc').write_text('\n'.join(constraints)+'\n')
 (r/'efx_pnr_settings.ini').write_text('loc_assignment = paired_dsp.placeloc\ndump_placeloc = on\n')
 pair_proof=dict(source=str(loc),sha256=sha(loc),constraints_sha256=sha(r/'paired_dsp.placeloc'))
else:(r/'efx_pnr_settings.ini').write_text('dump_placeloc = on\n')
(r/'DO_NOT_DOWNLOAD.txt').write_text('Internal dynamic stimulus. OFFLINE ONLY. Never generate/download a bitstream.\n')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,interface_reference=str(base),interface_unchanged=True,pair_proof=pair_proof,runner_sha256=sha(__file__)),indent=2)+'\n')
stages=[]
for flow,budget in [('map',120),('interface',150),('pnr',240)]:
 script=r/(flow+'.cmd');script.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat dw8_stream_physical.xml --flow '+flow+' --timeout '+str(budget)+'\nexit /b %errorlevel%\n')
 start=time.monotonic()
 with (r/(flow+'.log')).open('w') as log:proc=subprocess.run(['cmd.exe','/d','/c',str(script)],cwd=r,stdout=log,stderr=subprocess.STDOUT,timeout=budget+30)
 stages.append(dict(flow=flow,exit_code=proc.returncode,seconds=time.monotonic()-start,log_sha256=sha(r/(flow+'.log'))))
 (r/'stages.json').write_text(json.dumps(stages,indent=2)+'\n')
 if proc.returncode:break
for x in records:assert sha(x['snapshot'])==sha(x['source'])==x['sha256']
result=dict(status='DOT9_PHYSICAL_PROBE_FINISHED',stages=stages,strict_sta_pass=False,whole_cnn=False,board_download_allowed=False,inputs_sha256=sha(r/'inputs.json'))
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
