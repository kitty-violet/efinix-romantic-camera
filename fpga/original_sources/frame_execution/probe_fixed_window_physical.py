"""Private 128-DSP cascade placement probe; no programming artifacts."""
import argparse,collections,json,re,shutil,time,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha,run
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
base=ROOT/'tmp/efinity-dw8-stream-physical-attempt3b-seed1';v=HERE/'variants/fixed_window_dot8x32'
records=[]
for f,n in [(base/'dw8_stream_physical.peri.xml','dw8_stream_physical.peri.xml'),(base/'dw8_stream_physical.sdc','dw8_stream_physical.sdc'),(base/'strict_timing.tcl','strict_timing.tcl'),(v/'window_physical_probe.sv','dw8_stream_physical_probe.sv'),(v/'fixed_window_dot8x32.sv','fixed_window_dot8x32.sv')]:
 dst=r/n;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
ns='{http://www.efinixinc.com/enf_proj}';ET.register_namespace('efx',ns[1:-1]);tree=ET.parse(base/'dw8_stream_physical.xml');di=tree.find(ns+'design_info')
for e in list(di):
 if e.tag==ns+'design_file':di.remove(e)
for i,n in enumerate(('fixed_window_dot8x32.sv','dw8_stream_physical_probe.sv')):di.insert(i+1,ET.Element(ns+'design_file',name=n,version='default',library='default'))
tree.write(r/'dw8_stream_physical.xml',encoding='utf-8',xml_declaration=False)
# All sites come from actual Ti60 placed DSP cells matched to their netlist.
ref=ROOT/'tmp/efinity-shared-rq-dsp-input-absorb-full-seed1/outflow'
net=ref/'pw_shared_rq_physical.map.v';loc=ref/'pw_shared_rq_physical.out.placeloc'
names=set()
for line in net.read_text().splitlines():
 z=line.split()
 if len(z)>1 and z[0] in ('EFX_DSP24','EFX_DSP48'):names.add(z[1].lstrip(chr(92)))
sites=collections.defaultdict(set)
for line in loc.read_text().splitlines():
 z=line.split()
 if len(z)==4 and z[0] in names:sites[int(z[1])].add(int(z[2]))
assert len(sites)==10 and all(len(s)==16 for s in sites.values())
chains=[]
for x,ys in sorted(sites.items()):
 ys=sorted(ys);assert all(b-a==20 for a,b in zip(ys,ys[1:]))
 chains.extend([[(x,y) for y in ys[:8]],[(x,y) for y in ys[8:]]])
lines=[]
for row in range(4):
 for pair in range(4):
  for tap,(x,y) in enumerate(chains[row*4+pair]):
   for sub in (0,1):lines.append(f'arithmetic/ROW[{row}].CHANNEL[{pair*2+sub}].TAP[{tap}].mac {x} {y} {sub}')
(r/'paired_dsp.placeloc').write_text('\n'.join(lines)+'\n')
(r/'efx_pnr_settings.ini').write_text('loc_assignment = paired_dsp.placeloc\ndump_placeloc = on\n')
(r/'DO_NOT_DOWNLOAD.txt').write_text('Synthetic internal stimulus. OFFLINE ONLY. No bitstream or download allowed.\n')
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),site_reference=dict(netlist=str(net),netlist_sha256=sha(net),placement=str(loc),placement_sha256=sha(loc)),fixed_locations_sha256=sha(r/'paired_dsp.placeloc')),indent=2)+'\n')
stages=[]
for flow,budget in [('map',150),('interface',150),('pnr',240),('sta_tclsh',120)]:
 extra=' --tcl_script strict_timing.tcl' if flow=='sta_tclsh' else ''
 script=r/(flow+'.cmd');script.write_text('@echo off\ncall "'+str(EF/'bin/setup.bat')+'"\nif errorlevel 1 exit /b 1\nset "EFINITY_COLOR_PRINTING=0"\ncall efx_run.bat dw8_stream_physical.xml --flow '+flow+extra+' --timeout '+str(budget)+'\nexit /b %errorlevel%\n')
 try:
  body,seconds=run(['cmd.exe','/d','/c',str(script)],r,flow+'.log',timeout=budget+30)
  assert ('The entire flow of EFX_STA took' in body if flow=='sta_tclsh' else re.search(flow+r'\s*:\s*PASS',body)),body[-1500:]
  stages.append(dict(flow=flow,status='PASS',seconds=seconds,log_sha256=sha(r/(flow+'.log'))))
 except Exception as ex:
  stages.append(dict(flow=flow,status='FAIL',error=str(ex)));break
 (r/'stages.json').write_text(json.dumps(stages,indent=2)+'\n')
checks={};physical_sites={}
if len(stages)>=3 and stages[2]['status']=='PASS':
 for line in (r/'outflow/dw8_stream_physical.out.placeloc').read_text().splitlines():
  z=line.split()
  if len(z)==4 and re.fullmatch(r'arithmetic/ROW\[\d\].CHANNEL\[\d\].TAP\[\d\].mac',z[0]):physical_sites.setdefault(','.join(z[1:3]),[]).append(z[3])
 assert len(physical_sites)==128 and all(sorted(v)==['0','1'] for v in physical_sites.values())
if len(stages)==4 and stages[3]['status']=='PASS':
 ct=r/'outflow/dw8_stream_physical.check_timing.rpt';body=ct.read_text()
 for key in ('no_clock','unconstrained_internal_endpoints','no_input_delay','no_output_delay','multiple_clock'):
  match=re.search(r'checking '+key+r' \((\d+)\)',body);assert match and int(match[1])==0
 checks['check_timing_sha256']=sha(ct)
 for kind in ('setup','hold','process150','process150_hold'):
  f=r/f'outflow/dw8_stream_physical.strict_{kind}.rpt';m=re.search(r'Slack\s*:\s*([-+0-9.]+)',f.read_text());assert m
  checks[kind]=dict(slack_ns=float(m[1]),report_sha256=sha(f))
for x in records:assert sha(x['snapshot'])==sha(x['source'])==x['sha256']
result=dict(status='FIXED_WINDOW_PHYSICAL_PROBE_FINISHED',stages=stages,physical_dsp=len(physical_sites),checks=checks,
 strict_sta_pass=bool(checks) and all(v['slack_ns']>=0 for k,v in checks.items() if isinstance(v,dict)),whole_board=False,real_supply=False,download_allowed=False,inputs_sha256=sha(r/'inputs.json'))
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
