"""Freeze and validate the separate-to-combined AXI adapter in Questa."""
import argparse,json,os,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
QUESTA=Path('E:/quatra/questa_fse/win64')
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--compile-only',action='store_true');p.add_argument('--resume',action='store_true')
 p.add_argument('--bridge',type=Path);p.add_argument('--legacy-simulator',action='store_true')
 p.add_argument('--compare-reference',type=Path)
 p.add_argument('--valid-payload-equivalence',action='store_true',help='Compare AXI data on every valid cycle, including stalls; all control outputs remain unmasked')
 p.add_argument('--late-bready',action='store_true');p.add_argument('--dependent-wready',action='store_true')
 p.add_argument('--expect-read-buffered',action='store_true')
 p.add_argument('--mapped-models',action='store_true',help='Add public primitives for a mapped bridge netlist replay')
 p.add_argument('--write-gap',type=int,default=11);p.add_argument('--expect-buffered',action='store_true');a=p.parse_args()
 # The production default collects sparse writes before issuing native AW.
 # Old frozen negative controls remain selectable explicitly through --bridge.
 if not a.bridge:a.expect_buffered=True
 simulator=Path('E:/altera/15.0/modelsim_ase/win32aloem') if a.legacy_simulator else QUESTA
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp')
 if not a.resume:
  root.mkdir(parents=True,exist_ok=False);records=[]
  for name in ('ddr516_axi_address_bridge.sv','tb_ddr516_axi_address_bridge.sv'):
   src=a.bridge.resolve() if a.bridge and name=='ddr516_axi_address_bridge.sv' else HERE/name;dst=root/name;shutil.copy2(src,dst)
   records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  if a.mapped_models:
   for name in ('efx_lut4.v','efx_ff.v','efx_ram10.v','efx_add.v'):
    src=EF/'sim_models/verilog'/name;dst=root/name;shutil.copy2(src,dst)
    records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  if a.compare_reference:
   reference=a.compare_reference.resolve();original=reference.read_text()
   assert original.count('module ddr516_axi_address_bridge(')==1
   ref=root/'reference_bridge.sv';ref.write_text(original.replace('module ddr516_axi_address_bridge(','module reference_bridge('))
   declarations=[];connections=[];outputs=[]
   for line in original.split(');',1)[0].splitlines():
    m=re.match(r'\s*(input|output)\s+(?:wire|reg)\s+(\[[^]]+\]\s*)?(.+)',line)
    if not m:continue
    names=[n.strip() for n in m[3].rstrip(',').split(',')]
    for name in names:connections.append(f'.{name}({"ref_" if m[1]=="output" else ""}{name})')
    if m[1]=='output':
     declarations.append('wire '+(m[2] or '')+','.join('ref_'+n for n in names)+';');outputs+=names
   assert len(outputs)==26,len(outputs)
   probe='\n'.join(declarations)+'\nreference_bridge reference_dut('+','.join(connections)+');\n'
   probe+='integer equivalence_samples=0;\nalways @(posedge clk or negedge clk)begin #1;\n'
   payload_valid={n:'m_wvalid' for n in ('m_wdata','m_wstrb','m_wlast')}
   payload_valid.update({n:'s_rvalid' for n in ('s_rid','s_rresp','s_rlast','s_rdata')})
   for name in outputs:
    condition=''
    if a.valid_payload_equivalence and name in payload_valid:
     condition='('+payload_valid[name]+' || ref_'+payload_valid[name]+') && '
    probe+=f'if({condition}({name}!==ref_{name}))$fatal(1,"Bridge equivalence failed: {name}");\n'
   probe+='equivalence_samples=equivalence_samples+1;end\n'
   tb=root/'tb_ddr516_axi_address_bridge.sv';body=tb.read_text()
   assert body.count('endmodule')==1
   body=body.replace('endmodule',probe+'\nendmodule').replace('  $display("PASS DDR516_AXI_BRIDGE',
    '  $display("PASS DDR516_BUFFER_EQUIVALENCE samples=%0d",equivalence_samples);\n  $display("PASS DDR516_AXI_BRIDGE')
   tb.write_text(body);records[1]['sha256']=sha(tb)
   records.append(dict(source=str(reference),snapshot=str(ref),sha256=sha(ref),reference_original_sha256=sha(reference)))
  (root/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 meta=json.loads((root/'inputs.json').read_text())
 for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256']
 env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 def run(args,name):
  start=time.monotonic()
  with (root/name).open('w') as f:
   result=subprocess.run(list(map(str,args)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=90)
  body=(root/name).read_text(errors='replace')
  assert result.returncode==0 and not re.search(r'(?m)^#?\s*\*\* (?:Error|Fatal)',body),body[-4000:]
  return body,time.monotonic()-start
 if not a.resume:
  run([simulator/'vlib.exe','work'],'vlib.log')
  body,_=run([simulator/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in meta['inputs']]],'compile.log')
  assert 'Errors: 0, Warnings: 0' in body
 if a.compile_only:print('DDR516_AXI_BRIDGE_COMPILE_PASS');return
 results=[]
 for seed in (71,391,1027,4201):
  body,secs=run([simulator/'vsim.exe','-c','work.tb_ddr516_axi_address_bridge',f'-gSEED={seed}',
                f'-gLATE_BREADY={int(a.late_bready)}',f'-gDEPENDENT_WREADY={int(a.dependent_wready)}',
                f'-gWRITE_GAP={a.write_gap}',f'-gEXPECT_BUFFERED={int(a.expect_buffered)}',
                f'-gEXPECT_READ_BUFFERED={int(a.expect_read_buffered)}',
                '-do','onerror {quit -code 1}; run -all; quit -code 0'],f'seed{seed}.log')
  lines=re.findall(r'^# (PASS DDR516_AXI_BRIDGE .*)$',body,re.M);assert len(lines)==1
  if a.compare_reference:assert 'PASS DDR516_BUFFER_EQUIVALENCE samples=' in body
  results.append(dict(seed=seed,result=lines[0],seconds=secs,log_sha256=sha(root/f'seed{seed}.log')))
 result=dict(status='DDR516_AXI_BRIDGE_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),late_bready=a.late_bready,dependent_wready=a.dependent_wready,
             write_gap=a.write_gap,expect_buffered=a.expect_buffered,
             expect_read_buffered=a.expect_read_buffered,
             mapped_public_primitives=a.mapped_models,
             cycle_equivalence_reference=str(a.compare_reference.resolve()) if a.compare_reference else None,
             equivalence_valid_payload_only=a.valid_payload_equivalence,
             scope='Adapter protocol test, not DDR PHY or integrated board')
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
