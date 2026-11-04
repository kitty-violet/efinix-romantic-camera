"""Four-state reference equivalence of residual read address cursors."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
from simulation_primitives import with_public_lut4
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--variant',type=Path,default=HERE/'variants/fixed_skip_cursor');p.add_argument('--plain-interface',action='store_true');p.add_argument('--preoffset-interface',action='store_true');a=p.parse_args()
assert not (a.plain_interface and a.preoffset_interface)
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
try:
 v=a.variant.resolve();ref=HERE/'variants/fixed_residual_operand_share/fixed_skip_vectors4.sv';records=[]
 for src in with_public_lut4((ref,v/'fixed_skip_vectors4.sv',HERE/'variants/fixed_skip_cursor_test/tb_skip_cursor.sv')):
  dst=r/('reference_skip.sv' if src==ref else src.name)
  if src==ref:dst.write_text(src.read_text().replace('module fixed_skip_vectors4(', 'module reference_skip('))
  elif src.name=='tb_skip_cursor.sv' and (a.plain_interface or a.preoffset_interface):
   body=src.read_text();old='fixed_skip_vectors4 #(.REGISTER_ADDRESSES(1)) dut(.prepare(prepare),`PORTS(1));'
   assert body.count(old)==1
   new='fixed_skip_vectors4 dut(`PORTS(1));'
   if a.preoffset_interface:
    ports=next(line.split('`define PORTS(N) ',1)[1] for line in body.splitlines() if '`define PORTS(N) ' in line)
    ports=ports.replace('[N]','[1]').replace('.skip_bases(skip_bases)','.skip_bases(prepared_bases)')
    new="wire [43:0] prepared_bases; for(genvar r=0;r<4;r=r+1)begin:PREP assign prepared_bases[r*11+:11]={1'b0,skip_bases[r*10+:10]}+(offset_two ? 11'd4:11'd0);end\n"
    new+='fixed_skip_vectors4 #(.BASE_WIDTH(11),.OFFSET_PREAPPLIED(1)) dut('+ports+');'
   dst.write_text(body.replace(old,new))
  else:shutil.copy2(src,dst)
  records.append(dict(source=str(src),snapshot=str(dst),source_sha256=sha(src),sha256=sha(dst)))
 (r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__)),indent=2))
 run([MS/'vlib.exe','work'],r,'vlib.log')
 body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[i['snapshot'] for i in records]],r,'compile.log')
 assert 'Errors: 0, Warnings: 0' in body
 body,seconds=run([MS/'vsim.exe','-c','work.tb_skip_cursor','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=90)
 lines=re.findall(r'^# (PASS SKIP_CURSOR .*)$',body,re.M);assert len(lines)==1 and not re.search(r'\*\* (Error|Fatal)',body),body[-2000:]
 for i in records:assert sha(i['source'])==i['source_sha256'] and sha(i['snapshot'])==i['sha256']
 data=dict(status='SKIP_CURSOR_FOURSTATE_REFERENCE_PASS',line=lines[0],seconds=seconds,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),whole_cluster_proven=False)
 (r/'result.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
except Exception as e:
 (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
