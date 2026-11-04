"""Cycle-exact four-master protocol equivalence with actual board dependencies."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--rtl',type=Path);a=p.parse_args()
r=a.root.resolve();base=a.base.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
meta=json.loads((base/'inputs.json').read_text());br=json.loads((base/'result.json').read_text());assert sha(base/'inputs.json')==br['inputs_sha256']
records=[]
def freeze(src,name=None,text=None):
 d=r/(name or src.name)
 if text is None:shutil.copy2(src,d)
 else:d.write_text(text,encoding='utf-8')
 records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(d),sha256=sha(d)))
 return d
name='axi_three_master_burst_arbiter.sv';original=base/'assembly'/name
candidate=HERE/'variants/axi_response_broadcast'/name
if a.rtl:
 candidate=a.rtl.resolve();assert candidate.is_relative_to(HERE/'variants') and candidate.name==name
freeze(candidate)
ref=original.read_text().replace('module axi_three_master_burst_arbiter #','module arbiter_cycle_reference #',1)
freeze(original,'arbiter_cycle_reference.sv',ref)
ports=re.findall(r'^\s*(input|output)\s+(?:wire|reg)\s*(\[[^\]]+\])?\s*(\w+)\s*[,\n]',original.read_text(),re.M)
assert len(ports)>60 and ports[-1][2]=='protocol_error'
outs=[(w,n) for direction,w,n in ports if direction=='output']
checker='\n// Simulation-only reference checks all controls even when stalled/invalid.\n'
checker+='\n'.join('wire '+w+' ref_'+n+';' for w,n in outs)+'\n'
checker+='arbiter_cycle_reference #(.NUM_MASTERS(NUM_MASTERS),.AXI_ADDR_WIDTH(AXI_ADDR_WIDTH),.AXI_DATA_WIDTH(AXI_DATA_WIDTH),.AXI_ID_WIDTH(AXI_ID_WIDTH)) cycle_reference(\n'
checker+=',\n'.join('.'+n+'('+('ref_' if direction=='output' else '')+n+')' for direction,w,n in ports)+');\n'
checker+='integer equivalence_edges=0,equivalence_data=0;\nalways @(negedge clk)begin\n #0.001;\n equivalence_edges=equivalence_edges+1;\n'
payloads={n:'m_axi_arvalid' for n in ('m_axi_arid','m_axi_araddr','m_axi_arlen','m_axi_arsize','m_axi_arburst')}
payloads.update({n:'m_axi_awvalid' for n in ('m_axi_awid','m_axi_awaddr','m_axi_awlen','m_axi_awsize','m_axi_awburst')})
payloads.update({n:'m_axi_wvalid' for n in ('m_axi_wdata','m_axi_wstrb','m_axi_wlast')})
per_owner={'s_axi_rid':('s_axi_rvalid','AXI_ID_WIDTH'),'s_axi_rdata':('s_axi_rvalid','AXI_DATA_WIDTH'),'s_axi_rresp':('s_axi_rvalid','2'),'s_axi_rlast':('s_axi_rvalid','1'),'s_axi_bid':('s_axi_bvalid','AXI_ID_WIDTH'),'s_axi_bresp':('s_axi_bvalid','2')}
for w,n in outs:
 if n in per_owner:continue
 qualifier=payloads.get(n,'1\'b1')
 checker+=f' if({qualifier} && {n}!==ref_{n})$fatal(1,"AXI_EQ_{n}");\n'
for n,(valid,width) in per_owner.items():
 checker+=f' for(integer k=0;k<NUM_MASTERS;k=k+1)if({valid}[k] && {n}[k*{width}+:{width}]!==ref_{n}[k*{width}+:{width}])$fatal(1,"AXI_EQ_{n}");\n'
checker+=' for(integer k=0;k<NUM_MASTERS;k=k+1)if(s_axi_rvalid[k])begin\n'
checker+='  equivalence_data=equivalence_data+1;\n  if(s_axi_rdata[k*AXI_DATA_WIDTH+:AXI_DATA_WIDTH]!==ref_s_axi_rdata[k*AXI_DATA_WIDTH+:AXI_DATA_WIDTH])$fatal(1,"AXI_EQ_RDATA");\n end\nend\n'
checker+='final begin\n if(equivalence_edges<100 || equivalence_data<100)$error("AXI_EQ_COVERAGE");\n $display("PASS AXI_CYCLE_EQ edges=%0d valid_data=%0d",equivalence_edges,equivalence_data);\nend\n'
text=candidate.read_text();assert text.count('endmodule')==1
freeze(candidate,'arbiter_checked.sv',text.replace('endmodule',checker+'endmodule'))
for n in ('sapphire_axi32_to_axi128.sv','ti60_ddr_four_master_fabric.sv'):
 src=base/'assembly'/n;assert any(x['snapshot']==str(src) and x['sha256']==sha(src) for x in meta['inputs']);freeze(src)
freeze(base/'ddr_memory_map.vh')
freeze(ROOT/'hw/efinity/prototypes/ddr_fabric/tb_ti60_ddr_four_master_fabric.sv')
tb=ROOT/'hw/efinity/prototypes/video_scaler/tb_axi_arbiter_protocol.sv'
text=tb.read_text();assert text.count('i == 0 ? 127 :')==2
text=text.replace('i == 0 ? 127 :','i == 0 ? 255 :').replace('max_burst=128','max_burst=256')
freeze(tb,text=text)
(r/'inputs.json').write_text(json.dumps(dict(inputs=records,runner_sha256=sha(__file__),board=str(base),board_inputs_sha256=sha(base/'inputs.json'),candidate_sha256=sha(candidate),reference_sha256=sha(original)),indent=2)+'\n')
run([MS/'vlib.exe','work'],r,'vlib.log')
files=[x['snapshot'] for x in records if Path(x['snapshot']).suffix=='.sv' and Path(x['snapshot']).name!=name]
body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','+incdir+.',*files],r,'compile.log');assert 'Errors: 0, Warnings: 0' in body
results=[]
for top,arg in [('tb_axi_arbiter_protocol','-gN=3'),('tb_axi_arbiter_protocol','-gN=4')]+[('tb_ti60_ddr_four_master_fabric','-gRANDOM_SEED='+str(s)) for s in (71,391,1027,2039)]:
 log=top+arg.replace('-g','_')+'.log'
 body,seconds=run([MS/'vsim.exe','-c','work.'+top,arg,'-do','onerror {quit -code 1}; run -all; quit -code 0'],r,log,timeout=120)
 expected='PASS ARB_PROTOCOL' if 'arbiter_protocol' in top else 'PASS FOUR_FABRIC'
 assert expected in body and 'PASS AXI_CYCLE_EQ' in body and 'Errors: 0, Warnings: 0' in body
 results.append(dict(top=top,arg=arg,lines=re.findall(r'^# (PASS .*)$',body,re.M),seconds=seconds,log=log,log_sha256=sha(r/log)))
for x in records:assert sha(x['source'])==x['source_sha256'] and sha(x['snapshot'])==x['sha256']
result=dict(status='AXI_RESPONSE_BROADCAST_CYCLE_EQ_PASS',results=results,inputs_sha256=sha(r/'inputs.json'),candidate_sha256=sha(candidate),reference_sha256=sha(original),scope='All handshake/state/error outputs exact each falling edge; every channel payload exact whenever its VALID, including stalls. Three/four-master max256 protocol plus four seeds of actual board narrow adapter and four-master fabric.')
(r/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
