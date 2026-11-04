"""RAW10-only documented CSI parameter: paired encrypted models, A/B + oracle.

Only the public PACK_TYPE changes. All PHY timing, clocks and pins stay intact.
This is a bounded digital packet test, never board electrical qualification.
"""
import argparse,json,os,re,shutil,subprocess,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from prepare_full_board_simulation import compact,envelopes

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--board',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();board=a.board.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 component=EF/'ipm/ip/efx_csi2/efx_csi2_rx/ipm/ip_component.xml'
 tree=ET.parse(component).getroot();ns={'x':tree.tag.split('}')[0][1:]}
 parameter=next(v for v in tree.findall('./x:parameters/x:parameter',ns) if v.find('x:name',ns).text=='PACK_TYPE')
 desc=parameter.find('x:description',ns).text;expr=parameter.find('x:value',ns).text
 assert 'Pack 40 RAW10' in desc and expr=='(${Pack_64}*8)+(${Pack_56}*4)+(${Pack_48}*2)+(${Pack_40}*1)'
 csi=board/'rtl/csi_rx_controller.sv';raw=csi.read_text();norm=compact(raw)
 wrapper=raw[:raw.index('endmodule')+len('endmodule')]
 assert wrapper.count('.PACK_TYPE (15)')==1
 modified=raw.replace('.PACK_TYPE (15)', '.PACK_TYPE (1)',1)
 (root/'csi_rx_controller_raw10.sv').write_text(modified)
 # Exact installed synthesis pairing is required before choosing ModelSim views.
 files=json.loads((ROOT/'tmp/frame-execution/csi-official-bundle1/result.json').read_text())['files']
 paired=[];blocks=set();simtext=[wrapper.replace('module csi_rx_controller','module csi_rx_controller #(parameter integer PACK_TEST=15)',1).replace('.PACK_TYPE (15)','.PACK_TYPE (PACK_TEST)',1)]
 for rec in files:
  sim=Path(rec['path']);syn=Path(str(sim).replace('source_modelsim','source_efinity'));assert sha(sim)==rec['sha256']
  s=syn.read_text();enc=envelopes(s)
  assert all(compact(b) in norm for b in enc)
  blocks.update(map(compact,enc))
  if 'module ' in s:assert compact(s[s.index('module '):s.rfind('endmodule')+len('endmodule')]) in norm
  assert enc or 'module ' in s
  paired.append(dict(synthesis=str(syn),synthesis_sha256=sha(syn),simulation=str(sim),simulation_sha256=sha(sim)))
  simtext.append(sim.read_text())
 assert set(map(compact,envelopes(raw)))==blocks
 (root/'csi_rx_controller_sim.sv').write_text('\n'.join(simtext))
 fixture=(HERE/'board_integration/camera_raw10_fixture.sv').read_text()
 fixture=fixture.replace('hs=1;de=1;pdata={24\'d0,{4{10\'h200}}};\n   repeat(480)@(negedge pixel_clk);', '''hs=1;de=1;
   for(integer x=0;x<480;x=x+1)begin
    for(integer lane=0;lane<4;lane=lane+1)pdata[lane*10+:10]=10'(f*97+y*17+x*4+lane*251);
    @(negedge pixel_clk);
   end''')
 assert 'lane*251' in fixture
 (root/'camera_raw10_fixture.sv').write_text(fixture)
 tb='''`timescale 1ns/1ps
module tb;
reg clk=0,bc=0,pc=0,rst=0;
always #5 clk=~clk;always #7.142857 bc=~bc;always #10 pc=~pc;
wire cp,cn;wire [3:0] dp,dn,he;wire [7:0] d0,d1,d2,d3;
camera_raw10_fixture #(.LINES(20),.FRAMES(2),.FRAME_TICKS(35000)) cam(
 .clk(clk),.byte_clk(bc),.pixel_clk(pc),.reset_n(rst),.start(1'b1),
 .lp_cp(cp),.lp_cn(cn),.lp_dp(dp),.lp_dn(dn),.d0(d0),.d1(d1),.d2(d2),.d3(d3),.hs_enable(he));
wire [63:0] data[0:1];wire [1:0] valid,hs,vs;
wire [3:0] ppc[0:1];wire [5:0] dt[0:1];wire [31:0] debug[0:1];
'''
 header=wrapper.split('module csi_rx_controller',1)[1].split(');',1)[0]
 for i in range(2):
  conn={'reset_n':'rst','reset_byte_HS_n':'rst','reset_pixel_n':'rst','clk':'clk','clk_byte_HS':'bc','clk_pixel':'pc','axi_clk':'clk','axi_reset_n':'rst',
   'Rx_LP_CLK_P':'cp','Rx_LP_CLK_N':'cn','Rx_LP_D_P':'dp','Rx_LP_D_N':'dn',**{f'Rx_HS_D_{j}':f'd{j}' for j in range(4)},
   'fifo_rd_empty':"4'd0",'pixel_data':f'data[{i}]','pixel_data_valid':f'valid[{i}]','hsync_vc0':f'hs[{i}]','vsync_vc0':f'vs[{i}]',
   'pixel_per_clk':f'ppc[{i}]','datatype':f'dt[{i}]','mipi_debug_out':f'debug[{i}]'}
  ports=['.'+name+'('+conn.get(name,"'0" if direction=='input' else '')+')' for direction,width,name in re.findall(r'\b(input|output)\s+(\[[^]]+\])?\s*(\w+)',header)]
  tb+=f'csi_rx_controller #(.PACK_TEST({15 if i==0 else 1})) rx{i}('+','.join(ports)+');\n'
 tb+='''reg [39:0] expected[0:19199];integer sent=0,received=0,edges=0;reg lastvs=0;
always @(posedge pc)if(rst && cam.de)begin
 if(sent>=19200)$fatal(1,"oracle overflow");expected[sent]=cam.pdata[39:0];sent=sent+1;
end
always @(negedge pc)if(rst)begin
 if(valid[0]!==valid[1] || hs[0]!==hs[1] || vs[0]!==vs[1])$fatal(1,"RAW10 format timing mismatch");
 if(vs[0] && !lastvs)edges=edges+1;lastvs=vs[0];
 if(valid[0] && hs[0])begin
  if(data[0]!==data[1] || ppc[0]!==ppc[1] || dt[0]!==dt[1])$fatal(1,"RAW10 A/B payload mismatch");
  if(received>=sent || data[0][39:0]!==expected[received] || ppc[0]!==4 || dt[0]!==6'h2b)
   $fatal(1,"RAW10 oracle mismatch index=%0d sent=%0d actual=%h expected=%h ppc=%0d dt=%h",received,sent,data[0],expected[received],ppc[0],dt[0]);
  received=received+1;
 end
end
initial begin
 #200;rst=1;
 wait(received==19200);repeat(3000)@(negedge pc);
 if(sent!=19200 || edges!=2 || vs[0])$fatal(1,"RAW10 frame closure sent=%0d edges=%0d vs=%b",sent,edges,vs);
 // Reset both actual vendor state machines, then confirm a clean inactive view.
 rst=0;repeat(30)@(negedge pc);rst=1;repeat(6000)@(negedge pc);
 if(valid!==0 || hs!==0 || vs!==0)$fatal(1,"RAW10 reset did not quiesce");
 $display("PASS CSI_RAW10_SPECIALIZATION words=%0d pixels=%0d frames=%0d reset=1",received,received*4,edges);$finish;
end
initial begin #2200000;$fatal(1,"RAW10 timeout sent=%0d received=%0d",sent,received);end
endmodule
'''
 (root/'tb.sv').write_text(tb)
 tx=EF/'ipm/ip/efx_csi2/testbench/soft_dphy/rx/efx_csi2_tx_modelsim.sv'
 sources=[root/'csi_rx_controller_sim.sv',root/'camera_raw10_fixture.sv',tx,*sorted((EF/'sim_models/verilog').glob('*.v')),root/'tb.sv']
 records=[];frozen=[]
 for src in sources:
  dest=root/src.name
  if src!=dest:shutil.copy2(src,dest)
  frozen.append(dest);records.append(dict(source=str(src),snapshot=str(dest),sha256=sha(dest)))
 manifest=dict(inputs=records,board=str(board),board_csi_sha256=sha(csi),specialized_csi_sha256=sha(root/'csi_rx_controller_raw10.sv'),official_component=str(component),official_component_sha256=sha(component),parameter_description=desc,parameter_expression=expr,paired=paired,runner_sha256=sha(__file__))
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2))
 sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env.update(SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat',TEMP=str(root),TMP=str(root))
 start=time.monotonic()
 try:
  for name,args in [('vlib',[sim/'vlib.exe','work']),('compile',[sim/'vlog.exe','-sv',*frozen]),('simulation',[sim/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'])]:
   with (root/(name+'.log')).open('wb') as f:r=subprocess.run(list(map(str,args)),cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
   body=(root/(name+'.log')).read_text(errors='replace')
   assert not r.returncode and not re.search(r'(?m)^#?\s*\*\* (Error|Fatal)',body),body[-4000:]
  line=re.findall(r'^# (PASS CSI_RAW10_SPECIALIZATION.*)',body,re.M);assert len(line)==1
  for rec in records:assert sha(rec['snapshot'])==rec['sha256']
  result=dict(status='CSI_RAW10_SPECIALIZATION_PASS',line=line[0],seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),specialized_csi_sha256=manifest['specialized_csi_sha256'],full_board=False,physical_electrical=False)
  (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 except Exception:
  (root/'failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),seconds=time.monotonic()-start),indent=2));raise
if __name__=='__main__':main()
