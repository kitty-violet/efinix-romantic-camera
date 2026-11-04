"""Simulate exact edited native byte/gain and handshake cells, not RTL substitutes."""
import argparse,json,pickle,re,shutil,os,time
from pathlib import Path
from run_units import ROOT,EF,run,sha
from native_local_drivers import CELL,graph
from probe_full_native_split_map import params
from continuous_control_native import net
S='raw_frontend/consumer/serializer/'
G='raw_frontend/consumer/debayer/legacy_debayer/inst_rgb_gain/'

def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 native=a.native.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
 proof=json.loads((native/'input-isp-edit.json').read_text());assert sha(native/'native_source.v')==proof['candidate_sha256']
 source=(native/'native_source.v').read_text();g=graph(source);ps=params(source)
 (native/'isp-source-parse.pkl').write_bytes(pickle.dumps((sha(native/'native_source.v'),g,ps),protocol=4))
 control=['LUT__153491','LUT__153492','LUT__153493','LUT__153494','LUT__153496','LUT__153497','LUT__153498','LUT__153505','LUT__153506']
 selected=sorted([n for n in g if n.startswith(G)]+control)
 assert len(selected)==145
 ts={t for n in selected for vs in g[n][1].values() for t in vs if t[0]=='net'}
 names={t:'n'+str(i) for i,t in enumerate(sorted(ts,key=str))}
 def tok(t):return t[1] if t[0]=='literal' else names[t]
 body=['module actual(input wire[15:0] raw,input wire[9:0] ctl,output wire[7:0] high_byte,low_byte,output wire[5:0] flags);','wire '+','.join(names.values())+';']
 for i in range(16):body.append(f'assign {names[net("raw_frontend/consumer/rd",i)]}=raw[{i}];')
 controls=[net(S+'pair_fifo/queue_count',0),net(S+'pair_fifo/queue_count',1),net(S+'second_pending'),net('video_processing_pipeline/downscaler/pair_pending'),net('video_processing_pipeline/n19'),net('video_processing_pipeline/downscaler/source_x_phase',0),net('video_processing_pipeline/downscaler/source_x_phase',1),net('service_isp_ready'),net('service_isp_sof')]
 for i,t in enumerate(controls):body.append(f'assign {names[t]}=ctl[{i}];')
 body.append(f'assign {names[net("service_isp_fire")]}=ctl[9] & {names[net("n74389_2")]};')
 for i in range(8):
  body.append(f'assign high_byte[{i}]={names[net(G+"mul_high",i+8)]};assign low_byte[{i}]={names[net(G+"mul_low",i+8)]};')
 outputs=['~ceg_net7945',S+'pair_fifo/dequeue','video_processing_pipeline/downscaler/n2309','video_processing_pipeline/downscaler/n2624','video_processing_pipeline/downscaler/n2625','video_processing_pipeline/downscaler/n2628']
 for i,n in enumerate(outputs):body.append(f'assign flags[{i}]={names[net(n)]};')
 def param(x):return str(x) if not isinstance(x,tuple) else f"{x[0]}'h{x[1]:x}"
 for i,n in enumerate(selected):
  k,pins=g[n];pins=dict(pins)
  for port in (['I0','I1','CI','O','CO'] if k=='EFX_ADD' else ['I0','I1','I2','I3','O']):
   if port not in pins:pins[port]=[] if port in ['O','CO'] else [('literal',"1'b0",None)]
  args=','.join('.'+p+'('+('' if not vs else tok(vs[0]) if len(vs)==1 else '{'+','.join(tok(t) for t in vs)+'}')+')' for p,vs in pins.items())
  body.append(k+' #('+','.join('.'+k+'('+param(v)+')' for k,v in ps[n].items())+f') c{i}('+args+');')
 body.append('endmodule');(root/'actual.sv').write_text('\n'.join(body))
 (root/'tb.sv').write_text('''`timescale 1ns/1ps
module tb;
 reg[15:0] raw=0;reg[9:0] ctl=0;wire[7:0] high_byte,low_byte;wire[5:0] flags;
 reg[5:0] want;reg take;integer hi,lo;
 actual dut(.*);
 initial begin
  for(integer i=0;i<65536;i++)begin
   raw=i;hi=((i&255)*255)>>8;lo=((i>>8)*255)>>8;#1;
   if(high_byte!==8'(hi) || low_byte!==8'(lo))$fatal(1,"NATIVE_PAIR_SWAP %h %h/%h",raw,high_byte,low_byte);
  end
  for(integer i=0;i<1024;i++)begin
   ctl=i;take=ctl[9] && !ctl[3] && ctl[4];
   want[0]=(|ctl[2:0]) && ctl[7];want[1]=(|ctl[1:0]) && !ctl[2] && ctl[7];
   want[2]=ctl[8] && take;want[3]=!(|ctl[6:5]) && take;want[4]=ctl[6] && take;want[5]=ctl[5] && !ctl[6] && take;
   #1;if(flags!==want)$fatal(1,"NATIVE_HANDSHAKE ctl=%h got=%b want=%b",ctl,flags,want);
  end
  $display("PASS NATIVE_INPUT_CUT raw_pairs=65536 channel_checks=131072 handshake_combinations=1024");$finish;
 end
endmodule
''')
 for n in ['efx_add.v','efx_lut4.v']:shutil.copy2(EF/'sim_models/verilog'/n,root/n)
 save=dict(selected_cells=selected,source_sha256=sha(native/'native_source.v'),runner_sha256=sha(__file__),generated={f.name:sha(f) for f in root.iterdir()})
 (root/'inputs.json').write_text(json.dumps(save,indent=2));start=time.monotonic()
 ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
 try:
  run([ms/'vlib.exe','work'],root,'vlib.log',20)
  run([ms/'vlog.exe','-sv','-timescale','1ns/1ps','efx_add.v','efx_lut4.v','actual.sv','tb.sv'],root,'compile.log',30)
  body,_=run([ms/'vsim.exe','-c','work.tb','-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log',30)
  assert 'PASS NATIVE_INPUT_CUT' in body
  result=dict(status='PASS',lines=re.findall(r'^# PASS.*$',body,re.M))
 except Exception as e:result=dict(status='FAIL',error=str(e))
 result.update(seconds=time.monotonic()-start,inputs_sha256=sha(root/'inputs.json'),hardware_access=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
 return result['status']=='PASS'
if __name__=='__main__':raise SystemExit(0 if main() else 1)
