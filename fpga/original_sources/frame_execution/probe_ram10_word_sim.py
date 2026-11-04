"""A/B public simple dual-port RAM word storage, preserving all event timing.

Only equal read/write widths use the word path. Mixed widths keep official code.
Partial-byte writes, reset, polarity, #0 ordering and X/Z handling remain checked.
"""
import argparse,json,os,re,subprocess,time
from pathlib import Path
from run_units import ROOT,EF,MS,sha


def word_view(original):
    marker='reg mem [MEMORY_SIZE-1:0];'
    assert original.count(marker)==1
    text=original.replace(marker,marker+'''
   localparam WORD_MODE=READ_WIDTH==WRITE_WIDTH;
   // One-bit configurations never use half writes. Keep even the unused
   // task declarations legal for Questa's constant-width elaboration.
   localparam HALF_WORD_WIDTH=WRITE_WIDTH>1 ? WRITE_WIDTH/2 : 1;
   reg [READ_WIDTH-1:0] word_mem [0:(1<<READ_AWIDTH)-1];
   localparam [10239:0] WORD_INIT={'''+','.join(f'INIT_{i:X}' for i in reversed(range(40)))+'''};
   initial if(WORD_MODE)begin
      for(integer wi=0;wi<(1<<READ_AWIDTH);wi=wi+1)
         word_mem[wi]=WORD_INIT[wi*READ_WIDTH+:READ_WIDTH];
   end
''')
    expressions={
        'read_ram':'rdata=word_mem[addr];',
        'write_ram':'word_mem[addr]=wdata;',
        'write_ram_lo':'word_mem[addr][0+:HALF_WORD_WIDTH]=wdata[0+:HALF_WORD_WIDTH];',
        'write_ram_hi':'word_mem[addr][WRITE_WIDTH/2+:HALF_WORD_WIDTH]=wdata[WRITE_WIDTH/2+:HALF_WORD_WIDTH];'}
    for task,statement in expressions.items():
        pattern=rf'(task {task};.*?begin\s*)(for \(i=.*?;\s*)(end\s*endtask)'
        condition='WORD_MODE && WRITE_WIDTH>1' if task in ('write_ram_lo','write_ram_hi') else 'WORD_MODE'
        text,count=re.subn(pattern,lambda m:m[1]+'if('+condition+') '+statement+'\n else '+m[2]+m[3],text,count=1,flags=re.S)
        assert count==1,task
    return text


TB='''`timescale 1ns/1ps
module tb_ram10;
 parameter MODE=0,STEPS=10000,N=74;
 reg clk=0,other_clk=0;always #5 clk=~clk;always #7 other_clk=~other_clk;
 reg [12:0] wa=0,ra=0;reg [19:0] data=0;
 reg [1:0] we=0;reg wce=1,wae=1,re=1,rae=1,rst=0;
 integer step=0,seed=13571,checks=0;
 for(genvar k=0;k<N;k=k+1)begin:G
  localparam SEL=k%8;
  localparam WIDTH=SEL==0?1:SEL==1?2:SEL==2?4:SEL==3?5:SEL==4?8:SEL==5?10:SEL==6?16:20;
  localparam RW=k==72?5:k==73?8:WIDTH;
  localparam WW=k==72?10:k==73?16:WIDTH;
  localparam RAW=RW==1?13:RW==2?12:(RW==4||RW==5)?11:(RW==8||RW==10)?10:9;
  localparam WAW=WW==1?13:WW==2?12:(WW==4||WW==5)?11:(WW==8||WW==10)?10:9;
  wire [RW-1:0] qa,qb;
  __INSTANCES__
  if(MODE==0)always @(posedge clk or negedge clk or posedge other_clk or negedge other_clk)begin
   #0.001;if(qa!==qb)$fatal(1,"RAM10_WORD_MISMATCH cfg=%0d step=%0d old=%h new=%h",k,step,qa,qb);
   checks=checks+1;
  end
 end
 initial begin
  for(step=0;step<STEPS;step=step+1)begin
   @(negedge clk);#1;
   wa=step;ra=step;we=0;
   if(step>=1024)begin
    wa=$random(seed);ra=step%3==0?wa:$random(seed);data=$random(seed);we=step%4;
    wce=step%11!=0;wae=step%13!=0;re=step%17!=0;rae=step%19!=0;rst=step%101==0;
   end
   if(MODE==0 && step>8000)begin
    if(step%23==0)wa[3]=1'bx;if(step%29==0)ra[4]=1'bz;
    if(step%31==0)data[3]=1'bx;if(step%37==0)data[8]=1'bz;
    if(step%41==0)we[0]=1'bx;if(step%43==0)we[1]=1'bz;
    if(step%47==0)wce=1'bx;if(step%53==0)re=1'bz;
    if(step%59==0)wae=1'bx;if(step%61==0)rae=1'bz;if(step%67==0)rst=1'bx;
   end
  end
  repeat(4)@(negedge clk);#1;
  $display("PASS RAM10_WORD mode=%0d configs=%0d steps=%0d checks=%0d",MODE,N,STEPS,checks);$finish;
 end
endmodule
'''
PARAMS='''.READ_WIDTH(RW),.WRITE_WIDTH(WW),.OUTPUT_REG((k/8)%2),
 .WRITE_MODE(k>=72?"READ_FIRST":(k/24)%3==0?"READ_FIRST":(k/24)%3==1?"WRITE_FIRST":"READ_UNKNOWN"),
 .RESET_RAM((k/8)%3==0?"ASYNC":(k/8)%3==1?"SYNC":"NONE"),
 .RESET_OUTREG(k%2==0?"ASYNC":"NONE"),
 .WCLK_POLARITY(k%2),.RCLK_POLARITY(k%2),.WCLKE_POLARITY(k%2),
 .RE_POLARITY(k%2),.RST_POLARITY(k%2),.WADDREN_POLARITY(k%2),.RADDREN_POLARITY(k%2),.WE_POLARITY(k%4),
 .INIT_0(256'hxz3456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef012),
 .INIT_27(256'h89abcdef0123456789abcdef0123456789abcdef0123456789abcdef01234567)'''
PORTS='''.WCLK(clk),.RCLK(k%3==0?other_clk:clk),.WE(we),.WCLKE(wce),.WADDREN(wae),
 .WDATA(data[WW-1:0]),.WADDR(wa[WAW-1:0]),.RE(re),.RST(rst),.RADDREN(rae),.RADDR(ra[RAW-1:0]),.RDATA({out})'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--sim-bin',type=Path,default=MS);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    original=EF/'sim_models/verilog/efx_ram10.v';source=original.read_text()
    (root/'official.v').write_text(source.replace('EFX_RAM10','EFX_RAM10_REF'))
    (root/'word.v').write_text(word_view(source))
    instances='\n'.join(f'if(MODE!={exclude})begin:B{out}\n{module} #({PARAMS}) u({PORTS.format(out=out)});end' for exclude,module,out in [(2,'EFX_RAM10_REF','qa'),(1,'EFX_RAM10','qb')])
    (root/'tb.sv').write_text(TB.replace('__INSTANCES__',instances))
    inputs=dict(official_sha256=sha(original),word_sha256=sha(root/'word.v'),tb_sha256=sha(root/'tb.sv'),
                runner_sha256=sha(__file__),simulator_bin=str(a.sim_bin.resolve()),hardware_change=False)
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    sim=a.sim_bin.resolve();env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    def run(name,cmd):
        start=time.monotonic()
        with(root/(name+'.log')).open('wb')as log:r=subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
        body=(root/(name+'.log')).read_text(errors='replace')
        assert not r.returncode and not re.search(r'\*\* (?:Error|Fatal)|^# ERROR:',body,re.M),body[-2500:]
        return dict(seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log'))),body
    run('vlib',[sim/'vlib.exe','work']);run('compile',[sim/'vlog.exe','-sv','-timescale','1ns/1ps','official.v','word.v','tb.sv'])
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    measurements={}
    for mode,name in [(0,'equivalence'),(1,'official_speed'),(2,'word_speed')]:
        row,body=run(name,[sim/'vsim.exe','-c',*(['-voptargs=-O5'] if 'questa' in str(sim).lower() else []),'work.tb_ram10',f'-gMODE={mode}',f'-gSTEPS={10000 if mode==0 else 30000}','-do','run.do'])
        assert f'PASS RAM10_WORD mode={mode}' in body
        measurements[name]=row;print(name,round(row['seconds'],3),flush=True)
    result=dict(status='PUBLIC_RAM10_WORD_EQUIVALENCE_PASS',inputs_sha256=sha(root/'inputs.json'),measurements=measurements,
                host_speed_ratio=measurements['official_speed']['seconds']/measurements['word_speed']['seconds'],
                whole_board_speed_measured=False,hardware_change=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
