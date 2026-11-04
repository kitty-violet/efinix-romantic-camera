"""Short four-state preview-control tests; never connects to the board."""
import argparse,json,os,re,shutil,subprocess,time,traceback
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from prepare_continuous_run_control import OUT,generate

B=ROOT/'tmp/frame-execution/continuous-control-short1'
Q=Path('E:/quatra/questa_fse/win64')

def boot_tb():
    t=(HERE/'variants/fixed_boot_sequencer/tb_cnn_boot_sequencer.sv').read_text()
    t=once(t,'wire run_allowed,terminal,failed;',
           'wire [5:0] observer_state;wire run_allowed,terminal,failed;')
    t=once(t,'RUN_TIMEOUT_CYCLES(1024)','RUN_TIMEOUT_CYCLES(8192)')
    t=once(t,'mode==3 ? 17:','mode==3 ? 513:mode==16 ? 512:')
    t=once(t,'(mode==14 ? 16:mode==15 ? 1:3)',
           '(mode==14 ? 16:mode==15 ? 1:mode==16 ? 512:3)')
    # Real mailbox consumption: the host writes an explicit second command,
    # instead of relying on a state-dependent combinational auto-start.
    t=once(t,'reg verified=0,query=0,pending=0,held=0;',
           'reg [31:0] mailbox_command=0;integer clears=0;\n reg verified=0,query=0,pending=0,held=0;')
    t=once(t,"MB:response=status==1 ? (mode==1 ? 32'hbad0:32'h434e4e31):0;",
           'MB:response=mailbox_command;')
    t=once(t,'status=0;ctrl=0;ccrc=0;',
           'mailbox_command=0;clears=0;status=0;ctrl=0;ccrc=0;')
    t=once(t,'MB:if(cmd_data!=0)$fatal(1,"FAIL BOOT command clear");',
           'MB:begin if(cmd_data!=0)$fatal(1,"FAIL BOOT command clear");mailbox_command=0;clears=clears+1;end')
    t=once(t,'MB+4:status=cmd_data;',
           "MB+4:begin status=cmd_data;if(cmd_data==1)mailbox_command=mode==1 ? 32'hbad0:32'h434e4e31;end")
    t=once(t,"APB+'h1c:if(cmd_data!=1)$fatal(1,\"FAIL BOOT begin\");",
           "APB+'h1c:begin if(cmd_data!=1)$fatal(1,\"FAIL BOOT begin\");\n"
           "      loads=0;diags=0;verified=0;query=0;run_clocks=0;end")
    t=once(t,'mode<16','mode<17')
    # Restart successful normal run and prove no automatic restart follows;
    # an invalid subsequent command must still latch an unrecoverable fault.
    insert='''
   if(mode==16)begin
    repeat(100)@(negedge clk);
    if(!terminal || run_allowed || status!=5 || clears!=2)$fatal(1,"FAIL BOOT automatic restart");
    mailbox_command=32'h434e4e31;
    ticks=0;while(terminal && ticks<1000)begin @(negedge clk);ticks=ticks+1;end
    if(terminal)$fatal(1,"FAIL BOOT cannot rearm successful run");
    while(!terminal && ticks<600000)begin @(negedge clk);ticks=ticks+1;end
    if(!terminal || failed || status!=5 || clears!=3 || loads!=2560 || diags!=2560)
      $fatal(1,"FAIL BOOT restart skipped image verification or failed");
    mailbox_command=32'hbad0;
    ticks=0;while(!failed && ticks<1000)begin @(negedge clk);ticks=ticks+1;end
    while(!terminal && ticks<2000)begin @(negedge clk);ticks=ticks+1;end
    if(!failed || !terminal || fault!=1 || run_allowed)$fatal(1,"FAIL BOOT invalid restart");
    mailbox_command=32'h434e4e31;
    repeat(100)@(negedge clk);
    if(!terminal || !failed || cmd_valid || clears!=3)$fatal(1,"FAIL BOOT fault rearmed");
    $display("PASS PREVIEW_REARM explicit_start=1 second_image_verified=1 invalid_rearm_halts=1");
   end
'''
    t=once(t,'   passed=passed+1;',insert+'   passed=passed+1;')
    return t

ADMISSION_TB='''`timescale 1ns/1ps
module tb_preview_admission;
 reg clk=0;always #5 clk=~clk;
 reg reset=1,permit=0,start_request=0;reg [9:0] requested_frames=0;
 wire start_accept;wire [9:0] admitted;
 cnn_frame_admission dut(.*);
 integer n,k,starts,total=0;reg [31:0] rng=32'hfac12345;
 initial begin
  repeat(3)@(negedge clk);reset=0;
  for(n=0;n<=1023;n=n+1)begin
   permit=0;start_request=1;requested_frames=n;@(negedge clk);
   if(admitted!=0 || start_accept)$fatal(1,"FAIL ADMISSION disabled");
   permit=1;starts=0;
   for(k=0;k<(n>=1 && n<=512 ? n*3+32:40);k=k+1)begin
    rng={rng[30:0],rng[31]^rng[21]^rng[1]^rng[0]};start_request=(k>n*2)||rng[0];
    #1;if(start_accept)starts=starts+1;
    @(negedge clk);
   end
   if(starts!=(n>=1 && n<=512 ? n:0) || admitted!=starts)
    $fatal(1,"FAIL ADMISSION budget=%0d starts=%0d admitted=%0d",n,starts,admitted);
   repeat(5)begin start_request=1;#1;if(start_accept)$fatal(1,"FAIL ADMISSION overrun");@(negedge clk);end
   total=total+1;
  end
  permit=0;@(negedge clk);requested_frames=512;permit=1;
  repeat(10)@(negedge clk);reset=1;@(negedge clk);
  if(start_accept || admitted!=0)$fatal(1,"FAIL ADMISSION reset");
  $display("PASS PREVIEW_ADMISSION all_1024_budgets=1 exact_admission=1 invalid_rejected=1 reset=1");$finish;
 end
 initial begin #20000000;$fatal(1,"FAIL ADMISSION timeout");end
endmodule
'''

def main():
    global B
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=B)
    B=parser.parse_args().root.resolve();assert B.is_relative_to(ROOT/'tmp/frame-execution')
    assert not B.exists();B.mkdir()
    assert generate()==(OUT/'cnn_boot_sequencer.sv').read_text()
    sources=[OUT/'cnn_boot_sequencer.sv',OUT/'cnn_frame_admission.sv']
    for p in sources:shutil.copy2(p,B/p.name)
    (B/'tb_cnn_boot_sequencer.sv').write_text(boot_tb())
    (B/'tb_preview_admission.sv').write_text(ADMISSION_TB)
    env=dict(os.environ)
    for key in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):env[key]='E:/quatra/LR-195771_License.dat'
    commands=[('vlib',[Q/'vlib.exe','work']),('compile',[Q/'vlog.exe','-sv',*B.glob('*.sv')])]
    for top in ('tb_cnn_boot_sequencer','tb_preview_admission'):
        commands.append((top,[Q/'vsim.exe','-c','work.'+top,'-do','onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0']))
    evidence={str(p):sha(p) for p in [Path(__file__),HERE/'prepare_continuous_run_control.py',*sources,*B.glob('*.sv')]}
    (B/'inputs.json').write_text(json.dumps(evidence,indent=2))
    results=[]
    try:
        for name,cmd in commands:
            start=time.monotonic()
            with (B/(name+'.log')).open('w') as log:
                r=subprocess.run(list(map(str,cmd)),cwd=B,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90)
            body=(B/(name+'.log')).read_text()
            assert r.returncode==0 and not re.search(r'\*\* (Error|Fatal)|^# FAIL',body,re.M),(name,body[-2000:])
            results.append(dict(stage=name,seconds=time.monotonic()-start,markers=re.findall(r'^# (PASS.*)$',body,re.M)))
        assert 'cases=17' in (B/'tb_cnn_boot_sequencer.log').read_text()
        assert 'PASS PREVIEW_REARM' in (B/'tb_cnn_boot_sequencer.log').read_text()
        assert 'PASS PREVIEW_ADMISSION' in (B/'tb_preview_admission.log').read_text()
        for p,h in evidence.items():assert sha(p)==h
        result=dict(status='CONTINUOUS_PREVIEW_CONTROL_SHORT_PASS',stages=results,inputs_sha256=sha(B/'inputs.json'),
            hardware_access=False,full_video_pass=False,physical_pass=False,board_release=False)
        (B/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (B/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise

if __name__=='__main__':main()
