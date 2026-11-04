"""Actual A bounds plus independent arbitrary-INT32 RQ fault/queue short test."""
import argparse,json,os,re,shutil,sys,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from map_video_service_safe import safe_run as run

def once(s,old,new):
    assert s.count(old)==1,old
    return s.replace(old,new)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root.resolve()
    assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir();start=time.monotonic()
    src=ROOT/'tmp/frame-execution/connected-board-video-service-probe20/assembly/fixed_rq32_mixed.sv'
    rtl=src.read_text()
    rtl=once(rtl,'selected_split<=split_lane;',"selected_split<=32'd0;")
    # This is the HDL counterpart of the native scale-error guard. The
    # signed7 class is checked only on MAC, never on residual feedback.
    rtl=once(rtl,'assign scale_invalid_now[c]=multiplier[c*32+18+:14]!={14{multiplier[c*32+17]}};',
      'assign scale_invalid_now[c]=(multiplier[c*32+18+:14]!={14{multiplier[c*32+17]}}) || (scale_fits7_now[c] && scale_nonzero_now[c]);')
    (out/'fixed_rq32_mixed.sv').write_text(rtl)
    tbfile=HERE/'variants/fixed_dsp_round_feedback_test/tb_fixed_split_rq.sv';tb=tbfile.read_text()
    tb=once(tb,'fixed_rq32_mixed dut(.*);','wire feedback_post_selected;\n fixed_rq32_mixed #(.REGISTER_SCALE_CLASS(1),.SHARED_FEEDBACK(0)) dut(.*);')
    tb=once(tb,'invalid_scale=sc< -131072 || sc>131071;',
            'invalid_scale=sc< -131072 || sc>131071 || (k==0 && sc!=0 && sc>=-64 && sc<=63);')
    tb=once(tb,'for(integer epoch=0;epoch<6;epoch=epoch+1)','for(integer epoch=0;epoch<12;epoch=epoch+1)')
    tb=once(tb,'c==0 ? 0:c==1 ? 1:c==2 ? -1:c==3 ? 63:c==4 ? -64:c==5 ? 131071:c==6 ? -131072:131072;',
            'epoch<6 ? (c==0 ? 0:c==1 ? 1:c==2 ? -1:c==3 ? 63:c==4 ? -64:c==5 ? 131071:c==6 ? -131072:131072):\n     (c==0 ? 0:c==1 ? 64:c==2 ? -65:c==3 ? 421:c==4 ? 2048:c==5 ? 65536:c==6 ? -131072:131071);')
    tb=once(tb,'for(integer k=0;k<3;k=k+1)begin sent[k]=0;payload(k,0);end','repeat(2)@(negedge clk);\n   for(integer k=0;k<3;k=k+1)begin sent[k]=0;payload(k,0);end')
    tb=once(tb,'payload(0,450);in_valid=1;out_ready=0;', 'repeat(2)@(negedge clk);payload(0,450);in_valid=1;out_ready=0;')
    tb=once(tb,'if(delay<=3 && !dut.preparing)$fatal(1,"FIRST_PASS_CANCEL_NOT_ACTIVE");','if(dut.preparing)$fatal(1,"UNEXPECTED_CORRECTION_PASS");')
    tb=once(tb,'if(checks!=24576 || slow<100 || full<100 || held_cycles<100 || fault_flags==0)',
               'if(checks!=47616 || slow!=0 || full<100 || held_cycles<100 || fault_flags==0)')
    tb=tb.replace('#250000;','#500000;').replace('PASS FIXED_SPLIT_RQ','PASS GUARDED_FAST_RQ')
    (out/'tb.sv').write_text(tb)
    for n in ('efx_srl8.v','efx_dsp48.v'):shutil.copy2(EF/'sim_models/verilog'/n,out/n)
    ms=Path('E:/quatra/questa_fse/win64');os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    inputs={str(f):sha(f) for f in (src,tbfile,Path(__file__),out/'fixed_rq32_mixed.sv',out/'tb.sv',out/'efx_srl8.v',out/'efx_dsp48.v')}
    (out/'inputs.json').write_text(json.dumps(inputs,indent=2))
    run([ms/'vlib.exe','work'],out,'vlib.log',30)
    log,_=run([ms/'vlog.exe','-sv','efx_srl8.v','efx_dsp48.v','fixed_rq32_mixed.sv','tb.sv'],out,'compile.log',60)
    assert 'Errors: 0, Warnings: 0' in log
    log,seconds=run([ms/'vsim.exe','-c','work.tb_fixed_split_rq','-do','onerror {quit -code 1}; run -all; quit -code 0'],out,'simulation.log',120)
    m=re.findall(r'^# (PASS GUARDED_FAST_RQ.*)$',log,re.M);assert len(m)==1 and 'Errors: 0, Warnings: 0' in log
    for f,h in inputs.items():assert sha(f)==h
    result=dict(status='GUARDED_FAST_RQ_INT32_NUMERIC_QUEUE_CANCEL_PASS',line=m[0],seconds=time.monotonic()-start,simulation_seconds=seconds,
                inputs=inputs,hardware_access=False,board_release=False)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
