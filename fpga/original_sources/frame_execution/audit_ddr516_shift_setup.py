"""Bounded measurement of real vendor IP SHIFT setup at the V4 PLL boundary.

Use the previously passing short DDR fixture with the frozen deployed wrapper,
bridge and V4 clock plan. Do not force calibration or change protected code.
This audit stops after two shift requests; it is not a calibration PASS.
"""
import argparse, json, os, re, shutil, subprocess, time
from pathlib import Path
from run_units import ROOT, HERE, sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--shift-guard',action='store_true')
    p.add_argument('--complete-calibration',action='store_true');a=p.parse_args()
    assert not a.complete_calibration or a.shift_guard
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    fixture=ROOT/'tmp/frame-execution/ddr516-v4-wrapper1'
    board=ROOT/'tmp/frame-execution/connected-board-first-fault1'
    meta=json.loads((fixture/'inputs.json').read_text());records=[]
    for item in meta['inputs']:
        src=Path(item['snapshot']);assert sha(src)==item['sha256'],src
        dst=root/src.name
        if dst.exists():continue
        shutil.copy2(src,dst);records.append(dict(source=str(src),sha256=sha(src)))
    for name in ('ddr516_v4_controller.sv','ddr516_rx_retime.sv','ddr516_axi_address_bridge.sv'):
        shutil.copy2(board/name,root/name);records.append(dict(source=str(board/name),sha256=sha(board/name)))
    for name in ('ddr3_controller.vh','ddr3_controller.bin'):
        assert sha(board/name)==sha(root/name)
    if a.shift_guard:
        guard=HERE/'ddr516_pll_shift_guard.sv';shutil.copy2(guard,root/guard.name)
        records.append(dict(source=str(guard),sha256=sha(guard)))
        path=root/'ddr516_v4_controller.sv';wrapper=path.read_text()
        old='.shift(pll_shift),.shift_sel(pll_shift_sel),.shift_ena(pll_shift_ena),'
        assert wrapper.count(old)==1
        wrapper=wrapper.replace(old,'.shift(request_shift),.shift_sel(request_select),.shift_ena(request_enable),')
        wrapper=wrapper.replace(' wire calibration_done,calibration_pass;', '''
 wire [2:0] request_shift;wire [4:0] request_select;wire request_enable;
 ddr516_pll_shift_guard shift_guard(.clk(axi_clk),.reset_n(rstn),
  .request_shift(request_shift),.request_select(request_select),.request_enable(request_enable),
  .pll_shift(pll_shift),.pll_select(pll_shift_sel),.pll_enable(pll_shift_ena));
 wire calibration_done,calibration_pass;''')
        path.write_text(wrapper)
    tb=(root/'tb.sv').read_text()
    old=tb[tb.index('reg ref50=0;'):tb.index('wire reset;wire memory_reset;')]
    new=old.replace('always #100','always #200').replace('.M(1)','.M(2)')
    new=new.replace('.CLKOUT1_DIV(12)','.CLKOUT1_DIV(6)').replace('.CLKOUT3_DIV(6)','.CLKOUT3_DIV(12)')
    new=new.replace('.CLKOUT3_PHASE_STEP(3)','.CLKOUT0_PHASE_STEP(3)').replace('.REFCLK_FREQ(50.0)','.REFCLK_FREQ(25.0)')
    new=new.replace('.CLKOUT0(tdqss_clk),.CLKOUT1(core_clk)', '.CLKOUT0(twd_clk),.CLKOUT1(tdqss_clk)').replace('.CLKOUT3(twd_clk)', '.CLKOUT3(core_clk)')
    assert new!=old;tb=tb.replace(old,new)
    tb=tb.rsplit('endmodule',1)[0]+r'''
realtime last_code_change=0,ena_start=0;
integer measured_requests=0,setup_violations=0;
always @(shift) last_code_change=$realtime;
always @(posedge shift_ena) begin
  // Wait one precision tick to include same-time nonblocking code updates.
  ena_start=$realtime;
  #0.001;
  measured_requests=measured_requests+1;
  if((ena_start-last_code_change)/10.0<2.5) setup_violations=setup_violations+1;
  $display("SHIFT_SETUP_MEASURE request=%0d code=%0d select=%b setup_ns=%0.4f",measured_requests,shift,shift_sel,(ena_start-last_code_change)/10.0);
end
always @(negedge shift_ena) if(measured_requests>0) begin
  $display("SHIFT_HIGH_MEASURE request=%0d high_ns=%0.4f",measured_requests,($realtime-ena_start)/10.0);
  if(measured_requests==2) begin
    $display("SHIFT_AUDIT_COMPLETE requests=%0d setup_violations=%0d",measured_requests,setup_violations);$finish;
  end
end
initial begin #9500000;$fatal(1,"SHIFT audit 950us limit");end
endmodule
'''
    if a.complete_calibration:
        tb=tb.replace('if(measured_requests==2) begin','if(1\'b0) begin')
        tb=tb.replace('initial begin #9500000;$fatal(1,"SHIFT audit 950us limit");end','')
        # Original full calibration and four read/write bursts remain enabled.
        assert tb.count('$display ("DDR3 Controller PASS");')==1
        tb=tb.replace('$display ("DDR3 Controller PASS");', '''begin
if(setup_violations!=0)$fatal(1,"PLL setup violations=%0d",setup_violations);
$display("SHIFT_AUDIT_COMPLETE requests=%0d setup_violations=%0d",measured_requests,setup_violations);
$display ("DDR3 Controller PASS");end''')
    # Declare counters before the original fixture's early completion block.
    declarations='realtime last_code_change=0,ena_start=0;\ninteger measured_requests=0,setup_violations=0;'
    assert tb.count(declarations)==1
    tb=tb.replace(declarations,'').replace('module efx_ddr3_soft_controller_tb ();','module efx_ddr3_soft_controller_tb ();\n'+declarations)
    (root/'tb.sv').write_text(tb)
    files=['ddr3.v','efx_iddr.v','efx_oddr.v','efx_srl8.v','memory_checker_axi.v','ddr516_reference.sv','ddr516_axi_address_bridge.sv','ddr516_v4_controller.sv','example_top.v','ddr516_rx_retime.sv','EFX_GPIO_V3.v','EFX_PLL_V3_FAST_PROTOCOL.v','tb.sv']
    if a.shift_guard:files.insert(0,'ddr516_pll_shift_guard.sv')
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    # Full calibration actively exercises memory; the measured 5.9 ms / 600 s
    # prefix predicts about 750 s for 7.15 ms plus read/write. Keep a hard cap.
    budget=1000 if a.complete_calibration else 210
    shutil.copy2(__file__,root/'runner.py')
    inputs=dict(sources=records,files={f.name:sha(f) for f in root.iterdir() if f.is_file()},runner_sha256=sha(__file__),scope='V4 actual vendor calibration and read/write' if a.complete_calibration else 'real IP first two shift requests; no board access; no full calibration claim',shift_guard=a.shift_guard,wall_budget_s=budget)
    (root/'inputs.json').write_text(json.dumps(inputs,indent=2))
    sim=Path('E:/quatra/questa_fse/win64');env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat';start=time.monotonic()
    stages=[]
    try:
        for name,args in [('vlib',['vlib.exe','work']),('compile',['vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16',*files]),('simulation',['vsim.exe','-c','work.efx_ddr3_soft_controller_tb','-do','run.do'])]:
            cmd=[str(sim/args[0]),*args[1:]];t=time.monotonic()
            with (root/(name+'.log')).open('wb') as f:
                done=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=max(1,budget-(t-start)))
            body=(root/(name+'.log')).read_text(errors='replace')
            stages.append(dict(stage=name,seconds=time.monotonic()-t,returncode=done.returncode))
            if done.returncode or re.search(r'\*\* (Error|Fatal)',body):raise RuntimeError(body[-2500:])
        found=re.search(r'SHIFT_AUDIT_COMPLETE requests=(\d+) setup_violations=(\d+)',body);assert found
        if a.complete_calibration:
            assert 'DDR3 Controller PASS' in body and 'DDR516_CAL_DONE' in body
            assert re.search(r'DDR516_CAL_DONE .*pass=1',body)
            from audit_ddr_shift_completed_log import errors
            assert not errors(body),errors(body)
        result=dict(status='SHIFT_SETUP_VIOLATION_REPRODUCED' if int(found[2]) else 'SHIFT_SETUP_CHECK_PASS',requests=int(found[1]),violations=int(found[2]),measurements=re.findall(r'^# (SHIFT_(?:SETUP|HIGH)_MEASURE.*)$',body,re.M),seconds=time.monotonic()-start,stages=stages,inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),board_release=False)
        if a.complete_calibration:
            result.update(status='GUARDED_SHIFT_VENDOR_CALIBRATION_RW_PASS',markers=re.findall(r'^# (DDR516_(?:CAL_DONE|MEASURE).*|DDR3 Controller PASS)$',body,re.M))
        for item in records:assert sha(item['source'])==item['sha256']
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),stages=stages,seconds=time.monotonic()-start),indent=2));raise

if __name__=='__main__':main()
