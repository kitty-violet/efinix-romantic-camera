"""External, four-state-driven AXI/load/reset equivalence for the narrow bridge.

The production CNN is real in both engines; the DDR endpoint is a bounded test
stimulus. This does not claim PHY, CNN frame arithmetic, or board completion.
"""
import argparse
import json
import os
from pathlib import Path
import re
import time
from build_production_cosim import ports_of
from probe_board_dpi_cost import BIN, SIM, run
from run_units import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--native-only',action='store_true')
    p.add_argument('--reference-only',action='store_true')
    p.add_argument('--inject-unknown',action='store_true')
    p.add_argument('--active',action='store_true')
    a=p.parse_args();root=a.root.resolve();native=a.native.resolve()
    assert not(a.native_only and a.reference_only)
    assert not a.inject_unknown or a.native_only
    assert not(a.inject_unknown and a.active)
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    manifest=json.loads((native/'inputs.json').read_text())
    for r in manifest['inputs']:assert sha(r['snapshot'])==r['sha256']
    assert sha(native/'production.dll')==manifest['dll_sha256']
    original=(native/'source/cnn_production_frame.sv').read_text();header,ports=ports_of(original)
    ref=root/'reference.sv';ref.write_text(original.replace('module cnn_production_frame(', 'module cnn_production_frame_reference(',1))
    tb='module tb; timeunit 1ns;timeprecision 1ps;\n'
    for d,w,n in ports:tb+=f'{"reg" if d=="input" else "wire"} [{w-1}:0] {n}'+('=0;\n' if d=='input' else ';\n')
    tb+='cnn_production_frame dut(.*);\nalways #3.333 clk=~clk;\ninteger comparisons=0;\n'
    if not (a.native_only or a.reference_only):
        wires=[];connections=[];checks=[]
        qualifies={**{n:'diag_response' for n in ['diag_data','diag_error','diag_return_bank','diag_return_address','diag_return_tag']},
                   **{n:'arvalid' for n in ['araddr','arlen']},**{n:'awvalid' for n in ['awaddr','awlen']},
                   **{n:'wvalid' for n in ['wdata','wstrb','wlast']}}
        for d,w,n in ports:
            if d=='output':
                wires.append(f'wire [{w-1}:0] ref_{n};');connections.append(f'.{n}(ref_{n})')
                condition=f'ref_{qualifies[n]} && ' if n in qualifies else ''
                checks.append(f'if({condition}({n} !== ref_{n}))$fatal(1,"PRODUCTION_DPI_MISMATCH {n} t=%t actual=%h ref=%h",$time,{n},ref_{n});')
            else:connections.append(f'.{n}({n})')
        tb+='\n'.join(wires)+'\ncnn_production_frame_reference reference('+','.join(connections)+');\n'
        tb+='always @(posedge clk or negedge clk)begin #0.002;if(!reset)begin\n'+'\n'.join(checks)+'\ncomparisons++;end end\n'
        if 'wire compute_begin,compute_end;' in (native/'cnn_production_frame_dpi.sv').read_text():
            tb+='always @(posedge clk or negedge clk)begin #0.002;if(!reset)begin if({dut.compute_begin,dut.compute_end} !== {reference.compute_begin,reference.compute_end})$fatal(1,"Production monitor event mismatch");if(reference.compute_begin && dut.debug_program_index !== reference.debug_program_index)$fatal(1,"Production valid monitor index mismatch");end end\n'
    words=ROOT/'tmp/frame-execution/pruned-unit1/frozen/vectors/load_words.hex'
    tb+=r'''
reg [63:0] image[0:2559];integer loaded=0,readbacks=0,held=0,responses=0,scenarios=0;
task reset_all;begin
 reset=1;cancel=0;recover=0;begin_valid=0;load_valid=0;verify_valid=0;diag_valid=0;diag_ready=0;
 start_valid=0;arready=0;rvalid=0;rlast=0;rresp=0;
 repeat(5)@(negedge clk);reset=0;repeat(2)@(negedge clk);
 if(error || busy || frame_valid)$fatal(1,"reset recovery failed");
end endtask
task load_image;begin
 begin_version=1;begin_image_bytes=17792;begin_tag=16'h4321;job_image_tag=16'h4321;
 begin_valid=1;do @(posedge clk);while(!begin_ready);@(negedge clk);begin_valid=0;
 for(integer k=0;k<2560;k++)begin
  load_index=k;load_data=image[k];load_last=k==2559;load_valid=1;
  do @(posedge clk);while(!load_ready);@(negedge clk);load_valid=0;
  if(k%37==0)repeat(2)@(negedge clk);
 end
 if(!load_complete || coeff_ready || store_error)$fatal(1,"load completion barrier");
 loaded++;
 for(integer k=0;k<32;k++)begin
  diag_bank=k%5;diag_address=k/5;diag_valid=1;
  do @(posedge clk);while(!diag_accept);@(negedge clk);diag_valid=0;
  repeat(k%3)@(negedge clk);diag_ready=1;
  do @(posedge clk);while(!diag_response);
  if(diag_data!==image[k] || diag_error)$fatal(1,"load readback mismatch");
  readbacks++;@(negedge clk);diag_ready=0;
 end
 verify_tag=16'h4321;verify_ok=1;verify_valid=1;
 do @(posedge clk);while(!verify_ready);@(negedge clk);verify_valid=0;
 if(!coeff_ready)$fatal(1,"verify did not release");
end endtask
task start_job;begin
 start_valid=1;do @(posedge clk);while(!start_ready);@(negedge clk);start_valid=0;
 while(!arvalid)@(negedge clk);
end endtask
task serve(input integer kind);integer length;reg[31:0] address;begin
 length=arlen;address=araddr;
 repeat(11)begin
  @(negedge clk);if(!arvalid || araddr!==address || arlen!==8'(length))$fatal(1,"AR changed under backpressure");held++;
 end
 if(kind==0)begin cancel=1;repeat(3)@(negedge clk);if(!arvalid)$fatal(1,"cancel withdrew exposed AR");end
 arready=1;@(negedge clk);arready=0;
 for(integer k=0;k<=length;k++)begin
  repeat(1+k%3)@(negedge clk);
  rdata=128'(k);rresp=(kind==1 && k==1)?2'b10:0;rlast=k==length;rvalid=1;
  do @(posedge clk);while(!rready);responses++;@(negedge clk);rvalid=0;
  if(kind==2 && k==0)begin reset_all();scenarios++;return;end
 end
 repeat(16)@(negedge clk);
 if(kind==1 && (!error || !needs_common_reset))$fatal(1,"R error not reported");
 if(frame_valid)$fatal(1,"bad frame published");scenarios++;
end endtask
initial begin
 $readmemh("WORDS",image);
 for(integer k=0;k<3;k++)begin reset_all();load_image();start_job();serve(k);end
 reset_all();
 if(loaded!=3 || readbacks!=96 || held!=33 || scenarios!=3 || responses==0)$fatal(1,"coverage incomplete");
 $display("PASS PRODUCTION_DPI_BOUNDARY loaded=%0d readbacks=%0d held=%0d responses=%0d scenarios=%0d comparisons=%0d sim_ns=%0.3f",loaded,readbacks,held,responses,scenarios,comparisons,$realtime);
 $finish;
end
initial begin #200000;$fatal(1,"bounded boundary watchdog");end
endmodule
'''.replace('WORDS',words.as_posix())
    if a.active:
        from production_cosim_active_fixture import active_fixture
        tb=active_fixture(tb,root)
    if a.inject_unknown:
        tb=tb.replace('endmodule','initial begin #1000;rvalid=1;rdata=128\'bx;end\nendmodule')
    (root/'tb.sv').write_text(tb)
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    files=[]
    if a.reference_only:files=[Path(r['snapshot']) for r in manifest['inputs']]+[root/'tb.sv']
    else:
        if not a.native_only:files=[Path(r['snapshot']) for r in manifest['inputs'] if Path(r['snapshot']).name!='cnn_production_frame.sv']+[ref]
        files += [native/'cnn_production_frame_dpi.sv',root/'tb.sv']
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv','-timescale','1ns/1ps','+incdir+'+(native/'source').as_posix(),*files],root,'compile',env,60)
    start=time.perf_counter()
    try:
        seconds=run([SIM/'vsim.exe','-c','-voptargs=-O5',*([] if a.reference_only else ['-sv_lib',native/'production']),'work.tb','-do','run.do'],root,'simulation',env,240)
    except AssertionError:
        if not a.inject_unknown:raise
        seconds=time.perf_counter()-start
    log=(root/'simulation.log').read_text(errors='replace')
    if a.inject_unknown:
        assert '** Fatal: Production DPI unknown R payload' in log and 'PASS PRODUCTION_DPI_BOUNDARY' not in log
        result=dict(status='PRODUCTION_DPI_UNKNOWN_ACTIVE_R_REJECTION_PASS',wall_seconds=seconds,
            full_board=False,native_inputs_sha256=sha(native/'inputs.json'),log_sha256=sha(root/'simulation.log'),
            testbench_sha256=sha(root/'tb.sv'))
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    if a.active:
        line=re.search(r'PASS PRODUCTION_DPI_ACTIVE_PREFIX .*',log)
        assert line and not re.search(r'\*\* (Fatal|Error)',log)
        if not (a.native_only or a.reference_only):assert int(re.search(r'comparisons=(\d+)',line[0])[1])>10000
        status='PRODUCTION_ACTIVE_NATIVE_SMOKE_PASS' if a.native_only else 'PRODUCTION_ACTIVE_RTL_SMOKE_PASS' if a.reference_only else 'PRODUCTION_ACTIVE_DPI_EXTERNAL_EQUIVALENCE_PASS'
        result=dict(status=status,wall_seconds=seconds,line=line[0],external_axi=True,active_compute=True,
                    full_frame=False,phy=False,native_inputs_sha256=sha(native/'inputs.json'),
                    log_sha256=sha(root/'simulation.log'),testbench_sha256=sha(root/'tb.sv'),payload_sha256=sha(root/'active_payload.json'))
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return
    line=re.search(r'PASS PRODUCTION_DPI_BOUNDARY .*',log)
    assert line and not re.search(r'\*\* (Fatal|Error)',log)
    if not (a.native_only or a.reference_only):assert int(re.search(r'comparisons=(\d+)',line[0])[1])>10000
    status='PRODUCTION_DPI_NATIVE_BOUNDARY_SMOKE_PASS' if a.native_only else 'PRODUCTION_RTL_BOUNDARY_SMOKE_PASS' if a.reference_only else 'PRODUCTION_DPI_EXTERNAL_BOUNDARY_EQUIVALENCE_PASS'
    result=dict(status=status,
        wall_seconds=seconds,line=line[0],native_inputs_sha256=sha(native/'inputs.json'),
        external_axi=True,compute_numeric=False,phy=False,full_board=False,log_sha256=sha(root/'simulation.log'),
        testbench_sha256=sha(root/'tb.sv'))
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
