"""Characterize reset release inside an existing line; no hardware access."""
import json, os, shutil, subprocess
from run_units import ROOT, HERE, sha

def main():
    out=ROOT/'tmp/board-test/camera-observer-midline-scope1'
    assert not out.exists();out.mkdir()
    source=HERE/'camera_input_observer.sv';shutil.copy2(source,out/source.name)
    tb=out/'tb.sv'
    tb.write_text('''`timescale 1ns/1ps
module tb;
reg sys=0,pix=0,rst=0,hs=1,de=1;
always #3.333 sys=~sys;
always #10 pix=~pix;
wire [127:0] d;
camera_input_observer dut(.system_clk(sys),.system_reset_n(rst),
 .pixel_clk(pix),.pixel_reset_n(rst),.config_diagnostic(64'b0),
 .config_ok(1'b0),.config_failed(1'b0),.vs(1'b0),.hs(hs),.de(de),
 .raw_errors(11'b0),.boundary_error(1'b0),.diagnostic(d));
initial begin
 repeat(3) @(negedge pix);rst=1;
 repeat(10) @(negedge pix);hs=0;de=0;
 repeat(5) @(negedge pix);
 if(d[79:64]!==16'd10) $fatal(1,"mid-line remainder %d",d[79:64]);
 hs=1;de=1;repeat(480) @(negedge pix);hs=0;de=0;
 repeat(5) @(negedge pix);
 if(d[79:64]!==16'd10) $fatal(1,"held line changed");
 $display("PASS FIRST_OBSERVED_LINE_CAN_BE_PARTIAL_AFTER_MIDLINE_RESET value=%0d",d[79:64]);
 $finish;
end
initial begin #20000;$fatal(1,"timeout");end
endmodule
''',encoding='utf8')
    tool=ROOT/'tools/iverilog/ucrt64/bin';env=os.environ.copy();env['PATH']=str(tool)+os.pathsep+env.get('PATH','')
    for name,cmd in (('compile',[str(tool/'iverilog.exe'),'-g2012','-s','tb','-o','sim.vvp',source.name,'tb.sv']),
                     ('simulation',[str(tool/'vvp.exe'),'sim.vvp'])):
        r=subprocess.run(cmd,cwd=out,env=env,capture_output=True,text=True,timeout=30)
        (out/(name+'.log')).write_text(r.stdout+r.stderr,encoding='utf8')
        assert r.returncode==0,r.stdout+r.stderr
    assert 'PASS FIRST_OBSERVED_LINE_CAN_BE_PARTIAL_AFTER_MIDLINE_RESET' in (out/'simulation.log').read_text()
    result=dict(status='OBSERVER_MIDLINE_RESET_SCOPE_CONFIRMED',first_observed_line_may_be_partial=True,
        full_frame_counter_scope_unchanged=True,hardware_access=False,
        evidence={str(f):sha(f) for f in (source,Path(__file__),tb,out/'compile.log',out/'simulation.log')})
    (out/'result.json').write_text(json.dumps(result,indent=2));print(result['status'])

if __name__=='__main__':
    from pathlib import Path
    main()
