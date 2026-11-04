"""Bounded empty-kernel DPI attribution. No FPGA behavior is replaced."""
import argparse
import json
import os
from pathlib import Path
import re
from probe_board_dpi_cost import BIN, SIM, run
from run_units import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir();env=os.environ.copy()
    env.update(PATH=str(BIN)+';'+env['PATH'],SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    (root/'probe.cpp').write_text(r'''
#include <chrono>
#include <cstdio>
using C=std::chrono::steady_clock;
static C::time_point started;
extern "C" __declspec(dllexport) void stamp(int stage){
auto now=C::now();if(stage)std::printf("STAGE_HOST_SECONDS stage=%d seconds=%.9f\n",stage,std::chrono::duration<double>(now-started).count());
started=now;std::fflush(stdout);}
extern "C" __declspec(dllexport) unsigned step(unsigned x){static unsigned state=0;state=state*1664525U+x;return state;}
''')
    (root/'tb.sv').write_text('''module tb;
timeunit 1ns;timeprecision 1ps;
import "DPI-C" function void stamp(int stage);
import "DPI-C" function int unsigned step(int unsigned x);
int unsigned value=0,expect_value=0;
initial begin
 stamp(0);
 for(int i=0;i<75000;i++)begin #5;value=value*1664525+i;end
 stamp(1);
 value=0;
 for(int i=0;i<75000;i++)begin #5;value=step(i);expect_value=expect_value*1664525+i;if(value!==expect_value)$fatal(1,"DPI numeric mismatch");end
 stamp(2);
 for(int i=0;i<75000;i++)begin value=step(i);expect_value=expect_value*1664525+i;if(value!==expect_value)$fatal(1,"DPI numeric mismatch");end
 stamp(3);$display("PASS EMPTY_DPI_SCHEDULER");$finish;
end
endmodule
''')
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    run([BIN/'g++.exe','-O2','-shared',root/'probe.cpp','-o',root/'probe.dll'],root,'build',env,60)
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv',root/'tb.sv'],root,'compile',env,30)
    seconds=run([SIM/'vsim.exe','-c','-voptargs=-O5','-sv_lib',root/'probe','work.tb','-do','run.do'],root,'simulation',env,180)
    log=(root/'simulation.log').read_text(errors='replace')
    assert 'PASS EMPTY_DPI_SCHEDULER' in log and not re.search(r'\*\* (Fatal|Error)',log)
    rows=re.findall(r'STAGE_HOST_SECONDS stage=(\d+) seconds=([\d.]+)',log)
    assert len(rows)==3
    result=dict(status='EMPTY_SCHEDULER_DIAGNOSTIC_ONLY',wall_seconds=seconds,stages=dict(rows),
                stage_meaning={'1':'SV timed arithmetic','2':'DPI timed arithmetic','3':'DPI no delays'},calls_per_stage=75000,
                full_board=False,log_sha256=sha(root/'simulation.log'))
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
