"""Measure autonomous fixture batching; never a board/AXI lookahead proof."""
import argparse
import json
import os
from pathlib import Path
import re
from probe_board_dpi_cost import BIN, VR, SIM, run
from run_units import ROOT, sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--batch',type=int,required=True)
    a=p.parse_args();root=a.root.resolve();ref=a.reference.resolve();obj=ref/'obj'
    assert root.is_relative_to(ROOT/'tmp') and not root.exists() and 1<=a.batch<=4096
    root.mkdir();env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    cpp=(ref/'active.cpp').read_text()+r'''
extern "C" __declspec(dllexport) U active_batch(void*p,U t,unsigned count){
U next=t;for(unsigned i=0;i<count && next!=~0ULL;i++)next=active_step(p,next);return next;}
extern "C" __declspec(dllexport) U active_now(void*p){return static_cast<Fixture*>(p)->c->time();}
'''
    (root/'batch.cpp').write_text(cpp)
    run([BIN/'g++.exe','-O2','-std=c++20','-DVL_TIME_CONTEXT','-I'+obj.as_posix(),'-I'+(VR/'include').as_posix(),
         '-I'+(VR/'include/vltstd').as_posix(),root/'batch.cpp',obj/'Vtb_compute_column_cluster__ALL.a',
         obj/'verilated.o',obj/'verilated_threads.o',obj/'verilated_timing.o','-pthread','-latomic',
         '-shared','-o',root/'batch.dll'],root,'build',env,90)
    # ALL fixture inputs and checks are inside this DLL. Advancing ahead here
    # is safe only for this autonomous fixture, never for an external AXI bus.
    tb=(ref/'tb.sv').read_text().replace('active_step(chandle h,longint unsigned t);',
        'active_batch(chandle h,longint unsigned t,int unsigned count);')
    tb=tb.replace('active_step(h,0)',f'active_batch(h,0,{a.batch})').replace('active_step(h,$time)',f'active_batch(h,$time,{a.batch})')
    tb=tb.replace('chandle h;longint unsigned due;',
        'import "DPI-C" function longint unsigned active_now(chandle h);\nchandle h;longint unsigned due,finished_at;')
    tb=tb.replace('active_final(h);',
        'finished_at=active_now(h);if(finished_at<$time)$fatal(1,"finish time reversed");\n#(finished_at-$time);$display("BATCH_VISIBLE_FINISH_PS %0d",$time);active_final(h);')
    (root/'tb.sv').write_text(tb);(root/'run.do').write_text((ref/'run.do').read_text())
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv',root/'tb.sv'],root,'compile',env,30)
    seconds=run([SIM/'vsim.exe','-c','-voptargs=-O5','-sv_lib',root/'batch','work.tb','-do','run.do'],root,'simulation',env,180)
    log=(root/'simulation.log').read_text(errors='replace')
    golden=json.loads((ref/'result.json').read_text())['rows'][:-1]
    rows=re.findall(r'(?:^|# )(BLOCK case=.*|PASS COMPUTE_CLUSTER.*)$',log,re.M)
    assert rows==golden and 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in log
    assert not re.search(r'\bFAIL\b|%Error|\*\* (Fatal|Error)',log)
    stats=re.search(r'ACTIVE_DPI_STATS calls=(\d+) native_seconds=([\d.]+) sim_ps=(\d+)',log).groups()
    assert int(re.search(r'BATCH_VISIBLE_FINISH_PS (\d+)',log)[1])==int(stats[2])
    result=dict(status='AUTONOMOUS_BATCH_NUMERIC_CYCLE_MATCH_PASS',batch=a.batch,wall_seconds=seconds,
        native_stats=stats,full_board=False,external_axi_partition=False,scope='autonomous fixture only',
        reference_inputs_sha256=sha(ref/'inputs.json'),log_sha256=sha(root/'simulation.log'))
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
