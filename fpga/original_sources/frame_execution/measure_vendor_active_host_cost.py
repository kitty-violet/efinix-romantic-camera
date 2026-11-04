"""Re-time the already-passing real-DDR short gate; never a full-board PASS.

Uses a private copy of the compiled library and unchanged inputs. Measures
calibration separately from the actual four-client/CNN data-traffic window.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from probe_board_dpi_cost import BIN, SIM, run
from run_units import ROOT, sha


def measurement_do():
    # No hierarchical examine: optimized-away HDL objects are not a stable
    # control API. The original HDL wait/$display and final scoreboard prove
    # calibration; their exact timing/values are checked in the resulting log.
    return '''onerror {quit -code 1}
onbreak {resume}
set whole_started [clock milliseconds]
for {set k 1} {$k <= 7} {incr k} {
 set started [clock milliseconds]
 run 1 ms
 puts "HOST_SEGMENT sim_ms=$k wall_ms=[expr {[clock milliseconds]-$started}]"
 flush stdout
}
run 147216 ns
puts "CALIBRATION_WALL_MS [expr {[clock milliseconds]-$whole_started}]"
profile on -p
set active_started [clock milliseconds]
run -all
puts "ACTIVE_WALL_MS [expr {[clock milliseconds]-$active_started}]"
profile summary -file active-summary.txt
profile report -ranked -file active-ranked.txt
puts "MEASURE_VENDOR_ACTIVE_DONE"
quit -code 0
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--control-smoke', action='store_true')
    p.add_argument('--smoke-proof', type=Path)
    a = p.parse_args()
    root, base = a.root.resolve(), a.base.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    env = os.environ.copy()
    env.update(PATH=str(BIN)+';'+env['PATH'], SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    if a.control_smoke:
        root.mkdir()
        (root/'measure.do').write_text(measurement_do())
        (root/'tb.sv').write_text('''`timescale 1ns/1ps
module tb;
 initial begin
  #7147215.207;$display("CONTROL_SMOKE_CAL_DONE");
  #71729.493;$display("CONTROL_SMOKE_FINISHED");$finish;
 end
endmodule
''')
        run([SIM/'vlib.exe','work'],root,'vlib',env,15)
        run([SIM/'vlog.exe','-sv','tb.sv'],root,'compile',env,15)
        wall=run([SIM/'vsim.exe','-batch','-onfinish','stop','work.tb','-do','measure.do'],
                 root,'simulation',env,30)
        log=(root/'simulation.log').read_text()
        for token in ['CONTROL_SMOKE_CAL_DONE','CONTROL_SMOKE_FINISHED',
                      'CALIBRATION_WALL_MS','ACTIVE_WALL_MS','MEASURE_VENDOR_ACTIVE_DONE']:
            assert token in log,token
        assert not re.search(r'\*\* (?:Fatal|Error)|UI-Msg \(Error\)',log)
        result=dict(status='VENDOR_COST_TCL_CONTROL_SMOKE_PASS',wall_seconds=wall,
                    do_sha256=sha(root/'measure.do'),runner_sha256=sha(Path(__file__)),
                    log_sha256=sha(root/'simulation.log'),real_ddr_pass=False,whole_frame=False)
        (root/'result.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result),flush=True)
        return
    assert a.smoke_proof,'Run the exact Tcl flow smoke before another vendor test'
    smoke=json.loads((a.smoke_proof/'result.json').read_text())
    assert smoke['status']=='VENDOR_COST_TCL_CONTROL_SMOKE_PASS'
    assert smoke['runner_sha256']==sha(Path(__file__))
    proof = json.loads((base/'result.json').read_text())
    assert proof['status'] == 'PRODUCTION_DPI_REAL_DDR_FOUR_MASTER_EQUIVALENCE_PASS'
    assert sha(base/'inputs.json') == proof['inputs_sha256']
    assert sha(base/'simulation.log') == proof['log_sha256']
    inputs = json.loads((base/'inputs.json').read_text())
    for item in inputs['files']:
        assert sha(Path(item['path'])) == item['sha256'], item['path']
    root.mkdir()
    shutil.copytree(base/'work', root/'work')
    shutil.copy2(base/'ddr3_controller.bin', root/'ddr3_controller.bin')
    dll = ROOT/'tmp/frame-execution/production-cosim2/production.dll'
    manifest = dict(base=str(base), base_result_sha256=sha(base/'result.json'),
                    inputs_sha256=sha(base/'inputs.json'), dll_sha256=sha(dll),
                    control_smoke_sha256=sha(a.smoke_proof/'result.json'),
                    runner_sha256=sha(Path(__file__)),
                    host_timeout_seconds=1800, full_board=False)
    (root/'inputs.json').write_text(json.dumps(manifest, indent=2))
    (root/'measure.do').write_text(measurement_do())
    assert sha(root/'measure.do')==smoke['do_sha256']
    try:
        wall = run([SIM/'vsim.exe', '-batch', '-onfinish', 'stop', '-warning', '14408,16154',
                    '-voptargs=-warning 14408 -O5', '-sv_lib', dll.with_suffix(''),
                    'work.tb', '-do', 'measure.do'], root, 'simulation', env, 1800)
    except (subprocess.TimeoutExpired, AssertionError) as exc:
        (root/'result.json').write_text(json.dumps(dict(status='VENDOR_HOST_PROBE_FAILED',
             reason=str(exc), full_board=False, log_sha256=sha(root/'simulation.log')), indent=2))
        raise
    log = (root/'simulation.log').read_text(errors='replace')
    assert 'MEASURE_VENDOR_ACTIVE_DONE' in log
    assert not re.search(r'\*\* (?:Error|Fatal)| ERROR: |UI-Msg \(Error\)', log)
    assert 'VENDOR_DDR_CAL_DONE sim_ns=7147215.207' in log
    line = re.search(r'PASS PRODUCTION_DPI_VENDOR_DDR .*', log)[0]
    assert line == proof['line'], (line, proof['line'])
    coverage = re.search(r'DDR_CONCURRENT_COVERAGE .*', log)[0]
    assert coverage == proof['coverage']
    active_ms = int(re.search(r'ACTIVE_WALL_MS (\d+)', log)[1])
    end_ns = float(re.search(r'sim_ns=([\d.]+)', line)[1])
    result = dict(status='VENDOR_HOST_ACTIVE_MEASUREMENT_PASS', wall_seconds=wall,
                  calibration_wall_seconds=int(re.search(r'CALIBRATION_WALL_MS (\d+)', log)[1])/1000,
                  active_simulated_ns=end_ns-7147216,
                  active_wall_seconds=active_ms/1000,
                  host_seconds_per_active_simulated_ms=active_ms/(end_ns-7147216)*1000,
                  line=line, coverage=coverage, whole_frame=False, full_board=False,
                  log_sha256=sha(root/'simulation.log'), inputs_sha256=sha(root/'inputs.json'),
                  segments=[dict(sim_ms=int(k), wall_ms=int(v)) for k,v in
                            re.findall(r'HOST_SEGMENT sim_ms=(\d+) wall_ms=(\d+)',log)])
    (root/'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
