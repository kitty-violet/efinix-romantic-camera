"""Reuse the 24-block tensor oracle against the frozen registered-edge board."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once

BOARD=ROOT/'tmp/frame-execution/connected-board-split-registered2'
RUN=ROOT/'tmp/frame-execution/split-registered-cluster1'

def audit():
    bind=json.loads((BOARD/'boundary_binding.json').read_text())
    assert bind['status']=='FULL_BOARD_REGISTERED_BOUNDARIES_FROZEN_NOT_RELEASED'
    assert sha(BOARD/'inputs.json')==bind['inputs_sha256']=='aed4adc0f7777c26256ef625ea1f2ee3957111179c3fdb5fd772f811214804c8'
    for row in json.loads((BOARD/'inputs.json').read_text())['inputs']:
        assert sha(row['snapshot'])==row['sha256']
    return bind

def main():
    audit();assert not RUN.exists()
    prior=ROOT/'tmp/frame-execution/cached-head-cluster3'
    proof=json.loads((prior/'result.json').read_text())
    assert proof['status']=='FIXED_CLUSTER_REAL_RESIDENT_PASS'
    assert sha(prior/'inputs.json')==proof['inputs_sha256']
    assert sha(prior/'simulation.log')==proof['log_sha256']
    rec=json.loads((prior/'inputs.json').read_text())
    row=next(r for r in rec['inputs'] if Path(r['snapshot']).name=='tb_compute_column_cluster.sv')
    source=Path(row['snapshot']);assert sha(source)==row['sha256']
    text=once(source.read_text(),'row_pair_arena6 arena(.clk',"row_pair_arena6 arena(.aux_write_collision(6'd0),.clk")
    tb=RUN.parent/(RUN.name+'-tb_compute_column_cluster.sv');assert not tb.exists()
    tb.write_text(text,encoding='utf8')
    runner=HERE/'validate_fixed_production_cluster.py';code=runner.read_text()
    code=once(code,"sources=[tb if f.name==tb.name else f for f in sources]",
              "sources=[tb if f.name=='tb_compute_column_cluster.sv' else f for f in sources]")
    code=once(code,"assert tb.name=='tb_compute_column_cluster.sv' and tb.is_relative_to(HERE/'variants')",
              f"assert tb==Path({str(tb)!r})")
    code=once(code,'dst=r/f.name;assert not dst.exists(),dst',
              f"dst=r/('tb_compute_column_cluster.sv' if f==Path({str(tb)!r}) else f.name);assert not dst.exists(),dst")
    driver=RUN.parent/(RUN.name+'-driver.py');assert not driver.exists()
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(
        source=str(runner),source_sha256=sha(runner),driver_sha256=sha(driver),
        prior_verified_testbench=str(source),prior_testbench_sha256=sha(source),
        testbench=str(tb),testbench_sha256=sha(tb),adapter_sha256=sha(__file__),
        test_change='Explicitly tie unused collision input low; default internal collision remains active.',
        inputs_sha256=sha(BOARD/'inputs.json'),hardware_access=False),indent=2))
    sys.argv=[str(driver),'--root',str(RUN),'--engine','native','--simulator','questa',
              '--frozen-board',str(BOARD),'--testbench',str(tb)]
    runpy.run_path(str(driver),run_name='__main__');audit()

if __name__=='__main__':main()
