"""Run existing 24-block oracle with the current registered-owner RAM handshake."""
import json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from rq_cached_head_board import audit_delta

def main():
    args=sys.argv[1:];root=Path(args[args.index('--root')+1]).resolve()
    board=Path(args[args.index('--frozen-board')+1]).resolve();audit_delta(board)
    assert not root.exists() and root.is_relative_to(ROOT/'tmp')
    source=HERE/'variants/fixed_adapter_boundary_test/tb_compute_column_cluster.sv'
    text=source.read_text()
    text=once(text,'   @(posedge clk);if(tr!==6\'h3f)fail("initialize blocked");@(negedge clk);',
        '   begin\n'
        '    integer wait_ready;wait_ready=0;\n'
        '    do begin @(posedge clk);wait_ready=wait_ready+1;\n'
        '     if(wait_ready>32)fail("initialize handshake timeout");\n'
        '    end while(tr!==6\'h3f);\n'
        '   end\n'
        '   @(negedge clk);')
    tb=root.parent/(root.name+'-tb_compute_column_cluster.sv');assert not tb.exists()
    tb.write_text(text,encoding='utf8')
    runner=HERE/'validate_fixed_production_cluster.py';code=runner.read_text()
    code=once(code,"sources=[tb if f.name==tb.name else f for f in sources]",
              "sources=[tb if f.name=='tb_compute_column_cluster.sv' else f for f in sources]")
    code=once(code,"assert tb.name=='tb_compute_column_cluster.sv' and tb.is_relative_to(HERE/'variants')",
              f"assert tb==Path({str(tb)!r})")
    # The copied test keeps its conventional basename for the Verilator top.
    code=once(code,'dst=r/f.name;assert not dst.exists(),dst',
              f"dst=r/('tb_compute_column_cluster.sv' if f==Path({str(tb)!r}) else f.name);assert not dst.exists(),dst")
    driver=root.parent/(root.name+'-driver.py');assert not driver.exists()
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(runner),source_sha256=sha(runner),
        driver_sha256=sha(driver),tb_source=str(source),tb_source_sha256=sha(source),testbench=str(tb),testbench_sha256=sha(tb),
        adapter_sha256=sha(__file__),scope='Only initialization producer waits for the registered-owner handshake; all computation, data, response and error assertions unchanged.',hardware_access=False),indent=2))
    sys.argv=[str(driver),*args,'--testbench',str(tb)]
    runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
