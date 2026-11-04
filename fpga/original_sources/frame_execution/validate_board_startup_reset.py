"""Short four-state reset and real A APB/store regression; never access USB."""
import argparse
import json
import os
from pathlib import Path
import shutil
import traceback
from run_units import ROOT, HERE, EF, sha, run

MS = Path('E:/quatra/questa_fse/win64')
BASE = ROOT/'tmp/frame-execution/connected-board-measured-pinmap1'
VARIANT = HERE/'variants/board_deterministic_reset'


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir();os.environ['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    paths=[VARIANT/'board_startup_reset.sv',VARIANT/'tb_board_startup_reset.sv',
           EF/'sim_models/verilog/efx_ram10.v',BASE/'assembly/resident_weight_bank64.sv',
           BASE/'assembly/coefficient_bank_store_fast.sv',BASE/'assembly/cnn_apb_control.sv']
    records=[]
    for src in paths:
        dst=root/src.name;shutil.copy2(src,dst)
        records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
    original=HERE/'variants/board_bringup_status/tb_cnn_apb_control.sv'
    tb=original.read_text()
    assert tb.count('reg reset=1;')==1
    tb=tb.replace('reg reset=1;', 'reg key_reset_n=0; wire system_reset_n; wire reset=!system_reset_n;\n'
        ' board_startup_reset reset_manager(.clk(clk),.key_reset_n(key_reset_n),.plls_locked(1\'b1),.system_reset_n(system_reset_n));')
    # Preserve every existing numerical/readback/error test; only drive the
    # new external reset and wait for its actual release instead of a TB reset.
    assert tb.count('reset=0;')==1 and tb.count('reset=1;')==0
    tb=tb.replace('reset=0;', 'key_reset_n=1;wait(system_reset_n);@(negedge clk);')
    # After the original full test, assert reset again and verify the actual
    # store metadata is cleared, not merely the reset manager's output.
    assert tb.count('$finish;')==1
    tb=tb.replace('$finish;', '''
  @(negedge clk);key_reset_n=0;repeat(4)@(negedge clk);
  if(store_error || error_code!=0 || loaded_words!=0 || coeff_ready || loading || load_complete)
   $fatal(1,"store metadata not cleared by manual reset");
  key_reset_n=1;wait(system_reset_n);repeat(8)@(negedge clk);
  access(0,16'h0004,0,0,value);if(value!=0)$fatal(1,"idle store status after reset");
  access(0,16'h0064,0,0,value);if(value!=0)$fatal(1,"idle error code after reset");
  $display("PASS RESET_REAL_A_STORE_RESTART");$finish;''')
    (root/'tb_cnn_apb_control.sv').write_text(tb,encoding='utf-8')
    vectors=ROOT/'model-training-workspace/accepted/B1-A-20261008'
    meta=dict(inputs=records,original_tb=str(original),original_tb_sha256=sha(original),
              adapted_tb_sha256=sha(root/'tb_cnn_apb_control.sv'),runner_sha256=sha(__file__),
              vectors_sha256=sha(vectors/'load_words.hex'),hardware_access=False)
    (root/'inputs.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    try:
        run([MS/'vlib.exe','work'],root,'vlib.log',timeout=30)
        body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',
                    *[r['snapshot'] for r in records],root/'tb_cnn_apb_control.sv'],root,'compile.log',timeout=40)
        assert 'Errors: 0, Warnings: 0' in body
        rows=[]
        for top,marker in [('tb_board_startup_reset','PASS BOARD_STARTUP_RESET'),
                           ('tb_cnn_apb_control','PASS RESET_REAL_A_STORE_RESTART')]:
            body,seconds=run([MS/'vsim.exe','-c','work.'+top,'+VECTORS='+vectors.as_posix(),
                '-do','onerror {quit -code 1}; run -all; quit -code 0'],root,top+'.log',timeout=90)
            assert marker in body and 'Errors: 0, Warnings: 0' in body
            rows.append(dict(top=top,seconds=seconds,log_sha256=sha(root/(top+'.log'))))
        for r in records: assert sha(r['source'])==sha(r['snapshot'])==r['sha256']
        result=dict(status='RESET_AND_ACTUAL_A_STORE_FOURSTATE_PASS',rows=rows,
                    inputs_sha256=sha(root/'inputs.json'),board_release=False,
                    ddr_calibration_proven=False,physical_timing_pass=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()),indent=2))
        raise


if __name__=='__main__':main()
