"""Reuse all fifteen functional short cases with the new audited input clone."""
import json,sys
from pathlib import Path
from run_units import sha
from split_logic_outputs_board import audit_delta
import validate_startup_cause as suite

if __name__=='__main__':
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
 suite.audit_delta=audit_delta
 # The same deliberate stuck-low fault now targets the equivalent LUT O pin.
 before='dut.local_cells.pair_under_test.P'
 assert suite.CELL_TB.count(before)==1
 suite.CELL_TB=suite.CELL_TB.replace(before,'dut.local_cells.independent_p.O')
 suite.main()
 result=json.loads((root/'result.json').read_text())
 assert result['status']=='STARTUP_CAUSE_TRACE_CELLS_SERIAL_ADAPTER_PASS' and len(result['results'])==15
 (root/'split_wrapper_proof.json').write_text(json.dumps(dict(
  status='SPLIT_OUTPUTS_SAME_FIFTEEN_SHORT_CASES_PASS',
  original_result_sha256=sha(root/'result.json'),wrapper_sha256=sha(__file__),
  implementation_sha256=sha(Path(__file__).with_name('split_logic_outputs_board.py')),
  fault_target_change='COMB4 P -> equivalent independent_p LUT4 O; all assertions retained',
  actual_map_zero_comb4_proven=False,board_release=False,hardware_access=False),indent=2))
