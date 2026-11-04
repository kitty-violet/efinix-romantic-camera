"""Unchanged held-data CDC policy, inspecting the new independent LUT cells."""
from run_units import sha
from audit_split_logic_outputs_map import extra_map
import startup_cause_clock_cdc as original

def map_structure(board):
 original.extra_map=extra_map
 result=original.map_structure(board)
 result['auditor_sha256']=sha(__file__)
 result['original_auditor_sha256']=sha(original.__file__)
 return result

def inventory(board):
 original.extra_map=extra_map
 result=original.inventory(board)
 result['split_output_auditor_sha256']=sha(__file__)
 return result
