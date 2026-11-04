"""Same held-data CDC requirements; native split cell names only."""
from run_units import sha
from audit_native_split_map import extra_map
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
    result['native_split_auditor_sha256']=sha(__file__)
    return result
