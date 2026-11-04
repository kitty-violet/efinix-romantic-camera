"""Existing control/CDC truth gates, adjusted only for split instance names."""
import inspect,json,sys
from pathlib import Path
from run_units import sha
from native_split_board import BASE,audit_delta
import audit_split_logic_outputs_map as original

source=inspect.getsource(original.extra_map)
source=source.replace("suffix.startswith('independent_')","suffix in ('independent_o','independent_p')")
source=source.replace('independent_o','pair_under_test_splitO').replace('independent_p','pair_under_test_splitP')
scope=dict(original.__dict__,BASE=BASE,__file__=__file__)
exec(compile(source,__file__+':equivalent-cell-names','exec'),scope)
extra_map=scope['extra_map']


def main():
    board=Path(sys.argv[sys.argv.index('--board')+1]).resolve()
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    import audit_fault_probe_map as prior
    prior.audit_delta=audit_delta;prior.main()
    result=extra_map(board);result['original_checker_sha256']=sha(original.__file__)
    (root/'new_diagnostic_cells.json').write_text(json.dumps(result,indent=2))
    from native_split_cdc import map_structure
    (root/'held_data_structure.json').write_text(json.dumps(map_structure(board),indent=2))
    print(result['status'])


if __name__=='__main__':main()
