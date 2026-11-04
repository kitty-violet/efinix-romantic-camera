"""Run the existing mapped reset/control/observer checks with a strict new delta."""
from prepare_ddr_shift_guard_board import audit_delta
import validate_first_fault_mapped as checks

if __name__=='__main__':
    # The checker itself is unchanged; only the admissible source delta differs.
    checks.audit_delta=lambda board:audit_delta(board,False)
    checks.main()
