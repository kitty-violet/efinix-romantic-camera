"""Current RAM snapshot delta with unchanged mapped control/observe gates."""
from snapshot_ram_board import BASE,audit_delta
import registered_boundary_flow as flow
if __name__=='__main__':
 flow.BASE=BASE;flow.audit_delta=audit_delta;flow.main()
