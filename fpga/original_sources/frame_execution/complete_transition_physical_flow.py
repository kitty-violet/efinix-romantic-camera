"""Existing placement-stop/route-review policy for the local D-cone candidate."""
import sys
from complete_control_transition import audit_delta

if __name__=='__main__':
    import cached_head_flow as flow
    flow.audit_delta=audit_delta
    # The imported original parent-audit function remains intact. Only the
    # implementation flow's explicit candidate binding is replaced.
    flow.main()
