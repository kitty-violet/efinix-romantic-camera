"""Reuse the unchanged public PLL guard proof on the full OFP2 candidate."""
from control_command_repair import audit_delta
import audit_pll_shift_routed as audit

if __name__=='__main__':
    audit.audit_delta=audit_delta
    audit.main()
