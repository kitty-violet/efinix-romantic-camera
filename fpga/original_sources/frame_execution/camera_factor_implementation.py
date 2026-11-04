"""Existing implementation audit bound to the current transform."""
from camera_factor_board import audit_delta
import audit_camera_implementation as prior
prior.scope['audit_delta']=audit_delta
def audit(board,physical=None):return prior.audit(board,physical)
