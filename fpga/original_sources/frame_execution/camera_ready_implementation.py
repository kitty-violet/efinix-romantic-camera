"""Preserve all camera/reset/pin checks for the exact combinational delta."""
from camera_ready_board import audit_delta
import audit_camera_implementation as prior
prior.scope['audit_delta']=audit_delta
def audit(board,physical=None):return prior.audit(board,physical)
