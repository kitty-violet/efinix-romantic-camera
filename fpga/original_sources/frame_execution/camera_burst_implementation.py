"""Apply the existing full pin/clock/reset implementation gates to this map."""
from camera_capture_burst_board import audit_delta
import audit_camera_implementation as prior
prior.scope['audit_delta']=audit_delta
def audit(board,physical=None):return prior.audit(board,physical)
