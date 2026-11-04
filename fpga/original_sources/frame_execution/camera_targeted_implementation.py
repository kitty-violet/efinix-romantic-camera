"""Original full pin/clock/reset physical gates for the targeted map."""
from camera_targeted_board import audit_delta
import audit_camera_implementation as prior
prior.scope['audit_delta']=audit_delta
def audit(board,physical=None):return prior.audit(board,physical)
