"""Unchanged full-design STA, with complete-cone RCF evidence and trace copies."""
import inspect
from canonical_sparse_snapshot import *
import canonical_sparse_snapshot as previous
code=once(inspect.getsource(previous.main),'from audit_camera_sparse_route import audit','from audit_camera_sparse_route_v4 import audit')
exec(compile(code,__file__+':canonical-complete-cones','exec'),globals())
if __name__=='__main__':main()
