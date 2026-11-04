"""Original strict STA with complete reviewed-cone RCF evidence."""
import inspect
from canonical_sparse_snapshot import *
import canonical_sparse_snapshot as previous
code=once(inspect.getsource(previous.main),'from audit_camera_sparse_route import audit','from audit_camera_sparse_route_v3 import audit')
exec(compile(code,__file__+':canonical-with-cone-audit','exec'),globals())
if __name__=='__main__':main()
