"""Complete unchanged release gates for the reviewed timing-cone preservation."""
from run_units import HERE
body=(HERE/'finish_camera_checked_release_v5.py').read_text()
for before,after in (
    ('canonical_sparse_snapshot_v2.py','canonical_sparse_snapshot_v3.py'),
    ('audit_camera_sparse_route_v2','audit_camera_sparse_route_v3'),
    ('camera_sparse_route_launch.py','camera_sparse_route_launch_v3.py'),
    ('finish_camera_checked_release_v5','finish_camera_checked_release_v6'),
    ('CAMERA_CHECKED_RELEASE_SPARSE2.md','CAMERA_CHECKED_RELEASE_SPARSE3.md')):
    assert before in body,before
    body=body.replace(before,after)
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
exec(compile(prefix,__file__+':v5-gates','exec'),globals())
if __name__=='__main__':main()
