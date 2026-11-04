"""Complete unchanged gates bound to the explicit TIMING_1 route adapter."""
from run_units import HERE
body=(HERE/'finish_camera_early_ratio_release.py').read_text()
body=body.replace('finish_camera_early_ratio_release','finish_camera_early_timing_release')
body=body.replace('camera_early_ratio_route.py','camera_early_ratio_timing.py')
body=body.replace('from camera_early_ratio_route import prepare_code','from camera_early_ratio_timing import prepare_code')
body=body.replace('CAMERA_CHECKED_RELEASE_EARLY_RATIO.md','CAMERA_CHECKED_RELEASE_EARLY_TIMING.md')
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
exec(compile(prefix,__file__+':full-timing-route-release','exec'),globals())
if __name__=='__main__':main()
