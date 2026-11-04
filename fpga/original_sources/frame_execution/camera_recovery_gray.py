"""Unchanged routed Gray gate plus both old and new camera crossing checks."""
import inspect
import camera_checked_gray as prior
def main():
 code=inspect.getsource(prior.main);assert 'camera_checked_cdc' in code
 code=code.replace('camera_checked_cdc','camera_recovery_cdc')
 env=dict(prior.__dict__,__file__=__file__)
 exec(compile(code,__file__+':driver','exec'),env);env['main']()
if __name__=='__main__':main()
