"""Complete existing Gray/held-path gates, including current camera truth cut."""
import inspect
import camera_checked_gray as prior
def main():
 code=inspect.getsource(prior.main);assert 'camera_checked_cdc' in code
 code=code.replace('camera_checked_cdc','camera_joint_cdc')
 env=dict(prior.__dict__,__file__=__file__)
 exec(compile(code,__file__+':joint-driver','exec'),env);env['main']()
if __name__=='__main__':main()
