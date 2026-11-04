"""Run the unchanged routed Gray checks with all current live crossings."""
import inspect,sys
from pathlib import Path
import camera_checked_gray as prior

def main():
    code=inspect.getsource(prior.main)
    assert 'camera_checked_cdc' in code
    code=code.replace('camera_checked_cdc','camera_checked_cdc_v2')
    env=dict(prior.__dict__);env['__file__']=__file__
    exec(compile(code,__file__+':driver','exec'),env);env['main']()

if __name__=='__main__':main()
