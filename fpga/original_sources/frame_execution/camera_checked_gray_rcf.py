"""All original live CDC/Gray checks with complete RCF collateral copies."""
import inspect
import camera_checked_gray as previous
from control_command_repair import once
code=inspect.getsource(previous.main).replace('camera_checked_cdc','camera_checked_cdc_v2')
code=once(code," injections=[", " injections=[\n  (\"('.xml','.sdc','.ini','.placeloc')\",\"('.xml','.sdc','.ini','.placeloc','.rcf','.troutingtraces')\"),")
env=dict(previous.__dict__,__file__=__file__)
exec(compile(code,__file__+':same-checks-full-collateral','exec'),env)
if __name__=='__main__':env['main']()
