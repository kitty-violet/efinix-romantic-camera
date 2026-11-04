"""Apply the established static and failing-timing rejection release tests."""
import inspect
import preflight_native_release as original
import finish_native_timing_release as release
from control_command_repair import once

if __name__=='__main__':
    code=inspect.getsource(original.main)
    code=once(code,'native-package-preflight1','native-timing-package-preflight1')
    scope=dict(original.__dict__,release=release,__file__=__file__)
    exec(compile(code,__file__+':same-release-tests','exec'),scope);scope['main']()
