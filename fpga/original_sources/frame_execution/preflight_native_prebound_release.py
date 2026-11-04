"""Same mandatory release checks and failed-STA rejection, for early binding."""
import inspect
import preflight_native_release as old
import finish_native_prebound_release as release

def main():
    code=inspect.getsource(old.main).replace('native-package-preflight1','native-prebound-package-preflight1')
    scope=dict(old.__dict__,release=release,__file__=__file__)
    exec(compile(code,__file__+':mandatory-gates','exec'),scope)
    scope['main']()

if __name__=='__main__':main()
