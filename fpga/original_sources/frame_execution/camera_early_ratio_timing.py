"""One timing-oriented route after C2 proved this new placement routable."""
from run_units import HERE
from control_command_repair import once
body=(HERE/'camera_early_ratio_route.py').read_text()
body=once(body,'    code=once(code,"\'after\':\'TIMING_1\'","\'after\':\'CONGESTION_2\'")\n','')
body=once(body,"        code=code.replace('TIMING_1','CONGESTION_2')\n",'')
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
exec(compile(prefix,__file__+':timing-route-only','exec'),globals())
if __name__=='__main__':main()
