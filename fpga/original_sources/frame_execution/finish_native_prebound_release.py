"""Existing full native release gates, with pre-placement Gray binding proof."""
import inspect,json,runpy,sys
from pathlib import Path
import finish_native_split_release as old
from finish_native_split_release import (BOARD,BASE,SHORTS,RTL,SERIAL,MAP_AUDIT,ADAPTER,
    STATUS,parent_evidence,warning_audit,audit_delta)
from run_units import ROOT,HERE,sha
from control_command_repair import once

def extra_evidence(board,batch,physical,serial,bind,verify):
    code=inspect.getsource(old.extra_evidence)
    code=once(code,"'place_native_split_board.py'","'place_native_prebound_board.py'")
    start=code.index('    from rebind_native_placement import audit as rebind_audit')
    end=code.index("    proof=ROOT/'tmp/frame-execution/native-logic-split-fourstate128'",start)
    replacement="""    from native_prebound_constraints import audit as prebound_audit
    bound=prebound_audit(physical);bind(physical/'prebound_constraints.json')
    verify(bound['original']);verify({bound['reference']:bound['reference_sha256'],
        str(HERE/'native_prebound_constraints.py'):bound['script_sha256']})
"""
    code=code[:start]+replacement+code[end:]
    scope=dict(old.__dict__,__file__=__file__)
    exec(compile(code,__file__+':complete-native-gates','exec'),scope)
    scope['extra_evidence'](board,batch,physical,serial,bind,verify)
    verify({str(HERE/'finish_native_prebound_release.py'):sha(__file__),
            str(HERE/'place_native_prebound_board.py'):sha(HERE/'place_native_prebound_board.py')})

def package_code():
    source,code=old.package_code()
    return source,code.replace('finish_native_split_release','finish_native_prebound_release')

def finish_code():
    source,code=old.finish_code()
    return source,code.replace('finish_native_split_release.py','finish_native_prebound_release.py')

def main():
    # Keep the original mode handling, freeze/hash binding and no-download rule.
    code=inspect.getsource(old.main)
    scope=dict(globals());exec(compile(code,__file__+':entry','exec'),scope)
    scope['main']()

if __name__=='__main__':main()
