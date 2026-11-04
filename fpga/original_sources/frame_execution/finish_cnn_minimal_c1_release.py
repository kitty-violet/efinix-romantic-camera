"""Full current-map gates for C1 placement, preserving the 36-cell island."""
from control_command_repair import once
from cnn_small_release_factory import code_for
from audit_legacy_cache_recovery import install,prove,MANIFEST
install()

code=code_for('minimal')
code=code.replace('finish_cnn_minimal_release','finish_cnn_minimal_c1_release')
code=code.replace('cnn_minimal_route','cnn_minimal_c1_route')
code=code.replace('cnn_small_place.py','cnn_minimal_repack.py')
code=once(code,"dict(before='CONGESTION_2',after='CONGESTION_2')",
          "dict(before='CONGESTION_1',after='CONGESTION_2')")
recovery_tail='''
_complete_extra_evidence=extra_evidence
def extra_evidence(board,batch,physical,serial,bind,verify):
    _complete_extra_evidence(board,batch,physical,serial,bind,verify)
    recovery=bind(MANIFEST)
    assert recovery==prove()
    verify(recovery['evidence'])

_original_package_code=package_code
def package_code():
    source,body=_original_package_code()
    detail=dict(status=prove()['status'],manifest=str(MANIFEST),sha256=sha(MANIFEST),historical_vdb_bytes_restored=False)
    body=once(body,'manifest=dict(','manifest=dict(legacy_cache_revalidation='+repr(detail)+',')
    return source,body

if __name__=='__main__':main()
'''
inner_exec="exec(compile(code,str(SOURCE)+':joint-closure-release','exec'),globals())"
code=once(code,inner_exec,"code=once(code,\"if __name__=='__main__':main()\","+repr(recovery_tail)+")\n"+inner_exec)
exec(compile(code,__file__+':full-gates-c1-placement','exec'),globals())
