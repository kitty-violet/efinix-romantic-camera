"""Complete existing release gates plus sparse-RCF exact provenance checks."""
from run_units import HERE
from control_command_repair import once
from audit_camera_sparse_route import audit as sparse_audit

# Evaluate the same v3 gate construction with a narrowly scoped settings audit.
# All other original geometry hashes, timing checks and CDC tests still apply.
body=(HERE/'finish_camera_checked_release_v3.py').read_text()
needle='    return code\n\n\n# Reuse the exact v2 evidence definitions'
replacement='''    code=once(code,"verify({str(physical/n):h for n,h in rebound['geometry'].items()})",
        "verify({str(physical/n):h for n,h in rebound['geometry'].items() if n!='efx_pnr_settings.ini'})\\n assert rebound['geometry']['efx_pnr_settings.ini']==sha(Path(bind(physical/'sparse_route_preservation.json')['source'])/'efx_pnr_settings.ini')\\n sparse_audit(physical,bind,verify)")
    return code


# Reuse the exact v2 evidence definitions'''
body=once(body,needle,replacement)
body=body.replace('finish_camera_checked_release_v3','finish_camera_checked_release_v4')
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
body=prefix+tail
exec(compile(body,__file__+':v3-gates','exec'),globals())
previous_package=package_code

def package_code():
    source,code=previous_package()
    code=once(code,"routing_only_target=bind(physical/'targeted_route_margin.json') if (physical/'targeted_route_margin.json').exists() else None,",
        "routing_only_target=bind(physical/'targeted_route_margin.json') if (physical/'targeted_route_margin.json').exists() else None,sparse_route=bind(physical/'sparse_route_preservation.json'),")
    code=code.replace("HERE/'CAMERA_CHECKED_RELEASE.md',out/'README.md'","HERE/'CAMERA_CHECKED_RELEASE_SPARSE.md',out/'README.md'")
    code=once(code,"    for f in out.glob('*.py'):","    verify({str(HERE/'audit_camera_sparse_route.py'):sha(HERE/'audit_camera_sparse_route.py'),str(HERE/'CAMERA_CHECKED_RELEASE_SPARSE.md'):sha(HERE/'CAMERA_CHECKED_RELEASE_SPARSE.md')})\n    for f in out.glob('*.py'):")
    compile(code,'camera-v4-package-preflight','exec')
    return source,code

if __name__=='__main__':main()
