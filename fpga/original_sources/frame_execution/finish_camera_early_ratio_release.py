"""Same complete hardware gates; clock correction is now before placement."""
from run_units import HERE
from control_command_repair import once

body=(HERE/'finish_camera_checked_release_v3.py').read_text()
body=body.replace('finish_camera_checked_release_v3','finish_camera_early_ratio_release')
start=body.index('    add=""" ratio=')
end=body.index('    code=once(code,needle,add)',start)
early=''' ratio=bind(physical/'config_clock_ratio.json')
 meta=bind(physical/'physical_inputs.json')
 assert meta['config_clock_ratio_sha256']==sha(physical/'config_clock_ratio.json')
 assert ratio['applied_before']=='interface_pack_place'
 verify({str(HERE/'camera_config_clock_ratio.py'):ratio['transform_sha256'],str(physical/'ti60f225_oob.peri.xml'):ratio['peri_sha256']})
 import hashlib,os
 from camera_config_clock_ratio import transform
 origin_sdc=(origin/'ti60f225_oob.sdc').read_text()
 fixed='create_clock -period 199.980 -name CLK_5M [get_ports {CLK_5M}]'
 old='create_clock -period 200.000 -name CLK_5M [get_ports {CLK_5M}]'
 assert origin_sdc.count(fixed)==1 and old not in origin_sdc
 before=origin_sdc.replace(fixed,old)
 assert hashlib.sha256(before.replace('\\n',os.linesep).encode('utf8')).hexdigest()==ratio['before_sdc_sha256']
 corrected,change=transform(before,(origin/'ti60f225_oob.peri.xml').read_text())
 assert corrected==origin_sdc and change==ratio['change']
 assert sha(origin/'ti60f225_oob.sdc')==ratio['after_sdc_sha256']
 assert not (physical/'targeted_route_margin.json').exists()
 assert (physical/'ti60f225_oob.sdc').read_text()==expected and pins==rebound['pins'];'''
body=body[:start]+'    add='+repr(early)+'\n'+body[end:]
changes='''    code=code.replace("physical.name+'-place-wrapper.py'","physical.name+'-ratio-place-wrapper.py'")
    code=once(code,"wrappers(physical,[('-place-wrapper',physical.parent/(physical.name+'-local-driver-wrapper.py')),('-local-driver-wrapper',HERE/'camera_checked_flow.py')],bind,verify)",
                   "wrappers(physical,[('-ratio-place-wrapper',HERE/'camera_ratio_before_place.py')],bind,verify)")
    code=once(code,"str(HERE/'prepare_local_driver_route.py'):rebound['script_sha256']",
                   "str(physical.parent/(physical.name+'-prepare-ratio.py')):rebound['script_sha256']")
    code=once(code,' camera_evidence(bind,verify)',""" preparation=physical.parent/(physical.name+'-prepare-ratio.py')
 pm=bind(preparation.with_suffix('.json'))
 verify({str(preparation):pm['generated_sha256'],pm['source']:pm['source_sha256'],str(HERE/'camera_early_ratio_route.py'):pm['adapter_sha256']})
 from camera_early_ratio_route import prepare_code
 assert preparation.read_text()==prepare_code()[1]
 verify({str(HERE/'camera_ratio_before_place.py'):sha(HERE/'camera_ratio_before_place.py')})
 camera_evidence(bind,verify)""")
    return code


# Reuse'''
body=once(body,'    return code\n\n\n# Reuse',changes)
prefix,guard,tail=body.rpartition("if __name__=='__main__':main()")
assert guard and not tail.strip()
exec(compile(prefix,__file__+':complete-hardware-gates','exec'),globals())

early_camera_evidence=camera_evidence
def camera_evidence(bind,verify):
    r=early_camera_evidence(bind,verify)
    b=bind(ROOT/'tmp/board-test/camera-reader-budget1/result.json')
    assert b['status']=='CAMERA_READER_SEPARATE_BUDGET_TESTS_PASS' and b['tests']==9
    verify(b['evidence']);r['separate_budget_checks']=b['tests'];return r

early_package=package_code
def package_code():
    source,code=early_package()
    code=once(code,'from read_camera_checked_snapshot import main','from read_camera_checked_snapshot_v2 import main')
    code=once(code,"HERE/'CAMERA_CHECKED_RELEASE.md',out/'README.md'","HERE/'CAMERA_CHECKED_RELEASE_EARLY_RATIO.md',out/'README.md'")
    code=once(code,"    for f in out.glob('*.py'):compile",
        "    verify({str(HERE/'CAMERA_CHECKED_RELEASE_EARLY_RATIO.md'):sha(HERE/'CAMERA_CHECKED_RELEASE_EARLY_RATIO.md')})\n    for f in out.glob('*.py'):compile")
    compile(code,'early-ratio-package-preflight','exec');return source,code

if __name__=='__main__':main()
