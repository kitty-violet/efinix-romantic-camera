"""Apply the verified config/core PLL ratio before native packing/placement.

Same checked3 map, recipe, seed and hard-column anchors as place2. Stop after
placement for a new whole-chip routability review; never route or program here.
"""
import inspect,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once


def driver_code():
    source=HERE/'probe_local_decode_place.py'
    code=source.read_text(encoding='utf8')
    code=once(code,'from local_control_decode import audit_delta',
        'from camera_checked_v2 import BOARD\nimport camera_checked_board\n'
        'camera_checked_board.BOARD=BOARD\naudit_delta=camera_checked_board.audit_delta')
    threads='''xml=root/'ti60f225_oob.xml';before_threads=xml.read_text()
needle='<efx:param name="max_threads" value="-1" value_type="e_integer" />'
assert before_threads.count(needle)==1
xml.write_text(before_threads.replace(needle,needle.replace('value="-1"','value="8"')),encoding='utf8')
for rec in records:
 if Path(rec['snapshot'])==xml:rec['sha256']=sha(xml)
(root/'pnr_threads.json').write_text(json.dumps(dict(official_max_threads=8,
 scope='Host P&R only; same mapped hardware'),indent=2))
optimization_change=None'''
    ratio='''assert clock_profile and not a.setup_reserve_proof
from camera_config_clock_ratio import transform
sdc=root/'ti60f225_oob.sdc';before_ratio_sha=sha(sdc)
new_sdc,ratio_change=transform(sdc.read_text(),(root/'ti60f225_oob.peri.xml').read_text())
sdc.write_text(new_sdc,encoding='utf8')
reference_sdc=ROOT/'tmp/frame-execution/camera-checked-bound4/ti60f225_oob.sdc'
from rebind_native_placement import translate
rebound,_=translate(new_sdc,(root/'outflow/ti60f225_oob.map.v').read_text(),
 (ROOT/'tmp/frame-execution/camera-checked-place2/outflow/ti60f225_oob.out.placeloc').read_text())
assert rebound==reference_sdc.read_text(),'Only the existing post-place endpoint spelling may differ'
model=root/'config_clock_ratio.json'
model.write_text(json.dumps(dict(status='EXACT_CONFIG_CORE_PLL_RATIO_MODEL',change=ratio_change,
 before_sdc_sha256=before_ratio_sha,after_sdc_sha256=sha(sdc),
 transform_sha256=sha(Path(__import__('camera_config_clock_ratio').__file__)),
 peri_sha256=sha(root/'ti60f225_oob.peri.xml'),applied_before='interface_pack_place',
 hardware_access=False,board_release=False),indent=2))
for rec in records:
 if Path(rec['snapshot'])==sdc:rec['sha256']=sha(sdc)
clock_profile['before_config_ratio_sha256']=clock_profile['after_sha256']
clock_profile['after_sha256']=sha(sdc)
setup_margin=None'''
    anchor="source=HERE/'probe_fourpll_reset_physical.py';code=source.read_text()"
    code=once(code,anchor,anchor+
        "\n    code=once(code,'optimization_change=None',"+repr(threads)+")"+
        "\n    code=once(code,'setup_margin=None',"+repr(ratio)+")"+
        "\n    code=once(code,'clock_profile=clock_profile,','clock_profile=clock_profile,config_clock_ratio_sha256=sha(model),')")
    code=once(code,'from audit_global_place import metrics',
        'from camera_ratio_before_place import metrics')
    code=once(code,"ref=ROOT/'tmp/frame-execution/command-progress-seed3-1/pnr-seed1'",
        "ref=ROOT/'tmp/frame-execution/native-local-driver-place1'")
    compile(code,str(source)+':ratio_before_place','exec')
    return source,code


def metrics(root):
    import audit_global_place as old
    code=once(inspect.getsource(old.metrics),'assert len(rows)==22,len(rows)',
        'assert len(rows) in (22,23),len(rows)')
    env=dict(old.__dict__)
    exec(compile(code,__file__+':clock_rows','exec'),env)
    result=env['metrics'](root)
    if (root/'config_clock_ratio.json').exists():
        pair=[r for r in result['setup_clock_relationships']
              if (r['launch'],r['capture'])==('CLK_5M','process_clk_150')]
        assert len(pair)==1 and abs(pair[0]['constraint_ns']-6.666)<0.0001,pair
    return result


def main():
    source,code=driver_code()
    if sys.argv[1:]==['--check-only']:
        import camera_config_clock_ratio
        base=ROOT/'tmp/frame-execution/camera-checked-place2'
        after,record=camera_config_clock_ratio.transform((base/'ti60f225_oob.sdc').read_text(),
            (base/'ti60f225_oob.peri.xml').read_text())
        from rebind_native_placement import translate
        rebound,_=translate(after,(base/'outflow/ti60f225_oob.map.v').read_text(),
                            (base/'outflow/ti60f225_oob.out.placeloc').read_text())
        assert rebound==(ROOT/'tmp/frame-execution/camera-checked-bound4/ti60f225_oob.sdc').read_text()
        old=metrics(base)
        print(json.dumps(dict(status='DRIVER_COMPILES_PRIOR_VALIDATED_RATIO_EXACTLY_REPRODUCED',
            old_crossing=[r for r in old['setup_clock_relationships'] if r['launch']=='CLK_5M'],
            hardware_access=False)))
        return
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    assert root.name=='camera-checked-place3' and not root.exists()
    previous=ROOT/'tmp/frame-execution/camera-checked-route10/stages.json'
    assert all(s['status']!='running' for s in json.loads(previous.read_text()))
    driver=root.parent/(root.name+'-ratio-place-wrapper.py')
    assert not driver.exists()
    driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),applied_before='interface_pack_place',
        ordinary_logic_free=True,hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]]
    runpy.run_path(str(driver),run_name='__main__')


if __name__=='__main__':main()
