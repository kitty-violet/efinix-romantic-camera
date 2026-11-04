"""Retain the measured regressed enable tree, after shared-control review."""
from pathlib import Path
from run_units import HERE,sha
from control_command_repair import once

body=(HERE/'camera_sparse_route.py').read_text()
body=body.replace('camera-checked-bound6','camera-checked-bound7').replace('camera-checked-route-review6','camera-checked-route-review7')
body=once(body,"OUTPUT={'EFX_FF'","NETS.append('local_driver_net_261')\nOUTPUT={'EFX_FF'")
body=once(body,"    pos={}","    from probe_full_native_split_map import params\n    attributes=params((SOURCE/'outflow/ti60f225_oob.map.v').read_text())\n    pos={}")
body=once(body,"assert 1<=len(loads)<=12,(net,len(loads))",
    "assert 1<=len(loads)<=(18 if net=='local_driver_net_261' else 12),(net,len(loads))")
body=once(body,"assert not re.search(r'CLK|RESET|RST|^SR$|^CE$|^WE',port),port",'''if net=='local_driver_net_261':
                # No relocation: all18 existing CE loads have the same clock,
                # synchronous reset, polarity and priority in the exact map.
                if name!=driver[0]:
                    assert kind=='EFX_FF' and port=='CE'
                    pins=g[name][1]
                    assert pins['CLK']==[('net','process_clk_150',None)]
                    assert pins['SR']==[('net',PREFIX+'core/dut/engine/wsr',5)]
                    attrs=attributes[name]
                    assert attrs=={'CLK_POLARITY':(1,1),'CE_POLARITY':(1,1),'SR_POLARITY':(1,0),'D_POLARITY':(1,1),'SR_SYNC':(1,1),'SR_VALUE':(1,0),'SR_SYNC_PRIORITY':(1,1)}
                else:assert kind=='EFX_LUT4' and port=='O'
            else:assert not re.search(r'CLK|RESET|RST|^SR$|^CE$|^WE',port),port''')
body=body.replace('plus six measured near-critical/regressed data nets. Full driver/load positions checked. Exclude every net on route4 failed path, clocks, resets, direct enables and PHY.',
    'plus six measured near-critical/regressed data nets and one separately reviewed synchronous CE tree. Full driver/load positions checked. Exclude every net on route4 failed path, clocks, resets and PHY.')
body=body.replace('Seven nets only;','Eight nets only;')
body=body.replace('Seven preserved traces may still constrain other routes','Eight preserved traces may still constrain other routes')
body=once(body,"    docs=EF/'doc/topics'",'''    r['rationale'] += [
        'Route6 loaded seven nets and completed legal routing in728.125s, but failed setup-0.226ns. With complete RCF collateral, all5 timing coverage checks are0; no tool crash remains.',
        'Frozen-route endpoint probes show original failing RAM endpoint now+1.670ns. New worst enable path was+2.122ns inroute4 and-0.226ns inroute6; local_driver_net_261 wire increased1.120to3.042ns. Preserve this known-good route4 CE tree in addition to the seven prior data nets.',
        'Complete map review finds18 FF CE loads, same process150 clock, same wsr[5] synchronous active-low reset, same reset priority and polarity; all driver/load coordinates and shared packing remain unchanged. This is an enable route, not a clock/reset route, and is separately accounted for.'
    ]
    for f in (BASE/'camera-checked-route6/canonical-rcf-complete1/result.json',BASE/'camera-control-route4-probe1/result.json',BASE/'camera-control-route6-probe1/result.json',HERE/'camera_sparse_route.py',HERE/'canonical_sparse_snapshot.py'):
        r['reviewed_evidence'][str(f)]=sha(f)
    docs=EF/'doc/topics' ''')
# route6 has no selected.json because its first canonical run rejected missing
# RCF collateral. Bind the repaired complete result above, retain old failure.
body=body.replace('SEVEN_ORDINARY_NET_PRESERVATION_PREPARED_NOT_TIMING_PASS','SEVEN_DATA_ONE_ENABLE_PRESERVATION_PREPARED_NOT_TIMING_PASS')
body=once(body,'placement_rerun=False,clock_or_reset_routes_locked=False,hardware_access=False)',
    "placement_rerun=False,clock_or_reset_routes_locked=False,enable_route=dict(net='local_driver_net_261',loads=18,identical_clock_reset_and_polarities=True,locations_unchanged=True),hardware_access=False)")
body=once(body,"    generated(source,code,'-local-driver-route')",'''    code=once(code,"HERE/'canonical_snapshot_sta.py'","HERE/'canonical_sparse_snapshot_v2.py'")
    generated(source,code,'-local-driver-route')''')
prefix,guard,tail=body.rpartition("if __name__=='__main__':")
assert guard
exec(compile(prefix,__file__+':reviewed-v1-adapter','exec'),globals())

if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
