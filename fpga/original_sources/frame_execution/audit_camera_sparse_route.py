"""Bind sparse RCF settings without weakening any geometry or timing check."""
import json,re
from pathlib import Path
from run_units import HERE,sha

def audit(physical,bind,verify,require_routed=True):
    from camera_sparse_route import SOURCE,NETS,fanouts
    rule=bind(physical/'sparse_route_preservation.json')
    meta=bind(physical/'physical_inputs.json')
    assert meta['sparse_route_preservation_sha256']==sha(physical/'sparse_route_preservation.json')
    assert rule['status']=='SEVEN_ORDINARY_NET_PRESERVATION_PREPARED_NOT_TIMING_PASS'
    assert Path(rule['source'])==SOURCE and not rule['placement_rerun']
    assert not rule['clock_or_reset_routes_locked'] and not rule['hardware_access']
    verify(rule['source_files']);verify(rule['changed_files'])
    verify({str(HERE/'camera_sparse_route.py'):rule['script_sha256']})
    assert rule['rows']==json.loads(json.dumps(fanouts()))
    assert len(NETS)==7 and not set(NETS).intersection(rule['excluded_failed_path_nets'])
    for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.place','outflow/ti60f225_oob.out.placeloc','ti60f225_oob.sdc'):
        assert sha(physical/name)==sha(SOURCE/name),name
    assert rule['original_ini']==(SOURCE/'efx_pnr_settings.ini').read_text()
    assert (physical/'efx_pnr_settings.ini').read_text()=='loc_assignment = sparse_route.placeloc\ndump_placeloc = on\nrcf_file = sparse_route.rcf\ngenerate_rcf_template = on\n'
    trace=physical/'preserved_route4.troutingtraces'
    assert sha(trace)==sha(SOURCE/'outflow/ti60f225_oob.troutingtraces')
    expected='routeTraceFile '+trace.as_posix()+'\n'+''.join('restoreNetFromTraceFile '+n+'\n' for n in NETS)
    assert (physical/'sparse_route.rcf').read_text()==expected
    original=(physical/'fixed_dsp.placeloc').read_text()
    existing={p[0]:list(map(int,p[1:])) for line in original.splitlines() if not line.startswith('#') and len(p:=line.split())==4}
    points={n:xyz for row in rule['rows'] for n,xyz in row['locations'].items()}
    assert all(existing[n]==xyz for n,xyz in points.items() if n in existing)
    added=''.join(n+' '+' '.join(map(str,xyz))+'\n' for n,xyz in sorted(points.items()) if n not in existing)
    assert (physical/'sparse_route.placeloc').read_text()==original.rstrip()+'\n'+added
    if require_routed:
        log=physical/'outflow/ti60f225_oob.route.out';body=log.read_text()
        assert '7/79318 nets will be constrained (reloaded)' in body
        assert "Parsing routing constraints file 'sparse_route.rcf'" in body
        assert 'Parallel router is disabled when Constrained Router is used.' in body
        it=re.findall(r'Iter\s+(\d+):\s*overlap=(\d+)',body)
        assert it and int(it[-1][1])==0
        assert not re.search(r'^\s*(?:ERROR|FATAL)\s*:',body,re.M)
        verify({str(log):sha(log)})
    return dict(status='SPARSE_RCF_SOURCE_GEOMETRY_AND_TOOL_BINDING_PASS' if require_routed else 'SPARSE_RCF_PRE_ROUTE_BINDING_PASS',
                nets=7,cells=len(points),map_placement_clocks_unchanged=True,
                new_timing_exceptions=0,hardware_access=False)

if __name__=='__main__':
    import sys
    evidence={}
    def verify(rows):
        for f,h in rows.items():assert sha(f)==h,f
        evidence.update(rows)
    def bind(p):verify({str(p):sha(p)});return json.loads(p.read_text())
    result=audit(Path(sys.argv[1]).resolve(),bind,verify,'--pre-route' not in sys.argv)
    print(json.dumps(result))
