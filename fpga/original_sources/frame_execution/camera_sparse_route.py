"""Sparse, documented Titanium route preservation; never a timing waiver.

Keep seven known-good ordinary data nets from route4, not its failed path.
Reuse placement only (no placement rerun, resynthesis, or PHY modification).
"""
import inspect,json,re,shutil,sys
from collections import defaultdict
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from control_command_repair import once
from native_local_drivers import graph

BASE=ROOT/'tmp/frame-execution'
SOURCE=BASE/'camera-checked-bound4'
PHYSICAL=BASE/'camera-checked-bound6'
REVIEW=BASE/'camera-checked-route-review6.json'
PREFIX='video_processing_pipeline/pipeline/cnn/'
NETS=[PREFIX+'core/dut/engine/writer/CURSOR.ROW[0].addr[2]',
      PREFIX+'core/dut/engine/od[451]',
      PREFIX+'fetch/COMPUTE_WORD[2].BIT[57].q',
      PREFIX+'core/dut/engine/router/held_owner[3][1]',
      PREFIX+'core/dut/engine/mac/column_data[204]',
      'n70038_2',PREFIX+'core/dut/engine/mac/column_data[81]']
OUTPUT={'EFX_FF':'Q','EFX_SRL8':'Q','EFX_LUT4':'O'}

def fanouts():
    g=graph((SOURCE/'outflow/ti60f225_oob.map.v').read_text())
    pos={}
    for line in (SOURCE/'outflow/ti60f225_oob.out.placeloc').read_text().splitlines():
        p=line.split()
        if len(p)==4 and not line.startswith('#'):pos[p[0]]=list(map(int,p[1:]))
    index=defaultdict(list)
    for name,(kind,pins) in g.items():
        for port,ts in pins.items():
            for t in set(ts):
                if t[0]!='net':continue
                label=t[1]+('['+str(t[2])+']' if t[2] is not None else '')
                if label in NETS:index[label].append((name,kind,port,t))
    rows=[]
    for net in NETS:
        terms=index[net];assert terms,net
        drivers=[x for x in terms if OUTPUT.get(x[1])==x[2]]
        assert len(drivers)==1,(net,drivers)
        driver=drivers[0];loads=[x for x in terms if x!=driver]
        assert 1<=len(loads)<=12,(net,len(loads))
        assert all(x[3]==driver[3] for x in terms),('ambiguous escaped identifier',net)
        for name,kind,port,_ in terms:
            assert name in pos and not name.startswith('protected_inst/'),name
            assert not re.search(r'CLK|RESET|RST|^SR$|^CE$|^WE',port),port
        rows.append(dict(net=net,driver=driver[:3],loads=[x[:3] for x in loads],
                         locations={x[0]:pos[x[0]] for x in terms}))
    return rows

def prepare():
    import prepare_local_driver_route as old
    import camera_checked_board as camera
    from camera_checked_v2 import BOARD
    from camera_config_clock_ratio import transform
    camera.BOARD=BOARD
    assert not PHYSICAL.exists() and not REVIEW.exists()
    rows=fanouts()
    survey=json.loads((BASE/'camera-near-critical1/result.json').read_text())
    bad={n['net'] for row in survey['rows'] if row['slack_ns']<0 for n in row['nets']}
    assert not bad.intersection(NETS)
    selected=json.loads((BASE/'camera-checked-route4/selected.json').read_text())
    assert selected['wns_ns']==-0.045 and selected['whs_ns']==0.022
    r=json.loads((BASE/'camera-checked-route-review4b.json').read_text())
    r['physical']=str(PHYSICAL)
    r['rationale']=r['rationale'][:4]+[
        'Same immutable route4 placement/map/clock model, C2 recipe and seed3. Expanded STA covers200 distinct core endpoints through+0.357ns;144 paths are below+0.300ns. This band selects analysis only, not acceptance margin.',
        '109 of200 endpoints share CURSOR.ROW[0].addr[2]. Preserve its ordinary data routing plus six measured near-critical/regressed data nets. Full driver/load positions checked. Exclude every net on route4 failed path, clocks, resets, direct enables and PHY.',
        'Official Titanium beta RCF restores selected nets from a separately saved trace. Seven nets only; no whole-template lock. Existing .place is reused by route-only flow, so no placer runs and no new all-cell location constraints or placer keepout is needed.',
        'Previous feedback-loaded-route1 already proves --route with --load_route is rejected; do not repeat. Sparse preservation is an unproven bounded physical trial, not promised convergence.',
        'All original functional/source gates remain. Route1350s/host1395s; canonical WNS/WHS>=0, complete constraints and fresh CDC/Gray still required. No hardware access.'
    ]
    r['remaining_risks'] += ['Seven preserved traces may still constrain other routes; tool acknowledgement and unchanged endpoints are required. Remaining routes may regress; full STA is mandatory.']
    docs=EF/'doc/topics'
    for f in [Path(__file__).resolve(),BASE/'camera-near-critical1/result.json',BASE/'camera-near-critical1/inputs.json',
              BASE/'camera-near-critical1/outflow/near_setup.rpt',BASE/'camera-checked-route5/selected.json',
              SOURCE/'outflow/ti60f225_oob.troutingtraces',SOURCE/'outflow/ti60f225_oob.place',
              SOURCE/'outflow/ti60f225_oob.map.v',SOURCE/'outflow/ti60f225_oob.out.placeloc',
              *[docs/n for n in ('rcf-file.html','rcf-flow.html','rcf-bp.html','rcf-enable.html')]]:
        r['reviewed_evidence'][str(f)]=sha(f)
    REVIEW.write_text(json.dumps(r,indent=2))
    code=once(inspect.getsource(old.main),"'after':'TIMING_1'","'after':'CONGESTION_2'")
    env=dict(old.__dict__,audit_delta=camera.audit_delta)
    exec(compile(code,__file__+':copy','exec'),env)
    sys.argv=[str(HERE/'prepare_local_driver_route.py'),'--review',str(REVIEW)];env['main']()
    # Same exact config-clock ratio correction as routes3/4, no new exceptions.
    sdc=PHYSICAL/'ti60f225_oob.sdc';before=sha(sdc)
    text,change=transform(sdc.read_text(),(PHYSICAL/'ti60f225_oob.peri.xml').read_text());sdc.write_text(text)
    ratio=dict(status='EXACT_CONFIG_CORE_PLL_RATIO_MODEL',change=change,before_sdc_sha256=before,
        after_sdc_sha256=sha(sdc),transform_sha256=sha(HERE/'camera_config_clock_ratio.py'),
        peri_sha256=sha(PHYSICAL/'ti60f225_oob.peri.xml'),hardware_access=False,board_release=False)
    model=PHYSICAL/'config_clock_ratio.json';model.write_text(json.dumps(ratio,indent=2))
    assert sha(sdc)==sha(SOURCE/'ti60f225_oob.sdc')
    trace=PHYSICAL/'preserved_route4.troutingtraces'
    shutil.copy2(SOURCE/'outflow/ti60f225_oob.troutingtraces',trace)
    locations={name:xyz for row in rows for name,xyz in row['locations'].items()}
    original_loc=(PHYSICAL/'fixed_dsp.placeloc').read_text()
    existing={line.split()[0]:list(map(int,line.split()[1:])) for line in original_loc.splitlines() if len(line.split())==4 and not line.startswith('#')}
    for n,xyz in locations.items():
        if n in existing:assert existing[n]==xyz
    added=''.join(n+' '+' '.join(map(str,xyz))+'\n' for n,xyz in sorted(locations.items()) if n not in existing)
    loc=PHYSICAL/'sparse_route.placeloc';loc.write_text(original_loc.rstrip()+'\n'+added)
    rcf=PHYSICAL/'sparse_route.rcf'
    rcf.write_text('routeTraceFile '+trace.as_posix()+'\n'+''.join('restoreNetFromTraceFile '+n+'\n' for n in NETS))
    ini=PHYSICAL/'efx_pnr_settings.ini';original_ini=ini.read_text()
    assert original_ini=='loc_assignment = fixed_dsp.placeloc\ndump_placeloc = on\n'
    ini.write_text('loc_assignment = sparse_route.placeloc\ndump_placeloc = on\nrcf_file = sparse_route.rcf\ngenerate_rcf_template = on\n')
    proof=dict(status='SEVEN_ORDINARY_NET_PRESERVATION_PREPARED_NOT_TIMING_PASS',source=str(SOURCE),
        source_files={str(SOURCE/n):sha(SOURCE/n) for n in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.place','outflow/ti60f225_oob.out.placeloc','outflow/ti60f225_oob.troutingtraces','ti60f225_oob.sdc','efx_pnr_settings.ini')},
        rows=rows,excluded_failed_path_nets=sorted(bad),original_ini=original_ini,
        changed_files={str(f):sha(f) for f in (trace,loc,rcf,ini)},script_sha256=sha(__file__),
        placement_rerun=False,clock_or_reset_routes_locked=False,hardware_access=False)
    record=PHYSICAL/'sparse_route_preservation.json';record.write_text(json.dumps(proof,indent=2))
    meta_file=PHYSICAL/'physical_inputs.json';meta=json.loads(meta_file.read_text())
    for row in meta['inputs']:
        row['sha256']=sha(row['snapshot'])
    meta['clock_profile']['before_config_ratio_sha256']=meta['clock_profile']['after_sha256']
    meta['clock_profile']['after_sha256']=sha(sdc);meta['config_clock_ratio_sha256']=sha(model)
    meta['sparse_route_preservation_sha256']=sha(record);meta_file.write_text(json.dumps(meta,indent=2))
    review=PHYSICAL/'route_review.json';r=json.loads(review.read_text())
    for f in (meta_file,model,sdc,record,trace,loc,rcf,ini):r['required_evidence'][str(f)]=sha(f)
    review.write_text(json.dumps(r,indent=2))
    print(json.dumps(dict(status=proof['status'],nets=len(rows),cells=len(locations),physical=str(PHYSICAL))))

def route():
    from route_local_driver_board import driver_code
    from camera_checked_flow import generated
    source,code=driver_code();assert code.count('TIMING_1')==2
    code=code.replace('TIMING_1','CONGESTION_2')
    code=once(code,'from local_driver_board import audit_delta','from camera_checked_board import audit_delta')
    code=once(code,'import audit_control_state_implementation as impl','import audit_camera_implementation as impl')
    code=once(code,"HERE/'audit_native_split_gray.py'","HERE/'camera_checked_gray_v2.py'")
    # Strict source gates and canonical acceptance are unchanged.
    generated(source,code,'-local-driver-route')

if __name__=='__main__':
    action=sys.argv.pop(1)
    if action=='prepare':prepare()
    elif action=='route':route()
    else:raise ValueError(action)
