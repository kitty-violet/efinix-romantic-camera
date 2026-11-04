"""Read-only comparison of the two frozen routes; never launches Efinity."""
import json,re,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,sha


def paths(report):
    rows=[]
    for block in report.read_text().split('++++ Path ')[1:]:
        def field(label):
            match=re.search(r'^'+label+r'\s*:\s*(.+)$',block,re.M)
            assert match,label
            return match[1].strip()
        data=block.split('\nData Path\n',1)[1].split('\nCapture Clock Path',1)[0]
        nets=[dict(fanout=int(n),delay_ns=float(d),net=name.strip()) for n,d,name in
              re.findall(r'^\s+net \(fo=(\d+)\)\s+([\d.]+)\s+[\d.]+\s+(.+)$',data,re.M)]
        rows.append(dict(start=field('Path Start'),end=field('Path End'),
            slack_ns=float(field('Slack').split()[0]),nets=nets))
    assert len(rows)==5
    return rows


def main():
    root=ROOT/'tmp/frame-execution/camera-route-regression1'
    assert not root.exists();root.mkdir()
    base=ROOT/'tmp/frame-execution';physical=[base/f'camera-checked-bound{i}' for i in (4,5)]
    runs=[base/f'camera-checked-route{i}' for i in (4,5)]
    evidence={str(Path(__file__).resolve()):sha(__file__)}
    for name in ('outflow/ti60f225_oob.map.v','outflow/ti60f225_oob.place','ti60f225_oob.peri.xml'):
        assert sha(physical[0]/name)==sha(physical[1]/name)
        for p in physical:evidence[str(p/name)]=sha(p/name)
    assert sha(physical[0]/'ti60f225_oob.sdc')==sha(physical[1]/'camera_acceptance.sdc')
    rows=[]
    for p,r in zip(physical,runs):
        canonical=r/'canonical-seed3';result=json.loads((canonical/'result.json').read_text())
        assert result['status']=='CANONICAL_ROUTED_TIMING_FAIL'
        for f in (canonical/'result.json',canonical/'inputs.json',canonical/'outflow/canonical_setup.rpt',
                  canonical/'ti60f225_oob.sdc',p/'ti60f225_oob.sdc',r/'failure.json'):
            evidence[str(f)]=sha(f)
        rows.append(dict(run=str(r),wns_ns=result['rows']['setup']['worst_ns'],
            whs_ns=result['rows']['hold']['worst_ns'],checks=result['check_timing_actual'],
            reported_worst_paths=paths(canonical/'outflow/canonical_setup.rpt')))
    # The release adapter must actually reject this failed canonical receipt.
    import finish_camera_checked_release_v3 as release
    def bind(p):evidence[str(p)]=sha(p);return json.loads(p.read_text())
    def verify(items):
        for p,h in items.items():assert sha(p)==h;evidence[str(p)]=h
    try:
        release.targeted_evidence(physical[1],runs[1],(physical[0]/'ti60f225_oob.sdc').read_text(),bind,verify)
    except AssertionError:
        frames=traceback.extract_tb(__import__('sys').exc_info()[2])
        assert frames[-1].line=="assert result['status']=='CANONICAL_ROUTED_TIMING_PASS_NOT_RELEASE'"
    else:raise AssertionError('Failed STA was accepted')
    evidence[str(Path(release.__file__).resolve())]=sha(release.__file__)
    docs=EF/'doc/topics'
    for n in ('rcf-intro.html','rcf-enable.html','rcf-file.html','rcf-flow.html','rcf-bp.html'):
        evidence[str(docs/n)]=sha(docs/n)
    out=dict(status='SAME_PLACEMENT_ROUTE_REGRESSION_CONFIRMED_NOT_RELEASED',
        hardware_access=False,board_release=False,selected_for_release=None,
        same_map_placement_and_physical_clocks=True,same_final_acceptance_sdc=True,
        release_rejected_failed_sta=True,rows=rows,
        conclusion='One stricter routing target changed routing globally and regressed other endpoints. Keep route4 as the closest failed reference, never as a released bitstream.',
        next_method='Review all near-critical data nets and complete fanouts on route4; use documented sparse RCF trace preservation only after validating net/cell identity and clock/reset compatibility. No whole-template lock and no claim of incremental timing closure yet.',
        documented_rcf='Titanium routeTraceFile + restoreNetFromTraceFile; associated cell locations must remain fixed. Official beta feature, untested on this candidate.',
        evidence=evidence)
    (root/'result.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:out[k] for k in ('status','hardware_access','board_release','release_rejected_failed_sta')}))


if __name__=='__main__':main()
