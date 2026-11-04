"""Check existing same-map release evidence before waiting on routing.

Stops before all route-dependent checks; it is never a release substitute.
"""
import inspect,json
from run_units import ROOT,HERE,sha
import finish_local_driver_release as release

def main():
    root=ROOT/'tmp/board-test/local-driver-evidence-preflight1'
    assert not root.exists();root.mkdir();evidence={}
    def bind(path):
        evidence[str(path)]=sha(path);return json.loads(path.read_text())
    def verify(rows):
        for f,h in rows.items():assert sha(f)==h,f;evidence[str(f)]=h
    parent=bind(release.PARENT/'manifest.json');release.parent_evidence(parent,verify)
    source=inspect.getsource(release.extra_evidence)
    prefix=source[:source.index('    # Retain the complete serial/physical gate')]
    scope=dict(release.__dict__);exec(compile(prefix,__file__+':current-map-only','exec'),scope)
    scope['extra_evidence'](release.BOARD,None,None,None,bind,verify)
    report=dict(status='EXISTING_PARENT_AND_CURRENT_MAP_SHORT_EVIDENCE_BOUND',
        checked_evidence=len(evidence),evidence=evidence,
        preflight_sha256=sha(__file__),release_sha256=sha(release.__file__),
        routed_sta_checked=False,board_release=False,hardware_access=False)
    (root/'result.json').write_text(json.dumps(report,indent=2));print(report['status'],len(evidence))

if __name__=='__main__':main()
