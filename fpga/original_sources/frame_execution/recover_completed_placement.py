"""Recover native-checked placement after report-time timeout, not a fake PASS."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
a=p.parse_args();base=a.base.resolve();root=a.root.resolve()
assert base.is_relative_to(ROOT/'tmp/frame-execution') and root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
failure=json.loads((base/'failure.json').read_text());assert 'TimeoutExpired' in failure['error']
assert (base/'pnr_timeout_cleanup.json').exists()
body=(base/'outflow/ti60f225_oob.place.out').read_text()
markers=('Placement successful:','Post-placement cluster-level checks is successful','Post-placement primitive-level checks is successful')
assert all(m in body for m in markers)
assert not (base/'outflow/ti60f225_oob.troutingtraces').exists()
root.mkdir()
for folder in ('rtl','assembly','outflow','work_pnr'):shutil.copytree(base/folder,root/folder)
for f in base.iterdir():
    if f.is_file() and (f.suffix in ('.xml','.sdc','.bin','.v','.sv','.vh','.svh','.ini','.placeloc')
                        or f.name in ('physical_runner.py','capacity_gate.json')):shutil.copy2(f,root/f.name)
meta=json.loads((base/'physical_inputs.json').read_text())
for row in meta['inputs']:
    row['snapshot']=str(root/Path(row['snapshot']).relative_to(base));assert sha(row['snapshot'])==row['sha256']
(root/'physical_inputs.json').write_text(json.dumps(meta,indent=2))
report=(root/'outflow/ti60f225_oob.place.rpt').read_text()
resources={k:dict(used=int(u),capacity=int(c)) for k,u,c in re.findall(r'^(XLRs|Memory Blocks|DSP Blocks):\s*(\d+)\s*/\s*(\d+)',report,re.M)}
assert len(resources)==3 and all(r['used']<=r['capacity'] for r in resources.values())
peaks={k:float(v) for k,v in re.findall(r'Peak congestion smeared over (4x4|1/16|1/4) of the chip is ([\d.]+)',body)}
assert len(peaks)==3
hpwl=float(re.search(r'Congestion-weighted HPWL per net: ([\d.]+)',body)[1])
reference=ROOT/'tmp/frame-execution/registered-ownership-physical1/place-seed1'
old=json.loads((reference/'placement_gate.json').read_text())['current']
assert hpwl<old['weighted_hpwl_per_net'] and all(peaks[k]<v for k,v in old['congestion_peaks'].items())
evidence={str(base/'failure.json'):sha(base/'failure.json'),str(base/'pnr_timeout_cleanup.json'):sha(base/'pnr_timeout_cleanup.json'),
          str(reference/'placement_gate.json'):sha(reference/'placement_gate.json')}
for name in ('ti60f225_oob.map.v','ti60f225_oob.place','ti60f225_oob.vdb','ti60f225_oob.out.placeloc','ti60f225_oob.place.out','ti60f225_oob.place.rpt'):
    assert sha(base/'outflow'/name)==sha(root/'outflow'/name);evidence[str(base/'outflow'/name)]=sha(base/'outflow'/name)
gate=dict(status='PLACEMENT_RECOVERED_FOR_BOUNDED_ROUTE',current=dict(resources=resources,congestion_peaks=peaks,
    weighted_hpwl_per_net=hpwl,placement_legal=True,canonical_setup_estimate_ns=None),reference=old,
    reasons=['Placement completed and native legality checked; optional placement timing report was interrupted. No timing PASS claimed.'],
    required_evidence=evidence,board_release=False,hardware_access=False)
(root/'placement_gate.json').write_text(json.dumps(gate,indent=2))
result=dict(status='PLACEMENT_RECOVERED_AFTER_REPORT_TIMEOUT_NOT_RELEASE',stages=failure['stages'],source=str(base),
            placement_gate_sha256=sha(root/'placement_gate.json'),timing_checked=False,board_release=False)
(root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(gate['current']))
