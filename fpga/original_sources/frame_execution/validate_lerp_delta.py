"""Exhaustive RGB channel interpolation equivalence, official four-state simulator."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,HERE,MS,run,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
f=HERE/'variants/raw_row_delta/tb_lerp_delta.sv';shutil.copy2(f,r/f.name)
run([MS/'vlib.exe','work'],r,'vlib.log')
body,_=run([MS/'vlog.exe','-sv',f.name],r,'compile.log')
assert 'Errors: 0, Warnings: 0' in body
body,secs=run([MS/'vsim.exe','-c','work.tb_lerp_delta','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',90)
assert 'PASS LERP_DELTA checks=524288' in body and not re.search(r'\*\* (Fatal|Error)',body)
(r/'result.json').write_text(json.dumps(dict(status='EXHAUSTIVE_OFFICIAL_LERP_PASS',checks=524288,seconds=secs,source_sha256=sha(f),log_sha256=sha(r/'simulation.log')),indent=2))
print((r/'result.json').read_text())

