"""Recover completed map measurements only when tool rewrites XML EOF newline.

Never edits a frozen input, its manifest, or the original failed gate result.
All RTL/constraints must remain byte-identical. XML reconstruction must match
the exact pre-run hash; accepting arbitrary semantic changes is forbidden.
"""
import argparse,collections,hashlib,json,re
from pathlib import Path
from run_units import ROOT,sha
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
assert root.is_relative_to(ROOT/'tmp')
meta=json.loads((root/'inputs.json').read_text());normalizations=[]
for row in meta['inputs']:
 f=Path(row['snapshot']);actual=f.read_bytes()
 if sha(f)==row['sha256']:continue
 assert f==root/'ti60f225_oob.xml',str(f)
 suffixes=[b'',b'\n',b'\r\n',b'\n\n',b'\r\n\r\n']
 reconstruct=[s for s in suffixes if hashlib.sha256(actual.rstrip(b'\r\n')+s).hexdigest()==row['sha256']]
 assert len(reconstruct)==1,'More than an EOF newline changed'
 normalizations.append(dict(file=str(f),before_sha256=row['sha256'],after_sha256=sha(f),change='XML EOF newline only',pre_run_suffix_hex=reconstruct[0].hex()))
assert re.search(r'map\s*:\s*PASS',(root/'map.log').read_text())
rpt=root/'outflow/ti60f225_oob.map.rpt';net=root/'outflow/ti60f225_oob.map.v';warn=root/'outflow/ti60f225_oob.warn.log'
counts={k:int(v) for k,v in re.findall(r'^EFX_([A-Z0-9_]+)\s*:\s*(\d+)',rpt.read_text(),re.M)}
dumped=dict(collections.Counter(re.findall(r'^\s*EFX_([A-Z0-9_]+)\s+\S+',net.read_text(),re.M)))
for k in ('RAM10','DPRAM10','DSP24','DSP48'):assert counts.get(k,0)==dumped.get(k,0)
warnings=warn.read_text(errors='replace');prior=(Path(meta['base'])/'outflow/ti60f225_oob.warn.log').read_text(errors='replace')
pat=r'^\[EFX-0201 WARNING\] (.*?)\. \('
added=sorted(set(re.findall(pat,warnings,re.M))-set(re.findall(pat,prior,re.M)))
result=dict(status='MAP_MEASUREMENT_WITH_EXACT_XML_EOF_NORMALIZATION_AUDIT',resources=counts,new_undriven_signatures=added,comb_loop_warning='EFX-0654' in warnings,normalizations=normalizations,inputs_sha256=sha(root/'inputs.json'),map_sha256=sha(net),report_sha256=sha(rpt),warning_sha256=sha(warn),resource_fit_proven=False,physical_timing_pass=False,board_download_allowed=False)
dst=root/'completion_audit.json';assert not dst.exists();dst.write_text(json.dumps(result,indent=2));print(json.dumps(result))
