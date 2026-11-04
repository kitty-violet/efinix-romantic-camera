"""Exact retained graph with one proved unloaded legacy diagnostic FF pruned."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_service_map.py');code=source.read_text()
anchor=' missing=sorted(a.keys()-b.keys());added=sorted(b.keys()-a.keys());assert not missing'
replacement=''' discarded=[]
 dead='cnn_run_seen~FF'
 if dead in a and dead not in b:
  meta=json.loads((board/'inputs.json').read_text());native=Path(meta['native'])
  graft=json.loads((native/'graft.json').read_text())
  assert graft['serializer_extension']['legacy_progress_qualifier']=='service_page_sequence[4]'
  assert a[dead][0]=='EFX_FF' and a[dead][1]['Q']==[('net','cnn_run_seen',None)]
  # Exactly declaration and its own output connection; no RTL expression,
  # public primitive or encrypted boundary can consume this abandoned flag.
  assert len(re.findall(r'(?<![\\w/])cnn_run_seen(?![\\w~])',source))==2
  assert not any(('net','cnn_run_seen',None) in ts for n,(k,pins) in a.items() if n!=dead for ts in pins.values())
  del a[dead];discarded.append(dead)
 missing=sorted(a.keys()-b.keys());added=sorted(b.keys()-a.keys());assert not missing'''
assert code.count(anchor)==1;code=code.replace(anchor,replacement)
code=code.replace("retained=len(a),added=len(added)","retained=len(a),discarded_unloaded_diagnostic_cells=discarded,added=len(added)")
code=code.replace('checker_sha256=sha(__file__),','checker_sha256=sha(__file__),original_checker_sha256=sha(Path(__file__).with_name("audit_video_service_map.py")),')
exec(compile(code,str(source)+':unloaded-epoch-flag','exec'),globals())
