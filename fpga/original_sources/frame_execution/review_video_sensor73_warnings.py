"""Classify the fresh original-input synthesis warnings before implementation."""
import argparse,collections,json,re
from pathlib import Path
from run_units import sha
from complete_native_logic_split import protected_blocks

def warnings(root):
 text=(root/'outflow/ti60f225_oob.warn.log').read_text()
 return collections.Counter(re.sub(r' \([^\n]*?:\d+\)','',s).strip() for s in text.splitlines() if s.startswith('[EFX-'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);a=p.parse_args();b=a.board.resolve()
 meta=json.loads((b/'inputs.json').read_text());old=Path(meta['parent'])
 assert meta['original_synthesis_input_sha256']==sha(old/'native_source.v')
 assert protected_blocks((b/'native_source.v').read_text())==protected_blocks((old/'native_source.v').read_text())
 proof=json.loads((b/'sensor73-map-audit.json').read_text());assert proof['status']=='SENSOR73_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS'
 before,after=warnings(old),warnings(b);added=after-before;classified=[]
 for message,count in added.items():
  if message in ("[EFX-0011 VERI-WARNING] port 'I3' is not connected on this instance", "[EFX-0011 VERI-WARNING] input port 'I3' is not connected on this instance"):
   assert count==8
   reason='Eight added 3-input mux LUTs do not use I3; full mapped truth tables checked by retained graph audit.'
  elif re.fullmatch(r'\[EFX-0200 WARNING\] Removing redundant signal : (?:protected_inst/)*u_ddr3_top/controller/[^ ]+\.',message):
   reason='Unused internal vendor DDR signals; original encrypted input is byte-identical. Current mapped boundary is audited; fresh routed constraints/Gray/STA remain required.'
  else:raise AssertionError(('unclassified new synthesis warning',message,count))
  classified.append(dict(message=message,count=count,reason=reason))
 result=dict(status='SENSOR73_WARNINGS_REVIEWED_NO_NEW_UNCLASSIFIED',map_sha256=proof['map_sha256'],old_count=sum(before.values()),new_count=sum(after.values()),
  added=classified,removed=list((before-after).items()),unclassified=[],protected_input_identical=True,
  evidence={str(f):sha(f) for f in (Path(__file__),b/'inputs.json',b/'sensor73-map-audit.json',b/'outflow/ti60f225_oob.warn.log',old/'outflow/ti60f225_oob.warn.log',b/'native_source.v',old/'native_source.v')},hardware_access=False,board_release=False)
 (b/'sensor73-warning-review.json').write_text(json.dumps(result,indent=2));print(result['status'],len(classified))

if __name__=='__main__':main()
