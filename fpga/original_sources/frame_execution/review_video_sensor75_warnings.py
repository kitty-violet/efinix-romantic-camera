"""Classify the fresh original-input synthesis warnings before implementation."""
import argparse,collections,json,pickle,re
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
 proof=json.loads((b/'sensor75-map-audit.json').read_text());assert proof['status']=='SENSOR75_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS'
 _,g,_=pickle.loads((b/'sensor75-map-parse.pkl').read_bytes())
 from continuous_control_native import connections
 _,loads=connections(g)
 from native_local_drivers import lexical_tokens
 def unused(name):
  ts=lexical_tokens('\\'+name+' ')
  return bool(ts) and all(not loads[t] for t in ts)
 before,after=warnings(old),warnings(b);added=after-before;classified=[]
 for message,count in added.items():
  if message=="[EFX-0300 WARNING] Instantiated LUT 'LUT__241563' has LUTMASK '16'b1100111111001111'. It does not depend on connected input 'I0'. Connection will be removed.":
   assert count==1
   reason='Deliberate absorption of the old index-bit AND into existing decode. The replacement final LUT ignores I0; all 131072 combined cases and actual mapped-cell truth comparison cover this exact pruning.'
  elif re.fullmatch(r'\[EFX-0200 WARNING\] Removing redundant signal : (?:protected_inst/)*u_ddr3_top/controller/[^ ]+\.',message):
   assert count==1 and proof['protected_boundary']['source_protected_ports_and_blob_unchanged']
   reason='Vendor marks an internal DDR signal redundant. The complete encrypted source and its interface are unchanged, and all retained mapped non-LUT parameters are checked. No whole-map equivalence claim; new Gray endpoint and original constraint STA checks remain mandatory.'
  elif message in {"[EFX-0011 VERI-WARNING] Input/inout port jtag_inst1_"+p+" is unconnected. Place and route may remove this unconnected port." for p in ('DRCK','RUNTEST','TMS')}:
   assert count==1
   reason='These existing unused JTAG inputs retain the identical original source declarations and connections. No JTAG input was deleted or reassigned.'
  elif (m:=re.fullmatch(r'\[EFX-0011 VERI-WARNING\] Net (cell_probe/done_sync\[1\]|clock_probe/meta\[[45]\]|factor_address_0(?:25|26|28|29|31|32|34|35)) is dangling\. \(VDB-8009\)',message)):
   assert count==1 and unused(m[1]),('unexpected surviving load',m[1])
   reason='Outputless cleanup exposes an unused net. The fresh mapped primitive graph confirms zero loads; source cleanup also checks non-primitive ports and assignments.'
  elif message=='[EFX-0011 VERI-WARNING] Found 15 warnings in the post-synthesis netlist.':
   assert count==1
   reason='Summary count for individually classified or previously existing post-synthesis warnings; not an additional signal fault.'
  else:raise AssertionError(('unclassified new synthesis warning',message,count))
  classified.append(dict(message=message,count=count,reason=reason))
 result=dict(status='SENSOR75_WARNINGS_REVIEWED_NO_NEW_UNCLASSIFIED',map_sha256=proof['map_sha256'],old_count=sum(before.values()),new_count=sum(after.values()),
  added=classified,removed=list((before-after).items()),unclassified=[],protected_input_identical=True,
  evidence={str(f):sha(f) for f in (Path(__file__),b/'inputs.json',b/'sensor75-map-audit.json',b/'outflow/ti60f225_oob.warn.log',old/'outflow/ti60f225_oob.warn.log',b/'native_source.v',old/'native_source.v')},hardware_access=False,board_release=False)
 (b/'sensor75-warning-review.json').write_text(json.dumps(result,indent=2));print(result['status'],len(classified))

if __name__=='__main__':main()
