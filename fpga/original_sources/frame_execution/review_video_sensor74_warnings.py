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
 proof=json.loads((b/'sensor74-map-audit.json').read_text());assert proof['status']=='SENSOR74_EXACT_SOURCE_SCOPE_FRESH_MAP_PASS'
 before,after=warnings(old),warnings(b);added=after-before;classified=[]
 for message,count in added.items():
  if message=="[EFX-0300 WARNING] Instantiated LUT 'LUT__241563' has LUTMASK '16'b1100111111001111'. It does not depend on connected input 'I0'. Connection will be removed.":
   assert count==1
   reason='Deliberate absorption of the old index-bit AND into existing decode. The replacement final LUT ignores I0; all 131072 combined cases and actual mapped-cell truth comparison cover this exact pruning.'
  elif re.fullmatch(r'\[EFX-0200 WARNING\] Removing redundant signal : (?:protected_inst/)*inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/RxStopState_reg_c\[7\]\.',message):
   assert count==1 and proof['protected_boundary']['source_protected_ports_and_blob_unchanged']
   assert proof['protected_boundary']['mapped_added']==proof['protected_boundary']['mapped_removed']==[]
   assert proof['protected_boundary']['mapped_literal_differences']==['LUT__241563']
   reason='Unused internal CSI PHY signal. Original protected source and all 1201 protected input/output connections are unchanged; public mapped cells match the expected narrow delta except proven unused sensor LUT input. New physical and timing evidence is still required.'
  else:raise AssertionError(('unclassified new synthesis warning',message,count))
  classified.append(dict(message=message,count=count,reason=reason))
 result=dict(status='SENSOR74_WARNINGS_REVIEWED_NO_NEW_UNCLASSIFIED',map_sha256=proof['map_sha256'],old_count=sum(before.values()),new_count=sum(after.values()),
  added=classified,removed=list((before-after).items()),unclassified=[],protected_input_identical=True,
  evidence={str(f):sha(f) for f in (Path(__file__),b/'inputs.json',b/'sensor74-map-audit.json',b/'outflow/ti60f225_oob.warn.log',old/'outflow/ti60f225_oob.warn.log',b/'native_source.v',old/'native_source.v')},hardware_access=False,board_release=False)
 (b/'sensor74-warning-review.json').write_text(json.dumps(result,indent=2));print(result['status'],len(classified))

if __name__=='__main__':main()
