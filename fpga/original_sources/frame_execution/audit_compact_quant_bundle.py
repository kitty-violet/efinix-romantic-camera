"""Bind compact source, native, four-state, unit and resource evidence."""
import json
from pathlib import Path
from run_units import ROOT,sha
def main():
 home=ROOT/'tmp/frame-execution';names=('compact-quant-source1','compact-quant-native1','compact-quant-map1','compact-quant-unit1','stride-column-source1','stride-column-native1')
 results={n:json.loads((home/n/'result.json').read_text()) for n in names}
 for n,j in results.items():assert sha(home/n/'inputs.json')==j['inputs_sha256']
 assert results['compact-quant-source1']['results'][0]['rows']==results['stride-column-source1']['results'][0]['rows']
 assert results['compact-quant-native1']['rows']==results['stride-column-native1']['rows']
 inputs={n:json.loads((home/n/'inputs.json').read_text()) for n in names[:3]}
 native={Path(r['snapshot']).name:r['sha256'] for r in inputs['compact-quant-native1']['inputs']}
 for n in ('compact-quant-source1','compact-quant-map1'):
  for r in inputs[n]['inputs']:
   assert sha(r['snapshot'])==r['sha256']==native[Path(r['snapshot']).name]
 resource=results['compact-quant-map1']['resources']
 assert resource['LUT4']+resource['ADD']+resource['SRL8']==48996
 assert resource['DSP24']==256 and resource['DSP48']==32 and resource['RAM10']==32
 warn=ROOT/'output/toolchain/compact_quant_native_warning_audit_2026-09-30.json'
 w=json.loads(warn.read_text());assert not w['unknown'] and w['native_result_sha256']==sha(home/'compact-quant-native1/result.json')
 out=ROOT/'output/toolchain/compact_quant_shared_cluster_bundle_2026-09-30.json';assert not out.exists()
 result=dict(status='COMPACT_QUANT_SHARED_CLUSTER_SOURCE_NATIVE_MAP_MATCH_PASS',same_numeric_and_cycles=True,combo_lower_bound=48996,
  physical_dsp=160,local_ram_including_arena_resident=136,mapped_functional_gate=False,full_frame_30fps=False,final_resource_fit=False,pnr=False,
  scope='Normal/DW/PW/NN/stem/filtered-stride2 modes. New compact residual end-to-end regression remains open; filtered-stride2 throughput is rejected.',
  evidence={str(p):sha(p) for p in [Path(__file__),warn,*[home/n/'result.json' for n in names]]})
 out.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
