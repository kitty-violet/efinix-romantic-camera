"""Audit one frozen complete-board map without importing partial timing results."""
import argparse,json
from pathlib import Path
from run_units import ROOT,sha
from audit_connected_system import hierarchy,KEYS

def main():
 p=argparse.ArgumentParser();p.add_argument('--map',type=Path,required=True)
 p.add_argument('--fixed-window',action='store_true')
 p.add_argument('--output-name',default='resource_audit2.json');a=p.parse_args()
 root=a.map.resolve();assert root.is_relative_to(ROOT/'tmp')
 assert Path(a.output_name).name==a.output_name and a.output_name.endswith('.json')
 out=root/a.output_name;assert not out.exists()
 result=json.loads((root/'result.json').read_text());meta=json.loads((root/'inputs.json').read_text())
 # Generated IP folders can contain distinct files with the same basename.
 # Audit the frozen relative path, which is also the build's file identity.
 seen=set()
 for row in meta['inputs']:
  path=Path(row['snapshot']).resolve();assert path.is_relative_to(root) and path not in seen
  assert path.is_file() and sha(path)==row['sha256'],path
  seen.add(path)
 for file,key in [('inputs.json','inputs_sha256'),('outflow/ti60f225_oob.map.rpt','report_sha256'),('outflow/ti60f225_oob.map.v','map_sha256')]:
  assert sha(root/file)==result[key],file
 rows=hierarchy((root/'outflow/ti60f225_oob.map.rpt').read_text())
 # Assign every exclusive contribution exactly once; never sum inclusive parents.
 def group(path):
  if not path.startswith('video_processing_pipeline/pipeline/cnn/'):return 'board_periphery'
  if a.fixed_window:
   for label,part in [('source','/engine/mac/source'),('arithmetic','/engine/mac/arithmetic'),
                      ('weight_cache','/engine/mac/cache'),('partial','/engine/mac/partial'),
                      ('rq','/engine/rq'),('skip','/engine/skip'),('writer','/engine/writer'),
                      ('routing','/engine/router'),('weight_loader','/dut/loader'),('dma','/dma')]:
    if part+'/' in path+'/' or path.endswith(part):return label
   if '/dut/engine/mac' in path:return 'mac_control'
   if '/dut' in path:return 'compute_control'
   return 'other_cnn'
  for label,part in [('dispatch','/dispatch/'),('routing','/route/'),('routing','/router/'),('dma','/dma/'),('reader','/reader/'),('pool','/pool/')]:
   if part in '/'+path+'/':return label
  return 'other_cnn'
 groups={}
 for row in rows:
  dst=groups.setdefault(group(row['path']),dict.fromkeys(KEYS,0))
  for key in KEYS:dst[key]+=row['exclusive'][key]
 for key in KEYS:assert sum(v[key] for v in groups.values())==rows[0]['inclusive'][key],key
 if not a.fixed_window:assert set(groups)=={'board_periphery','dispatch','routing','dma','reader','pool','other_cnn'}
 assert rows[0]['inclusive']['LUT4']==result['resources']['LUT4']
 data=dict(status='BOARD_RESOURCE_AUDIT_PASS_NOT_FIT',map=str(root),inputs_sha256=result['inputs_sha256'],
  report_sha256=result['report_sha256'],map_sha256=result['map_sha256'],official_resources=result['resources'],
  ram_blocks=result['resources']['RAM10']+result['resources']['DPRAM10'],physical_dsp=result['resources']['DSP24']//2+result['resources']['DSP48'],
  device=dict(logic_elements=60800,xlr=62016,ram_blocks=256,physical_dsp=160,
   source='TI60/Ti60 Data Sheet.pdf, page 5, Table 1; XLR=62016, LE=60800',
   official_source_url='https://www.efinixinc.com/docs/titanium60-ds-v3.8.pdf',
   source_sha256=sha(ROOT/'TI60/Ti60 Data Sheet.pdf')),partitions=groups,hierarchy=rows,
  full_board_frame_ms=None,placed_xlr=None,fit_pass=False,sta_pass=False,board_release=False,
  note='Official common map; all inputs hashed, exclusive partitions sum exactly. No partial-frame time transplanted.')
 out.write_text(json.dumps(data,indent=2)+'\n')
 print(json.dumps({k:data[k] for k in ('status','ram_blocks','physical_dsp','partitions')}))

if __name__=='__main__':main()
