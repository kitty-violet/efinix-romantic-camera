"""Evidence binding and honest timing/resource boundaries for this wave."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,sha
BASE=ROOT/'tmp/frame-execution'
def evidence(name):
 root=BASE/name;result=json.loads((root/'result.json').read_text());manifest=json.loads((root/'inputs.json').read_text())
 assert result['inputs_sha256']==sha(root/'inputs.json')
 sources={}
 for row in manifest['inputs']:
  assert sha(row['snapshot'])==row['sha256']
  if Path(row['path']).suffix in ('.v','.sv') and not Path(row['path']).name.startswith('tb_'):assert sha(row['path'])==row['sha256']
  sources[Path(row['path']).name]=row['sha256']
 for rel,h in manifest.get('vectors',{}).items():assert sha(root/'vectors'/rel)==h
 if 'log_sha256' in result:
  log=root/'simulation.log';assert sha(log)==result['log_sha256'];assert not re.search(r'^#\s*(FAIL|\*\* (Error|Fatal))',log.read_text(),re.M)
 for name,h in result.get('logs',{}).items():assert sha(root/name)==h
 for key,glob in [('map_sha256','*.map.v'),('warning_sha256','*.warn.log'),('report_sha256','*.map.rpt')]:
  if key in result:
   files=list((root/'outflow').glob(glob));assert len(files)==1 and sha(files[0])==result[key]
 assert not result.get('full_frame',False) and not result.get('pnr',False)
 return dict(root=str(root),result_sha256=sha(root/'result.json'),sources=sources,result=result)
def bind(sim,mapping):
 common=set(sim['sources'])&set(mapping['sources']);assert common
 for n in common:assert sim['sources'][n]==mapping['sources'][n],n
 return sorted(common)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
 names=['stride-tile-source2','stride-tile-map1','stride-scatter4-source2','stride-scatter4-map1','stride-scatter4-bank-source1','stride-scatter4-bank-map1','stride-scatter4-compact-source1','stride-scatter4-compact-map1','stride-pipeline-source2','compact-video-source2','compact-video-map1','reference-video-map1','compact-area-video-pipeline-map1','reference-area-video-pipeline-map1']
 records={n:evidence(n) for n in names};pairs=[]
 for s,m in [('stride-tile-source2','stride-tile-map1'),('stride-scatter4-source2','stride-scatter4-map1'),('stride-scatter4-bank-source1','stride-scatter4-bank-map1'),('stride-scatter4-compact-source1','stride-scatter4-compact-map1'),('stride-pipeline-source2','stride-tile-map1'),('stride-pipeline-source2','stride-scatter4-compact-map1'),('compact-video-source2','compact-video-map1')]:pairs.append(dict(sim=s,map=m,matching_sources=bind(records[s],records[m])))
 video_root=BASE/'compact-area-video-pipeline2';video=json.loads((video_root/'result.json').read_text());assert video['area'] and video['status'].endswith('_PASS')
 for row in video['results']:
  d=video_root/f'{"compact" if row["compact"] else "reference"}-{"wide" if row["wide"] else "small"}'
  manifest=json.loads((d/'inputs.json').read_text());assert sha(d/'inputs.json')==row['inputs_sha256']
  for r in manifest['inputs']:assert sha(r['path'])==sha(r['snapshot'])==r['sha256']
  assert sha(d/'pixels.hex')==row['pixel_sha256'] and sha(d/'simulation.log')==row['log_sha256']
  assert not re.search(r'^#\s*(FAIL|\*\* (Error|Fatal))',(d/'simulation.log').read_text(),re.M)
  if row['wide']:
   m=records['compact-area-video-pipeline-map1' if row['compact'] else 'reference-area-video-pipeline-map1']
   for r in manifest['inputs']:
    n=Path(r['path']).name
    if n in m['sources']:assert r['sha256']==m['sources'][n]
 assert video['results'][0]['pixel_sha256']==video['results'][1]['pixel_sha256']
 assert video['results'][2]['pixel_sha256']==video['results'][3]['pixel_sha256']
 res=records['compact-area-video-pipeline-map1']['result']['resources'];old=records['reference-area-video-pipeline-map1']['result']['resources']
 assert (old['RAM10'],res['RAM10'])==(44,35)
 rows=records['stride-pipeline-source2']['result']['pipeline_rows'];parsed=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',r)} for r in rows]
 d1=next(r for r in parsed if r['case']==2 and r['random']==0);d2=next(r for r in parsed if r['case']==5 and r['random']==0)
 assert (d1['input_to_all_write_ack'],d2['input_to_all_write_ack'])==(1123,1249)
 contract=ROOT/'network_ab/model_b_contract.json';assert sha(contract)=='5750c4668d75077746ec5d4accf8773ab437540cf933c202f153a8797c21dd01'
 result=dict(status='STRIDE_VIDEO_LOCAL_EVIDENCE_BINDING_PASS',pairs=pairs,evidence=records,video_ab=video,
  local_timing=dict(stride_consumer_only_cycles=315*480+217*480,stride_consumer_only_ms_at_150MHz=(315*480+217*480)/150000,
   serial_interior_tile_extrapolation_cycles=1123*480+1249*240,serial_interior_tile_extrapolation_ms_at_150MHz=(1123*480+1249*240)/150000,
   scope='Representative interior tile extrapolation, NOT whole-frame timing; includes synthetic input scatter, real resident fetch, compute, write ACK and exercised command/status waits. Real upstream production, exact edge schedule, DDR and rest of graph are absent.'),
  resource_budget=dict(cnn_fifo80=128,ddr_reference=39,csi_reference=11,raw_reference=30,video_area_compact=35,cpu_reference=15,conditional_sum=258,device_ram=256,whole_system_fit=False),
  decision=dict(serial_scatter_selected_for_30fps=False,compact_scatter_adoption_gate_passed=False,video_candidate='variants/compact_area_fifo/rgb_scaler_display_pipeline.sv'),
  measured_frame_ms=None,full_frame_30fps=False,final_mapped_function=False,pnr=False,board_test=False,model_contract_sha256=sha(contract))
 a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ('evidence','pairs','video_ab')}))
if __name__=='__main__':main()
