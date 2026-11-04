"""One full-board ledger: source identity, exclusive accounting, explicit open leaves.

This is a resource audit, not a fit/STA or end-to-end performance certificate.
Every baseline source name must remain present in each compared board build.
"""
import argparse,csv,json,re
from pathlib import Path
from run_units import ROOT,sha
from audit_connected_system import manifest,hierarchy,KEYS

REQUIRED={
 'Sapphire CPU':'cpu', 'APB控制与装载':'control', 'MIPI CSI接收':'inst_efx_csi2_rx',
 'DDR桥与厂商控制器':'u_ddr3_top', '相机RAW读写与CDC':'raw_frame_bridge',
 'RAW行缓存':'raw_line_source', '去马赛克':'debayer_stream',
 'RGB主链与CNN':'video_processing_pipeline', 'TMDS编码':'dvi_encoder_m0',
 'CNN生产发令与计算':'video_processing_pipeline/pipeline/cnn',
 '四主fabric及CPU位宽适配':'video_processing_pipeline/pipeline/fabric',
 '显示DDR读出':'video_processing_pipeline/pipeline/display',
 '输出升采样':'video_processing_pipeline/upscaler', '显示输出FIFO':'video_processing_pipeline/output_fifo',
 '显示时序调度':'video_processing_pipeline/scheduler',
}

def read_board(root):
 result=json.loads((root/'result.json').read_text());meta,names=manifest(root)
 for name,key in [('inputs.json','inputs_sha256'),('outflow/ti60f225_oob.map.rpt','report_sha256'),('outflow/ti60f225_oob.map.v','map_sha256')]:
  assert sha(root/name)==result[key],name
 assert result['status']=='CONNECTED_BOARD_MAP_MEASURED_NOT_RELEASE'
 assert not result['comb_loop_warning']
 assert sha(root/'outflow/ti60f225_oob.warn.log')==result['warning_sha256']
 rows=hierarchy((root/'outflow/ti60f225_oob.map.rpt').read_text());by_path={r['path']:r for r in rows}
 for name,path in REQUIRED.items():
  assert path in by_path and by_path[path]['inclusive']['LUT4']>0,(name,path)
 assert sum(r['exclusive']['LUT4'] for r in rows)==result['resources']['LUT4']
 # These are disjoint, whole subtrees, except the explicitly labelled residual.
 peripheral={}
 for r in rows:
  path=r['path']
  if path.startswith('video_processing_pipeline/pipeline/cnn'):continue
  label=path.split('/')[0]
  if label=='video_processing_pipeline':
   parts=path.split('/')
   label='/'.join(parts[:3] if len(parts)>1 and parts[1]=='pipeline' else parts[:2])
  dst=peripheral.setdefault(label,dict.fromkeys(KEYS,0))
  for key in KEYS:dst[key]+=r['exclusive'][key]
 cnn=by_path['video_processing_pipeline/pipeline/cnn']['inclusive']
 for key in KEYS:assert cnn[key]+sum(x[key] for x in peripheral.values())==rows[0]['inclusive'][key],key
 return dict(root=str(root),result=result,names=names,rows=rows,peripheral=peripheral,
  included={name:dict(path=path,resources=by_path[path]['inclusive']) for name,path in REQUIRED.items()})

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True)
 p.add_argument('--candidate',type=Path,action='append',required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--selected',type=Path,help='Baseline or candidate retained for the next build; default last candidate');a=p.parse_args()
 out=a.output.resolve();assert out.is_relative_to(ROOT/'output') and not out.exists()
 boards=[read_board(v.resolve()) for v in [a.baseline]+a.candidate];ref=boards[0]
 for b in boards[1:]:
  assert b['result']['new_undriven_signatures']==[],b['root']
  assert b['names'].keys()==ref['names'].keys(),'Component added/omitted: update scope review first'
  b['changed_inputs']=[n for n in b['names'] if b['names'][n]!=ref['names'][n]]
  b['delta']={k:v-ref['result']['resources'][k] for k,v in b['result']['resources'].items()}
 final=boards[-1]
 if a.selected:
  chosen=[b for b in boards if Path(b['root'])==a.selected.resolve()]
  assert len(chosen)==1,'Selected build must be the baseline or one compared candidate'
  final=chosen[0]
 out.mkdir(parents=True)
 # Inclusion rows are coverage witnesses and may nest. Never sum them.
 # Only this exclusive CSV and the official root summary are additive.
 with (out/'exclusive_modules.csv').open('w',newline='',encoding='utf-8-sig') as f:
  writer=csv.DictWriter(f,fieldnames=['path',*KEYS]);writer.writeheader()
  writer.writerows(dict(path=r['path'],**r['exclusive']) for r in final['rows'])
 open_leaves=[
  dict(name='左手进入/同一交互手划出消失退出',status='NOT_INTEGRATED',resources=None,
   treatment='保留基线，算法与硬件预算未完成；不得填0，发布预算OPEN'),
  dict(name='恶灵像素后处理及警告红橙色保护',status='NOT_INTEGRATED',resources=None,
   treatment='保留基线，后续真实map单独计入并最终共同map'),
  dict(name='真实AI FPS左上角叠字',status='NOT_INTEGRATED',resources=None,
   treatment='已有提交计数不等于FPS计时/叠字完成，不按现有计数器面积替代'),
 ]
 data=dict(status='WHOLE_MAIN_CHAIN_RESOURCE_SCOPE_AUDIT_PASS_NOT_FIT',selected_board=final['root'],
  device=dict(xlr=62016,logic_elements=60800,ram=256,dsp=160),
  boards=[{k:v for k,v in b.items() if k not in ('rows','names')} for b in boards],
  baseline_scope_sha256=sha(ROOT/'BASELINE_SCOPE_LOCK.md'),
  source_name_count=len(ref['names']),unimplemented_required_leaves=open_leaves,
  baseline_complete=False,resource_fit=False,placed_xlr=None,frame_ms=None,board_download_allowed=False)
 (out/'resource_scope.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
 print(json.dumps(dict(status=data['status'],source_name_count=data['source_name_count'],
  resources=final['result']['resources'],delta=final.get('delta'),open_leaves=len(open_leaves)),ensure_ascii=False))
if __name__=='__main__':main()
