"""Full native candidate synthesis probe; never a physical/board release."""
import argparse,copy,json,re,shutil,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from complete_native_logic_split import resource_counts

BASE=ROOT/'tmp/frame-execution/connected-board-continuous-srl-register1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--prepare-only',action='store_true');p.add_argument('--run-existing',action='store_true');a=p.parse_args()
 src=a.native.resolve();out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp')
 if not a.run_existing:
  assert not out.exists();out.mkdir(parents=True)
  proof=json.loads((src/'native-graph-audit.json').read_text());assert proof['status']=='SERVICE_NATIVE_RETAINED_GRAPH_AND_CUT_PASS'
  for f,h in proof['inputs'].items():assert sha(f)==h,f
  meta=json.loads((BASE/'inputs.json').read_text());rows=[]
  for row in meta['inputs']:
   old=Path(row['snapshot']);assert sha(old)==row['sha256'],old
   rel=old.relative_to(BASE);replacement=src/rel
   origin=replacement if replacement.exists() else old
   dst=out/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dst)
   rows.append(dict(source=str(origin),snapshot=str(dst),sha256=sha(dst)))
  extras=['frame_pair_owner.sv','rgb_frame_admission.sv','video_run_button.sv','video_display_observer.sv','video_debug_mailbox.sv','video_display_recovery.sv','video_service_modes.sv','video_service_health.sv']
  extras.append('video_dualclock_sample.sv')
  for name in extras:
   shutil.copy2(src/name,out/name);rows.append(dict(source=str(src/name),snapshot=str(out/name),sha256=sha(out/name)))
  xml=out/'ti60f225_oob.xml';s=xml.read_text()
  s=s.replace('<efx:top_vhdl_arch', '\n'.join('<efx:design_file name="'+name+'" version="default" library="default" />' for name in extras)+'\n<efx:top_vhdl_arch',1)
  xml.write_text(s,encoding='utf8')
  peri=out/'ti60f225_oob.peri.xml';s=peri.read_text();assert 'GPIOL_07' not in s
  old=re.search(r'<efxpt:comp_gpio name="i_arstn".*?</efxpt:comp_gpio>',s,re.S)[0]
  new=old.replace('i_arstn','service_key_n').replace('GPIOR_22','GPIOL_07')
  s=s.replace(old,old+'\n'+new,1);peri.write_text(s,encoding='utf8')
  for row in rows:row['sha256']=sha(row['snapshot'])
  shutil.copy2(BASE/'map.cmd',out/'map.cmd')
  (out/'inputs.json').write_text(json.dumps(dict(inputs=rows,parent=str(BASE),parent_sha256=sha(BASE/'inputs.json'),
    native=str(src),native_audit_sha256=sha(src/'native-graph-audit.json'),builder_sha256=sha(__file__),
    added_pin='SW0 C4 GPIOL_07',original_sdc_unchanged=True,physical_cdc_and_pin_audit_pending=True,
    hardware_access=False,board_release=False),indent=2))
 if a.prepare_only:print('VIDEO_SERVICE_MAP_PREPARED',out);return
 meta=json.loads((out/'inputs.json').read_text());assert not (out/'map.log').exists()
 for r in meta['inputs']:assert sha(r['snapshot'])==r['sha256']
 result=dict(status='RUNNING',hardware_access=False,physical_pass=False,board_release=False);start=time.monotonic()
 try:
  body,_=run(['cmd.exe','/d','/c','map.cmd'],out,'map.log',480)
  assert re.search(r'map\s*:\s*PASS',body),body
  result.update(status='VIDEO_SERVICE_COMPLETE_NATIVE_SYNTHESIS_PROBE_PASS',counts=resource_counts(out/'outflow/ti60f225_oob.map.rpt'),
   map_sha256=sha(out/'outflow/ti60f225_oob.map.v'),vdb_sha256=sha(out/'outflow/ti60f225_oob.vdb'))
  warnings=(out/'outflow/ti60f225_oob.warn.log').read_text();assert '[EFX-0654' not in warnings,'combinational loop'
  assert not re.search(r"net 'service_[^']*' does not have a driver",warnings),'new service net undriven'
  assert not re.search(r"Set/reset port 'SR' of instance 'video_processing_pipeline/[^']*' is always enabled",warnings),'video reset synthesized permanently active'
  result['warnings_pending_review']=True
 except Exception as e:result.update(status='FAIL',error=repr(e));raise
 finally:
  result['seconds']=time.monotonic()-start;result['inputs_sha256']=sha(out/'inputs.json')
  (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
