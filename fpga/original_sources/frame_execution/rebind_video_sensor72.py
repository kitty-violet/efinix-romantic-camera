"""Bind unchanged Gray constraints to the new protected hierarchy, in a copy."""
import argparse,json,re,shutil
from pathlib import Path
from run_units import ROOT,sha
from video_sensor72_physical import REFERENCE
from review_input_isp_geometry import locations

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 base=a.base.resolve();root=a.root.resolve();assert base.is_relative_to(ROOT/'tmp') and root.is_relative_to(ROOT/'tmp') and not root.exists()
 result=json.loads((base/'result.json').read_text());assert result['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE' and all(s['exit_code']==0 for s in result['stages'])
 assert not (base/'outflow/ti60f225_oob.troutingtraces').exists()
 meta=json.loads((base/'physical_inputs.json').read_text())
 for row in meta['inputs']:assert sha(row['snapshot'])==row['sha256']
 oldloc=locations(REFERENCE/'outflow/ti60f225_oob.out.placeloc');newloc=locations(base/'outflow/ti60f225_oob.out.placeloc')
 original=(base/'ti60f225_oob.sdc').read_text();assert original==(REFERENCE/'ti60f225_oob.sdc').read_text()
 mapping={};bus_count=0
 pattern=r'(?:protected_inst/)*u_ddr3_top/controller/[^ {}]+\|(?:CLK|D)'
 def pin(m):
  old=m[0];name,port=old.rsplit('|',1);assert name in oldloc and name.endswith('~FF')
  stem=re.sub(r'^(?:protected_inst/)+','',name)
  matches=[n for n in newloc if re.sub(r'^(?:protected_inst/)+','',n)==stem]
  assert len(matches)==1,(old,matches)
  new=matches[0]+'|'+port;mapping[old]=new;return new
 lines=[]
 for line in original.splitlines(keepends=True):
  if line.startswith('set_bus_skew '):
   assert line.rstrip().endswith(' 2.500');bus_count+=1;line=re.sub(pattern,pin,line)
  lines.append(line)
 changed=''.join(lines);assert bus_count==10 and len(mapping)==140 and len(set(mapping.values()))==140
 assert len({n for n in mapping.values() if n.endswith('|D')})==70
 reverse={v:k for k,v in mapping.items()};assert re.sub(pattern,lambda m:reverse[m[0]],changed)==original
 assert sum(k!=v for k,v in mapping.items())==126
 assert all(k==v or v=='protected_inst/'+k for k,v in mapping.items())
 geometry=['outflow/ti60f225_oob'+s for s in ('.map.v','.vdb','.place','.out.placeloc')]+['work_pnr/ti60f225_oob.net_proto','fixed_dsp.placeloc','ti60f225_oob.xml','ti60f225_oob.peri.xml','efx_pnr_settings.ini']
 frozen={n:sha(base/n) for n in geometry};shutil.copytree(base,root)
 (root/'ti60f225_oob.sdc').write_text(changed,encoding='utf8')
 record=dict(status='EXACT_GRAY_ENDPOINT_NAME_REBINDING_NOT_TIMING_PASS',source=str(base),pins=mapping,buses=10,destination_bits=70,
  geometry=frozen,original={str(base/n):sha(base/n) for n in ('physical_inputs.json','result.json','ti60f225_oob.sdc','outflow/ti60f225_oob.place.out')},
  script_sha256=sha(__file__),timing_values_unchanged=True,clock_relationships_unchanged=True,placement_rerun=False,
  rationale='Full native reimport adds one protected hierarchy level. Only exact FF endpoint spellings change; existing 2.500ns Gray constraints preserved. Earlier placement estimate is historical; actual route/STA/nonempty Gray checks remain mandatory.',
  hardware_access=False,board_release=False)
 (root/'endpoint_rebinding.json').write_text(json.dumps(record,indent=2),encoding='utf8')
 for row in meta['inputs']:
  row['snapshot']=str(root/Path(row['snapshot']).relative_to(base));row['sha256']=sha(row['snapshot'])
 meta['placement_origin']=str(base);meta['endpoint_rebinding_sha256']=sha(root/'endpoint_rebinding.json')
 meta['clock_profile']['before_endpoint_rebinding_sha256']=meta['clock_profile']['after_sha256'];meta['clock_profile']['after_sha256']=sha(root/'ti60f225_oob.sdc')
 (root/'physical_inputs.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
 profile=json.loads((root/'service_clock_profile.json').read_text());profile['before_endpoint_rebinding_sha256']=profile['after_sha256'];profile['after_sha256']=sha(root/'ti60f225_oob.sdc');profile['endpoint_rebinding_sha256']=sha(root/'endpoint_rebinding.json')
 (root/'service_clock_profile.json').write_text(json.dumps(profile,indent=2),encoding='utf8')
 for n,h in frozen.items():assert sha(root/n)==h
 print(json.dumps(dict(status=record['status'],changed_pins=126,unchanged_pins=14,root=str(root))),flush=True)

if __name__=='__main__':main()
