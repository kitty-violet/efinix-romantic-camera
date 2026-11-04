"""Independent external DDR initialized-byte and final-pixel coverage audit."""
import argparse,json
from pathlib import Path
from run_units import sha
def audit(root):
 m=json.loads((root/'frame_manifest.json').read_text());p=json.loads((root/'program.json').read_text())
 assert len(p)==m['commands']
 windows={'encoder':(0,33554432),'residual':(33554432,50331648),'decoder':(50331648,8388608)}
 shape={0:(720,1280,3),33554432:(180,320,32),50331648:(180,320,32),8388608:(720,1280,3)}
 valid={base:bytearray(h*w*c) for base,(h,w,c) in shape.items()};valid[0][:]=bytes([1])*len(valid[0])
 loads=stores=0;bytes_loaded=bytes_stored=0
 for q in p:
  if q['kind']=='compute':continue
  base=windows[q['group']][int(q['kind']=='store')];row=q['row'];h,w,c=shape[base]
  assert w==q['global_width'] and c==q['channels'] and 0<=row['y']<h
  off=(row['y']*w+row['origin'])*c;n=row['width']*c
  assert 0<=row['origin']<w and row['origin']+row['width']<=w and off+n<=len(valid[base])
  if q['kind']=='load':
   assert all(valid[base][off:off+n]),('read before actual prior store',q['command'],base,off,n)
   loads+=1;bytes_loaded+=n
  else:
   assert not any(valid[base][off:off+n]),('duplicate external commit',q['command'],base,off,n)
   valid[base][off:off+n]=bytes([1])*n;stores+=1;bytes_stored+=n
 assert all(all(v) for v in valid.values())
 return dict(status='FINITE_DDR_DEPENDENCY_COVERAGE_PASS',commands=len(p),loads=loads,stores=stores,bytes_loaded=bytes_loaded,bytes_stored=bytes_stored,
  external_intermediates_only_from_prior_DUT_stores=True,final_unique_pixels=921600,rtl_measurement=False,
  vector_manifest_sha256=sha(root/'frame_manifest.json'),program_sha256=sha(root/'program.json'))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--vectors',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert not a.output.exists();r=audit(a.vectors);a.output.write_text(json.dumps(r,indent=2));print(json.dumps(r))

