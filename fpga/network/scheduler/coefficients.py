"""Lossless S4R1PKG1 -> existing five-bank resident transport layout.

The resulting ENGINE image is NOT a legacy B1 model package. Its coefficient
regions retain the physical reader ABI, while S4R1 descriptors define the new
topology. No RTL file parser, quantization, training or colour filter is added.
"""
from pathlib import Path
import hashlib,importlib.util,json,struct,zlib
R=Path(__file__).resolve().parent
INPUTS=json.loads((R/'inputs.json').read_text('utf8'))
PKG=(R/INPUTS['package']).resolve()
for item in INPUTS['selected']:item['folder']=str((R/item['folder']).resolve())
spec=importlib.util.spec_from_file_location('received_verifier',PKG/'verify_package.py')
oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
def sha(data):return hashlib.sha256(data).hexdigest()
def align(n,a):return (n+a-1)//a*a
def convert(folder):
 graph,parameters=oracle.load(folder)
 raw=(folder/'coefficients-padded.bin').read_bytes()
 image=bytearray(20480);image[:8]=b'S4R1ENG1'
 struct.pack_into('<6I',image,8,1,17792,11,1,1280,720)
 image[32:64]=hashlib.sha256(raw).digest()
 cursors=dict(weight=1792,bias=12608,multiplier=15040,relu=17472)
 limits=dict(weight=12608,bias=15040,multiplier=17472,relu=17600)
 layout=[];verified=0
 for node in graph['nodes']:
  if node['id'] not in parameters:continue
  ps=parameters[node['id']];ocount=node['channels'];icount=node['cin'];k=node['kernel']
  dw=node['kind']=='dw3x3';padded=align(ocount,8)
  record=dict(node=node['id'],kind=node['kind'],stride=node['stride'],input_channels=icount,output_channels=ocount,padded_channels=padded)
  for field in cursors:
   # Header RAM needs group-aligned32B; preserve previous64B alignment.
   start=align(cursors[field],64) if field!='relu' else cursors[field]
   payload=bytearray()
   if field=='weight':
    for group in range(padded//8):
     for ibase in range(0,1 if dw else icount,8):
      for ky in range(k):
       for kx in range(k):
        for ic in range(ibase,min(ibase+8,1 if dw else icount)):
         for lane in range(8):
          oc=group*8+lane
          idx=((oc*(1 if dw else icount)+ic)*k+ky)*k+kx
          payload.append((ps['weight'][idx] if oc<ocount else 0)&255)
   elif field=='relu':
    for group in range(padded//8):payload.append((1<<min(8,ocount-group*8))-1 if node['activation']=='relu' else 0)
   else:
    for oc in range(padded):payload.extend(struct.pack('<i',ps[field][oc] if oc<ocount else 0))
   end=start+len(payload);assert end<=limits[field],(node['id'],field,end)
   image[start:end]=payload;record[field]=dict(offset=start,length=len(payload));cursors[field]=end
  layout.append(record)
  # Decode by independently derived OC/IC/tap address, not the encoder loop.
  wb=record['weight']['offset']
  for oc in range(ocount):
   for ic in range(1 if dw else icount):
    for ky in range(k):
     for kx in range(k):
      if dw:word=(oc//8)*k*k+ky*k+kx
      elif icount==3:word=(ky*k+kx)*3+ic
      else:word=(oc//8)*icount*k*k+(ic//8)*k*k*8+(ky*k+kx)*8+ic%8
      got=struct.unpack_from('<b',image,wb+word*8+oc%8)[0]
      assert got==ps['weight'][((oc*(1 if dw else icount)+ic)*k+ky)*k+kx]
      verified+=1
   for field in ('bias','multiplier'):
    assert struct.unpack_from('<i',image,record[field]['offset']+oc*4)[0]==ps[field][oc];verified+=1
   assert bool(image[record['relu']['offset']+oc//8]&(1<<(oc%8)))==(node['activation']=='relu')
  # All padded channels are zero, not uninitialized state.
  for oc in range(ocount,padded):
   assert all(image[wb+word*8+oc%8]==0 for word in range((oc//8)*(1 if dw else icount)*k*k,(oc//8+1)*(1 if dw else icount)*k*k))
   assert all(struct.unpack_from('<i',image,record[f]['offset']+oc*4)[0]==0 for f in ('bias','multiplier'))
 for i,name in enumerate(('res1_add',)):
  p=graph['residual'][name];fields=(int(p['main_scaled']),p['multiplier'],p['post_multiplier'])
  assert fields[0] in (0,1) and all(-(1<<17)<=v<(1<<17) for v in fields[1:])
  struct.pack_into('<3i',image,17600+i*64,*fields)
  assert struct.unpack_from('<3i',image,17600+i*64)==fields
 assert not any(image[17792:])
 return bytes(image),dict(schema='S4R1ENG1',source_sha256=sha(raw),resident_sha256=sha(image),
  resident_crc32=f'{zlib.crc32(image):08x}',transport_image_bytes=17792,transport_padded_bytes=20480,
  graph=graph,operators=layout,residual=graph['residual'],region_used_end=cursors,
  independent_weight_bias_multiplier_checks=verified,legacy_B1_package=False)

def main():
 out=R/'data/resident1';out.mkdir(parents=True,exist_ok=False)
 rows=[]
 for style in INPUTS['selected']:
  folder=Path(style['folder']);raw,record=convert(folder);d=out/f"slot{style['slot']}";d.mkdir()
  (d/'resident-padded.bin').write_bytes(raw)
  (d/'resident-layout.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),'utf8')
  (d/'load_words.hex').write_text(''.join(f'{int.from_bytes(raw[i:i+8],"little"):016x}\n'for i in range(0,len(raw),8)),'ascii')
  rows.append(dict(slot=style['slot'],id=style['id'],name=style['name'],sha256=record['resident_sha256'],checks=record['independent_weight_bias_multiplier_checks']))
 result=dict(status='LOSSLESS_RESIDENT_CONVERSION_PASS_NOT_RTL_RELEASE',styles=rows,hardware_access=False)
 (out/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),'utf8')
 print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
