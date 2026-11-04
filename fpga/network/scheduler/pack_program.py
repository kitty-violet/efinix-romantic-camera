"""Rebuild the selected r15 compute/DMA streams from checked-in field records."""
from pathlib import Path
import json
def build(program_dir):
 p=Path(program_dir);schema=json.loads((p/'schema.json').read_text());rows=json.loads((p/'descriptor-fields.json').read_text());out={}
 for j,name in enumerate(('compute','dma')):
  size,bits,version=(64,8,0x55) if j==0 else (16,4,3);raw=bytearray()
  for values in rows[name]:
   word=offset=0
   for field,width in schema[j]:
    v=values[field];assert 0<=v<(1<<width);word|=v<<offset;offset+=width
   assert offset<=size*8-bits;word|=version<<(size*8-bits);raw.extend(word.to_bytes(size,'little'))
  out[name]=bytes(raw)
 return out
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 a.output.mkdir(parents=True,exist_ok=True)
 for n,b in build(Path(__file__).resolve().parents[1]/'programs').items():(a.output/(n+'.bin')).write_bytes(b)
 print('COMPUTE_DMA_REBUILT')
