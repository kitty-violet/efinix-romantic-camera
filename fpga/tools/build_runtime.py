"""Rebuild the complete selected r15 runtime image from package-local sources."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'network'))
from coefficients import convert
import style_catalog as catalog
from scheduler.pack_program import build
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=ROOT/'build/runtime');a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 streams=build(ROOT/'network/programs')
 for n,b in streams.items():
  assert b==(ROOT/'network/programs'/f'{n}.bin').read_bytes();(a.output/f'{n}.bin').write_bytes(b)
 original=(ROOT/'network/runtime_data/flash-user-data.bin').read_bytes();header=catalog.validate(original)
 isp=original[header[10]-catalog.CATALOG_BASE:header[10]-catalog.CATALOG_BASE+768]
 rows=json.loads((ROOT/'network/runtime_data/manifest.json').read_text())['styles'];styles=[]
 for i,r in enumerate(rows):
  folder=ROOT/'network/models'/f'slot{i}'
  data=convert(folder)[0] if i<5 else (folder/'resident-padded.bin').read_bytes()
  assert data==(folder/'resident-padded.bin').read_bytes();(a.output/f'slot{i}-resident.bin').write_bytes(data)
  styles.append(dict(name=r['name'],tag=i+1,compute=streams['compute'],coefficients=data))
 raw,_=catalog.build(styles,streams['dma'],isp);assert raw==original
 (a.output/'flash-user-data.bin').write_bytes(raw)
 print(json.dumps({'status':'R15_RUNTIME_REBUILD_MATCH','bytes':len(raw),'slots':len(styles)}))
if __name__=='__main__':main()
