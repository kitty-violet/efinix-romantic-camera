"""Exact residual table encoding for explicit unequal runtime scales.
Keeps the shared reference math; supports common=max(main,skip), rejects other
common scales instead of silently changing them. No trained parameters supplied.
"""
import argparse,json,random,sys,zlib
from fractions import Fraction
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'network_ab'))
from contract_reference import positive_scale,q17_multiplier,residual_value,requantize
from model_contract import ContractError
def encode(p):
 if set(p)!={'main_scale','skip_scale','common_scale','output_scale'}:raise ContractError('four explicit scales')
 a,b,c,o=[positive_scale(p[k]) for k in ('main_scale','skip_scale','common_scale','output_scale')]
 if c!=max(a,b):raise ContractError('lookup encoding requires supplied common=max(main,skip)')
 scaled_main=a<=b;ratio=min(a,b)/c;table=[]
 for byte in range(256):
  x=byte if byte<128 else byte-256
  n=abs(x)*ratio.numerator;v=(2*n+ratio.denominator)//(2*ratio.denominator)
  if x<0:v=-v
  assert -128<=v<=127
  table.append(v)
 return dict(main_scaled=scaled_main,table=table,post_multiplier=q17_multiplier(c,Fraction(1),o),parameters=p)
def fixtures():
 return [dict(main_scale=[3,7],skip_scale=[1,1],common_scale=[1,1],output_scale=[2,1]),
         dict(main_scale=[1,1],skip_scale=[5,9],common_scale=[1,1],output_scale=[2,1]),
         dict(main_scale=[11,128],skip_scale=[11,128],common_scale=[11,128],output_scale=[11,64])]
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();a.root.mkdir(parents=True,exist_ok=False)
 records=[encode(x) for x in fixtures()];checks=0
 for r in records:
  for x in range(-128,128):
   for y in range(-128,128):
    aa=r['table'][x&255] if r['main_scaled'] else x
    bb=y if r['main_scaled'] else r['table'][y&255]
    assert requantize(aa+bb,0,r['post_multiplier'])[0]==residual_value(x,y,r['parameters'])
    checks+=1
 rng=random.Random(71)
 for _ in range(256):
  x,y=Fraction(rng.randrange(1,100000),rng.randrange(1,100000)),Fraction(rng.randrange(1,100000),rng.randrange(1,100000))
  c=max(x,y);o=2*c
  p={k:[v.numerator,v.denominator] for k,v in zip(('main_scale','skip_scale','common_scale','output_scale'),(x,y,c,o))}
  r=encode(p)
  for xx in range(-128,128):
   yy=rng.randrange(-128,128);aa=r['table'][xx&255] if r['main_scaled'] else xx;bb=yy if r['main_scaled'] else r['table'][yy&255]
   assert requantize(aa+bb,0,r['post_multiplier'])[0]==residual_value(xx,yy,p);checks+=1
 rejected=0
 for wrong in ({},dict(fixtures()[0],common_scale=[2,1]),dict(fixtures()[0],main_scale=[0,1]),dict(fixtures()[0],output_scale=[1,4])):
  try:encode(wrong)
  except ContractError:rejected+=1
 assert rejected==4
 raw=bytes(v&255 for r in records for v in r['table']);crc=zlib.crc32(raw)
 (a.root/'table.hex').write_text(''.join(f'{x:02x}\n' for x in raw))
 (a.root/'crc.hex').write_text(f'{crc:08x}\n')
 result=dict(status='EXACT_UNEQUAL_SCALE_TABLE_REFERENCE_PASS',checks=checks,rejected=rejected,crc32=f'{crc:08x}',
  mode_bits=5,records=records,common_policy='EXPLICIT_MAX_REQUIRED',production_parameters=False,contract_modified=False)
 (a.root/'result.json').write_text(json.dumps(result,indent=2));print(dict(status=result['status'],checks=checks,crc32=result['crc32']))
if __name__=='__main__':main()
