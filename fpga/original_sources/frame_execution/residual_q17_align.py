"""Exact INT8 residual alignment through existing Q17 hardware, not nearest-ratio approximation.
For0<r<=1 the target function has finitely many half-away rounding breakpoints.
Intersect all128 magnitude constraints, then validate every signed byte. Caller
must explicitly supply common=max; unsupported parameters are rejected.
"""
import argparse,json,sys
from fractions import Fraction
from pathlib import Path
from run_units import HERE,ROOT,sha
from residual_tables import encode,fixtures
from contract_reference import residual_value,requantize
from model_contract import ContractError
SCALE=1<<17
def multiplier_for_ratio(r):
 r=Fraction(r)
 if not 0<r<=1:raise ContractError('alignment ratio must be in (0,1]')
 lower,upper=1,SCALE-1
 for k in range(1,129):
  target=(2*k*r.numerator+r.denominator)//(2*r.denominator)
  lower=max(lower,((2*target-1)*(SCALE//2)+k-1)//k)
  upper=min(upper,((2*target+1)*(SCALE//2)-1)//k)
 if lower>upper:raise ContractError('no exact signed18 Q17 encoding')
 nearest=(2*r.numerator*SCALE+r.denominator)//(2*r.denominator)
 m=min(upper,max(lower,nearest))
 for x in range(-128,128):
  target=(2*abs(x)*r.numerator+r.denominator)//(2*r.denominator)
  if x<0:target=-target
  actual=(abs(x)*m+(SCALE//2))//SCALE
  if x<0:actual=-actual
  assert actual==target
 return dict(multiplier=m,valid_interval=[lower,upper],ratio=[r.numerator,r.denominator])
def encode_alignment(parameters):
 table=encode(parameters)
 a=Fraction(*parameters['main_scale']);b=Fraction(*parameters['skip_scale']);common=max(a,b)
 code=multiplier_for_ratio(min(a,b)/common)
 for x in range(-128,128):assert requantize(x,0,code['multiplier'])[0]==table['table'][x&255]
 return dict(**code,main_scaled=table['main_scaled'],post_multiplier=table['post_multiplier'],parameters=parameters)
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 a.root.mkdir(parents=True,exist_ok=False)
 # Every open interval and every breakpoint in0..1. This exhausts the target
 # lookup functions on signed INT8, including exact ties and the -128 endpoint.
 edges=sorted({Fraction(2*n+1,2*k) for k in range(1,129) for n in range(k)})
 samples=[Fraction(1)];previous=Fraction(0)
 for edge in edges:
  samples += [(previous+edge)/2,edge];previous=edge
 samples.append((previous+1)/2)
 min_width=SCALE;hist={};values=set()
 for r in samples:
  code=multiplier_for_ratio(r);lo,hi=code['valid_interval'];min_width=min(min_width,hi-lo+1)
  values.add(tuple(code['valid_interval']))
 records=[encode_alignment(p) for p in fixtures()];pairs=0
 for record in records:
  for x in range(-128,128):
   for y in range(-128,128):
    if record['main_scaled']:aa,bb=requantize(x,0,record['multiplier'])[0],y
    else:aa,bb=x,requantize(y,0,record['multiplier'])[0]
    assert requantize(aa+bb,0,record['post_multiplier'])[0]==residual_value(x,y,record['parameters'])
    pairs+=1
 rejected=0
 for r in (Fraction(0),Fraction(-1),Fraction(10001,10000)):
  try:multiplier_for_ratio(r)
  except ContractError:rejected+=1
 assert rejected==3
 result=dict(status='EXACT_INT8_ALIGNMENT_Q17_FUNCTION_EXHAUSTION_PASS',breakpoints=len(edges),sampled_intervals_and_ties=len(samples),
  signed_byte_checks=len(samples)*256,minimum_valid_integer_count=min_width,distinct_target_functions=len(values),
  residual_pair_checks=pairs,rejected=rejected,records=records,
  implementation='EXISTING_Q17_ALIGNMENT_PASS_THEN_EXISTING_N9_ADD_PASS',extra_mac_terms_per_oc8_job=8,
  hypothetical_frame_extra_cycles=172800,hypothetical_frame_extra_ms_at_150mhz=1.152,
  removes_lookup_ram_candidate=16,extra_alignment_read_write_and_reload_not_counted=True,
  arithmetic_exact=True,hardware_integrated=False,production_parameters=False,contract_modified=False,full_frame_30fps=False,
  evidence={str(f):sha(f) for f in (Path(__file__),HERE/'residual_tables.py',ROOT/'network_ab/contract_reference.py',ROOT/'network_ab/model_b_contract.json')})
 (a.root/'result.json').write_text(json.dumps(result,indent=2))
 print(json.dumps({k:result[k] for k in ('status','breakpoints','signed_byte_checks','minimum_valid_integer_count','residual_pair_checks')}))
if __name__=='__main__':main()
