"""Exact native warning sites; reject any unreviewed message/source change."""
import argparse,json,re
from pathlib import Path
from run_units import ROOT,EF,sha
def main():
 p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 m=json.loads((a.native/'inputs.json').read_text());result=json.loads((a.native/'result.json').read_text())
 assert sha(a.native/'inputs.json')==result['inputs_sha256']
 rows={Path(r['snapshot']).name:r for r in m['inputs']}
 for r in rows.values():assert sha(r['snapshot'])==r['sha256']
 body=(a.native/'compile.log').read_text()
 allowed={
  ('compute_column_reader.sv',83):'Unsigned13 next_index is zero-extended to14 before a right shift. All8192 bit patterns have identical integer quotient.',
  ('tb_residual_column.sv',53):'Testbench IF on5-bit request is the intended any-request check; nonzero iff at least one bit is set.',
  ('efx_dsp24.v',280):'Official signed8/16 product extension to24.',
  ('efx_dsp24.v',327):'Official signed24 extension into25-bit add operands.',
  ('efx_dsp24.v',328):'Official signed24 extension into25-bit add operands.',
  ('efx_dsp24.v',331):'Official unsigned24 zero extension into25-bit add operands.',
  ('efx_dsp24.v',332):'Official unsigned24 zero extension into25-bit add operands.',
  ('efx_dsp24.v',95):'Official Verilog packed string comparison, left zero padding.',
  ('efx_dsp24.v',109):'Official Verilog packed string comparison, left zero padding.',
  ('efx_dsp24.v',189):'Official Verilog packed string comparison, left zero padding.',
  ('efx_dsp24.v',190):'Official Verilog packed string comparison, left zero padding.',
  ('efx_dsp24.v',193):'Official Verilog packed string comparison, left zero padding.',
  ('efx_dsp24.v',215):'Official Verilog packed string comparison, left zero padding.',
  ('efx_ram10.v',199):'Official READ_FIRST/WRITE_FIRST packed strings, left zero padding.',
  ('efx_ram10.v',211):'Official WRITE_MODE packed strings, left zero padding.',
  ('efx_ram10.v',392):'Official WRITE_MODE packed strings, left zero padding.',
  ('efx_dpram10.v',255):'Official WRITE_MODE_A packed strings, left zero padding.',
  ('efx_dpram10.v',259):'Official WRITE_MODE_B packed strings, left zero padding.',
  ('efx_dpram10.v',411):'Official WRITE_MODE_A packed strings, left zero padding.',
  ('efx_dpram10.v',505):'Official WRITE_MODE_B packed strings, left zero padding.',
  ('shared_rq_oc8_cluster.sv',64):'Unsigned2 group0..3 compared to integer header index0..3.',
  ('finish_rr_arbiter.sv',41):'Unsigned3 winner0..7 compared to nonnegative owner index.',
  ('shared_pool_banked_fifo.sv',63):'Unsigned3 selected0..7 compared to loop index0..7.',
  ('compute_column_cluster.sv',57):'Pixels1..320 plus7 fits10bits; unsigned13 extension before >>3 is exact.',
  ('compute_column_cluster.sv',58):'Pixels1..320 minus1 is0..319; low3 bits then13-bit count.',
  ('compute_column_cluster.sv',59):'Pixels1..320 minus1 is0..319; low3 bits then13-bit count.',
  ('tb_compute_column_cluster.sv',123):'Test expected-error argument is always literal0 or1; boolean NOT.',
  ('tb_compute_column_cluster.sv',127):'Test expected-error argument is always literal0 or1; boolean NOT.'}
 sites=[];unknown=[]
 for line in body.splitlines():
  if not line.startswith('%Warning'):continue
  x=re.match(r'%Warning-([A-Z0-9_]+): (.*[\\/])([^\\/]+):(\d+):(\d+): (.*)',line);assert x,line
  code,_,name,ln,col,msg=x.groups();ln=int(ln);key=(name,ln)
  if key not in allowed or code!=('WIDTHTRUNC' if name.startswith('tb_') else 'WIDTHEXPAND'):unknown.append(line);continue
  f=Path(rows[name]['snapshot']);source_line=f.read_text().splitlines()[ln-1]
  sites.append(dict(file=name,line=ln,column=int(col),code=code,message=msg,source_line=source_line,source_sha256=sha(f),reason=allowed[key]))
 for name in ('efx_dsp24.v','efx_ram10.v','efx_dpram10.v'):assert rows[name]['sha256']==sha(EF/'sim_models/verilog'/name)
 if 'tb_compute_column_cluster.sv' in rows:
  tb=Path(rows['tb_compute_column_cluster.sv']['snapshot']).read_text()
  assert set(re.findall(r'finish_group\(([^)]+)\)',tb))=={'input integer bad','0','1'}
  expected=30
 else:
  assert 'if(weight_req)fail("residual issued coefficient read");' in Path(rows['tb_residual_column.sv']['snapshot']).read_text()
  expected=29
 for idx in range(8192):
  for lg in (0,1,2):assert (idx>>(lg+2))==((idx&16383)>>(lg+2))
 for px in range(1,321):
  assert ((px+7)>>3)==(px+7)//8 and ((px-1)&7)==(px-1)%8
 assert len(sites)==expected and not unknown,(len(sites),unknown)
 out=dict(status='EXACT_SOURCE_NATIVE_WARNING_AUDIT_PASS',native_result_sha256=sha(a.native/'result.json'),
  compile_log_sha256=sha(a.native/'compile.log'),sites=sites,unknown=unknown,official_models_unmodified=True,
  full_frame=False,pnr=False)
 a.output.write_text(json.dumps(out,indent=2));print(json.dumps(dict(status=out['status'],sites=len(sites),unknown=len(unknown))),flush=True)
if __name__=='__main__':main()
