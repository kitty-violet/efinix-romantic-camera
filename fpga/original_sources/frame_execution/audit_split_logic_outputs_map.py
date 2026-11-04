"""Audit actual mapped independent outputs, retained FFs, and control logic."""
import json,re,sys
from pathlib import Path
from run_units import sha
from split_logic_outputs_board import BASE,audit_delta
from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm

def extra_map(board):
 net=board/'outflow/ti60f225_oob.map.v';body=net.read_text();cc=cells(body);params={}
 for n,k,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',body,re.M):
  params.setdefault(norm(n),{})[k]=v.strip()
 assert not any(k=='EFX_COMB4' for k,p,a in cc.values()),'COMB4 remains in actual map'
 selected={};chains=[];truth_checks=0
 for domain,clock,index_prefix in (('local_cells','process_clk_150','cell_diagnostic'),('ddr_cells','sdram_clk','cell_probe/ddr_probe')):
  prefix='cell_probe/'+domain+'/'
  for suffix in ('independent_o','independent_p','negative_ce_ff','positive_ce_ff','feedback_ff','test_reset_register'):
   n=prefix+suffix;k,p,attrs=cc[n];selected[n]=dict(kind=k,pins=p,params=params[n])
   if suffix.startswith('independent_'):
    assert k=='EFX_LUT4'
    m=re.fullmatch(r"16'([hb])([0-9a-fA-F]+)",params[n]['LUTMASK']);assert m
    mask=int(m[2],16 if m[1]=='h' else 2)
    for inputs in range(16):
     logical=inputs&3;pin_index=0
     for bit in range(4):
      value=p.get('I'+str(bit))
      if value==index_prefix+'[50]':v=logical&1
      elif value==index_prefix+'[51]':v=(logical>>1)&1
      elif value in ("1'b0","1'b1"):v=int(value[-1])
      else:
       assert value is None,(n,bit,value)
       v=(inputs>>bit)&1
      pin_index|=v<<bit
     expected=int(logical==(2 if suffix=='independent_o' else 0))
     assert (mask>>pin_index)&1==expected,(n,inputs,mask,p)
     truth_checks+=1
   else:
    assert k=='EFX_FF' and p['CLK']==clock
    assert params[n]['SR_POLARITY']=="1'b0" and params[n]['SR_SYNC']=="1'b0"
    assert params[n]['CE_POLARITY']==("1'b0" if suffix=='negative_ce_ff' else "1'b1")
    assert any(p['Q']==q.get(pin) for _,q,_ in cc.values() for pin in ('D','SR','I0','I1','I2','I3'))
  reset=selected[prefix+'test_reset_register'];assert reset['pins']['CE']=="1'b1"
  assert all(selected[prefix+s]['pins']['SR']==reset['pins']['Q'] for s in ('negative_ce_ff','positive_ce_ff','feedback_ff'))
  o=selected[prefix+'independent_o']['pins']['O'];p=selected[prefix+'independent_p']['pins']['O']
  assert o!=p
  # Both sets of eight independent capture FFs must survive optimization.
  outputs={}
  for signal in (o,p):
   sinks=[n for n,(k,q,a) in cc.items() if k=='EFX_FF' and q.get('D')==signal and q.get('CLK')==clock]
   assert len(sinks)==8,(signal,sinks)
   outputs[signal]=sinks
  selected[prefix+'capture_outputs']=outputs
  n=prefix+'release_pipe[0]~FF';k,p,a=cc[n]
  assert k=='EFX_FF' and p['CLK']==clock and 'async_reg=' in a
  tail=[n for n,(k2,p2,a2) in cc.items() if k2=='EFX_FF' and p2.get('D')==p['Q'] and p2.get('CLK')==clock and 'async_reg=' in a2]
  assert len(tail)==1;chains.append(dict(first=n,second=tail[0],clock=clock))
 n='cell_probe/done_sync[0]~FF';names=[n]
 for stage in range(3):
  k,p,a=cc[n];assert k=='EFX_FF' and p['CLK']=='process_clk_150' and 'async_reg=' in a
  if stage<2:
   tail=[name for name,(k2,p2,a2) in cc.items() if k2=='EFX_FF' and p2.get('D')==p['Q'] and p2.get('CLK')=='process_clk_150' and 'async_reg=' in a2]
   assert len(tail)==1;n=tail[0];names.append(n)
 old=cells((BASE/'outflow/ti60f225_oob.map.v').read_text())
 counts={};classified={}
 for n,(k,p,a) in old.items():
  if k!='EFX_COMB4':continue
  labels=sorted({v.lstrip('~').split('/')[0] for v in p.values() if '/' in v})
  label='+'.join(labels) if labels else 'flat_names_unclassified'
  counts[label]=counts.get(label,0)+1;classified[n]=label
 return dict(status='SPLIT_OUTPUTS_FULL_MAP_ZERO_COMB4_PASS',whole_map_comb4=0,
  selected=selected,truth_checks=truth_checks,reset_release_pairs=chains,done_sync_chain=names,
  parent_comb4_direct_net_name_classification=counts,parent_comb4_classification=classified,
  classification_limit='Direct named pins only; flat names are not guessed as physical ownership.',
  map_sha256=sha(net),auditor_sha256=sha(__file__),hardware_access=False,physical_cdc_pass=False,board_release=False)

def main():
 args=sys.argv[:];board=Path(args[args.index('--board')+1]).resolve();root=Path(args[args.index('--root')+1]).resolve()
 import audit_fault_probe_map as prior
 prior.audit_delta=audit_delta;prior.main()
 r=extra_map(board);(root/'new_diagnostic_cells.json').write_text(json.dumps(r,indent=2))
 from split_logic_outputs_cdc import map_structure
 (root/'held_data_structure.json').write_text(json.dumps(map_structure(board),indent=2))
 print(r['status'])

if __name__=='__main__':main()
