"""Actual full-map control audit plus preservation of new diagnostic cells."""
import json,re,sys
from pathlib import Path
from run_units import sha
from startup_cause_board import audit_delta
from audit_snapshot_cells import cells
from audit_local_reset_implementation import norm
def extra_map(board):
 net=board/'outflow/ti60f225_oob.map.v';body=net.read_text();cc=cells(body);params={}
 for n,k,v in re.findall(r'^\s*defparam\s+(\S+)\s*\.(\w+)\s*=\s*([^;]+);',body,re.M):params.setdefault(norm(n),{})[k]=v.strip()
 selected={};chains=[]
 for domain,clock in (('local_cells','process_clk_150'),('ddr_cells','sdram_clk')):
  prefix='cell_probe/'+domain+'/'
  for suffix in ('pair_under_test','negative_ce_ff','positive_ce_ff','feedback_ff','test_reset_register'):
   n=prefix+suffix;k,p,attrs=cc[n];selected[n]=dict(kind=k,pins=p,params=params[n])
   if suffix=='pair_under_test':
    m=re.fullmatch(r"16'([hb])([0-9a-fA-F]+)",params[n]['LUTMASK']);assert m
    assert k=='EFX_COMB4' and params[n]['MODE']=='"LOGIC"' and int(m[2],16 if m[1]=='h' else 2)==0x1144
    assert 'O' in p and 'P' in p
   else:
    assert k=='EFX_FF' and p['CLK']==clock and params[n]['SR_POLARITY']=="1'b0" and params[n]['SR_SYNC']=="1'b0"
    assert params[n]['CE_POLARITY']==("1'b0" if suffix=='negative_ce_ff' else "1'b1")
    assert any(p['Q']==pins.get(pin) for kind,pins,attr in cc.values() for pin in ('D','SR','I0','I1','I2','I3')),('unobserved Q',n)
  reset=selected[prefix+'test_reset_register'];assert reset['pins']['CE']=="1'b1"
  assert all(selected[prefix+suffix]['pins']['SR']==reset['pins']['Q'] for suffix in ('negative_ce_ff','positive_ce_ff','feedback_ff'))
  name=prefix+'release_pipe[0]~FF';k,p,attrs=cc[name]
  assert k=='EFX_FF' and p['CLK']==clock and 'async_reg=' in attrs
  tail=[n for n,(k2,p2,a2) in cc.items() if k2=='EFX_FF' and p2.get('D')==p['Q'] and p2.get('CLK')==clock and 'async_reg=' in a2]
  assert len(tail)==1,tail;chains.append(dict(first=name,second=tail[0],clock=clock))
 n='cell_probe/done_sync[0]~FF';names=[n]
 for stage in range(3):
  k,p,a=cc[n];assert k=='EFX_FF' and p['CLK']=='process_clk_150' and 'async_reg=' in a
  if stage<2:
   tail=[name for name,(k2,p2,a2) in cc.items() if k2=='EFX_FF' and p2.get('D')==p['Q'] and p2.get('CLK')=='process_clk_150' and 'async_reg=' in a2]
   assert len(tail)==1,tail;n=tail[0];names.append(n)
 return dict(status='OFP5_FULL_MAP_DIAGNOSTIC_CELLS_PRESERVED',selected=selected,reset_release_pairs=chains,done_sync_chain=names,
  map_sha256=sha(net),hardware_access=False,physical_cdc_pass=False)
def main():
 args=sys.argv[:]
 board=Path(args[args.index('--board')+1]).resolve();root=Path(args[args.index('--root')+1]).resolve()
 import audit_fault_probe_map as prior
 prior.audit_delta=audit_delta
 prior.main()
 r=extra_map(board);(root/'new_diagnostic_cells.json').write_text(json.dumps(r,indent=2));print(r['status'])
if __name__=='__main__':main()
