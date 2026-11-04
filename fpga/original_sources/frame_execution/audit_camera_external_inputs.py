"""All mapped new input sources and exact RAW6 OR / RAW5 zero check."""
import itertools,json
from run_units import HERE,sha
from camera_checked_v2 import BOARD
from camera_checked_board import audit_delta
from native_local_drivers import graph
from probe_full_native_split_map import params

def main():
 audit_delta(BOARD);text=(BOARD/'outflow/ti60f225_oob.map.v').read_text();g=graph(text);ps=params(text)
 r=json.loads((BOARD/'result.json').read_text());new=set(r['new_cells'])
 outputs={'EFX_FF':['Q'],'EFX_LUT4':['O'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA'],'EFX_SRL8':['Q','Q7']}
 drivers={t:n for n,(k,p) in g.items() for pin in outputs.get(k,[]) for t in p.get(pin,[])}
 ports={'CLK_5M','process_clk_150','i_mipi_rx_pclk','io_cam_scl_IN','io_cam_sda_IN'}
 protected={'w_mipi_rx_vs','w_mipi_rx_de','inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/vc_sync'}
 used={t for n in new for pin,ts in g[n][1].items() if pin not in outputs[g[n][0]] for t in ts if t[0]!='literal'}
 missing=[t for t in used if t not in drivers and t[1] not in ports|protected];assert not missing,missing
 lp=g['camera_observer/status_meta[9]~FF'][1]
 source=[g[f'raw_frontend/consumer/{n}~FF'][1]['Q'][0] for n in ('reservation_error','fifo_error')]
 def ev(t,values):
  if t in values:return values[t]
  if t[0]=='literal':assert t[1] in ("1'b0","1'b1");return int(t[1][-1])
  n=drivers[t];k,p=g[n];assert k=='EFX_LUT4',(n,k)
  index=sum(ev(p.get('I'+str(i),[('literal',"1'b0",None)])[0],values)<<i for i in range(4))
  return ps[n]['LUTMASK'][1]>>index&1
 for v in itertools.product((0,1),repeat=2):assert ev(lp['D'][0],dict(zip(source,v)))==int(any(v))
 assert g['camera_observer/status_meta[8]~FF'][1]['D']==[('literal',"1'b0",None)]
 evidence={str(f):sha(f) for f in (BOARD/'outflow/ti60f225_oob.map.v',BOARD/'camera_wiring_inputs.json',HERE/'camera_checked_v2.py',HERE/'audit_camera_external_inputs.py')}
 out=BOARD/'camera_external_input_audit.json';assert not out.exists()
 out.write_text(json.dumps(dict(status='ALL_CAMERA_MAPPED_INPUTS_DRIVEN_RAW6_OR_RAW5_ZERO_PASS',new_cells=len(new),input_tokens=len(used),raw6_checks=4,evidence=evidence,hardware_access=False),indent=2));print('ALL_CAMERA_MAPPED_INPUTS_DRIVEN_RAW6_OR_RAW5_ZERO_PASS')
if __name__=='__main__':main()
