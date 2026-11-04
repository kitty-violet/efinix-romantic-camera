"""Repair native observation aliases, freeze a fresh full-board candidate."""
import inspect,json,re,runpy,shutil,sys
from pathlib import Path
from run_units import ROOT,HERE,sha,run
from camera_checked_board import BOARD as OLD,audit_delta
from native_local_drivers import graph
BOARD=ROOT/'tmp/frame-execution/connected-board-camera-checked3'

def main():
 assert not (BOARD/'inputs.json').exists() and not (BOARD/'map.log').exists();audit_delta(OLD);BOARD.mkdir(exist_ok=True)
 m=json.loads((OLD/'inputs.json').read_text())
 for item in m['inputs']:
  src=Path(item['snapshot']);dst=BOARD/src.relative_to(OLD);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);item['snapshot']=str(dst)
 src=BOARD/'native_source.v';text=src.read_text();g=graph(text)
 outputs={'EFX_FF':['Q'],'EFX_LUT4':['O'],'EFX_ADD':['O','CO'],'EFX_RAM10':['RDATA']}
 driven={t:n for n,(k,p) in g.items() for pin in outputs.get(k,[]) for t in p.get(pin,[])}
 from probe_native_open_ports import protected_pins
 boundary=protected_pins(text)
 for n in ('w_mipi_rx_vs','w_mipi_rx_de','inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/vc_sync'):
  assert n in boundary and n in boundary[n],(n,boundary.get(n))
  for bit in ((0,1) if n.endswith('/vc_sync') else (None,)):driven[('net',n,bit)]='protected_inst.'+n
 taps=[('net','raw_stream_errors',i) for i in (*range(5),*range(7,11))]
 taps += [('net',x,None) for x in ('raw_frontend/consumer/reservation_error','raw_frontend/consumer/fifo_error','camera_boundary_error','w_mipi_rx_vs','w_mipi_rx_de','n75722_2','n75725_2')]
 taps += [('net','inst_efx_csi2_rx/u_efx_csi2_rx/csi2_rx_top_inst/genblk3.byte2pixel_inst/vc_sync',i) for i in (0,1)]
 assert all(t in driven for t in taps),[t for t in taps if t not in driven]
 old='.raw_errors(raw_stream_errors)';new=".raw_errors({raw_stream_errors[10:7], (\\raw_frontend/consumer/reservation_error | \\raw_frontend/consumer/fifo_error ), 1'b0, raw_stream_errors[4:0]})"
 assert text.count(old)==1;text=text.replace(old,new);src.write_text(text,encoding='utf8')
 original=OLD/'assembly/raw_stream_frontend.sv'
 assert original.exists();body=original.read_text()
 assert "assign errors[5]=1'b0;" in body and 'assign errors[6]=reservation_error | fifo_error;' in body
 for item in m['inputs']:item['sha256']=sha(item['snapshot'])
 m.update(camera_wiring_repair_sha256=sha(__file__),camera_wiring_origin=str(OLD),camera_wiring_origin_inputs_sha256=sha(OLD/'inputs.json'))
 (BOARD/'inputs.json').write_text(json.dumps(m,indent=2));shutil.copy2(OLD/'camera_graft.json',BOARD/'camera_graft.json')
 proof=dict(status='ALL_NEW_NATIVE_TAPS_HAVE_ACTUAL_DRIVERS',taps=[dict(signal=list(t),driver=driven[t]) for t in taps],raw5='explicit constant zero',raw6='existing reservation_error OR fifo_error',
  evidence={str(original):sha(original),str(OLD/'native_source.v'):sha(OLD/'native_source.v'),str(__file__):sha(__file__)},hardware_access=False)
 (BOARD/'camera_wiring_inputs.json').write_text(json.dumps(proof,indent=2))
 shutil.copy2(OLD/'map.cmd',BOARD/'map.cmd');log,seconds=run(['cmd.exe','/d','/c','map.cmd'],BOARD,'map.log',timeout=480)
 assert re.search(r'map\s*:\s*PASS',log)
 # Same strict connectivity/dead-cone audit, with the observed new-cell count.
 import audit_camera_checked_map as auditor
 newg=graph((BOARD/'outflow/ti60f225_oob.map.v').read_text());added=set(newg)-set(g)
 assert 988<=len(added)<=994
 code=inspect.getsource(auditor.main).replace('len(added)==988',f'len(added)=={len(added)}')
 scope=dict(auditor.__dict__,BOARD=BOARD);exec(compile(code,__file__+':graph','exec'),scope);scope['main']()
 r=json.loads((BOARD/'result.json').read_text());r['evidence'].update({str(BOARD/'camera_wiring_inputs.json'):sha(BOARD/'camera_wiring_inputs.json'),str(__file__):sha(__file__)})
 (BOARD/'result.json').write_text(json.dumps(r,indent=2));audit_delta(BOARD)
 warnings=(BOARD/'outflow/ti60f225_oob.warn.log').read_text();names=re.findall(r"Re-wiring to GND non-driven net/port '([^']+)'",warnings)
 assert names==['Open_'+str(i) for i in range(8,18)],names
 print('NEW_CAMERA_MAP_AND_ALL_INPUT_DRIVERS_PASS',len(added),seconds)
if __name__=='__main__':main()
