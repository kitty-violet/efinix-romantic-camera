"""Bind observed boot failure and area inventory to the frozen board sources."""
import json,re
from pathlib import Path
from run_units import ROOT,EF,sha

def main():
 board=ROOT/'tmp/frame-execution/connected-board-sharing1';out=ROOT/'tmp/frame-execution/board-global-audit1'
 assert not out.exists();out.mkdir()
 top=(board/'rtl/ti60f225_oob_top.v').read_text();cpu=(board/'assembly/EfxSapphireSoc.v').read_text()
 apb=(board/'assembly/cnn_apb_control.sv').read_text()
 files=[board/f'EfxSapphireSoc.v_toplevel_system_ramA_logic_ram_symbol{i}.bin' for i in range(4)]
 planes=[f.read_text().splitlines() for f in files]
 words=[sum(int(planes[k][i],2)<<(8*k) for k in range(4)) for i in range(1024)]
 lui_hits=[dict(pc=hex(0xf9000000+4*i),word=hex(w),address='0xf8014000') for i,w in enumerate(words) if w&0x7f==0x37 and w&0xfffff000==0xf8014000]
 assert lui_hits and "== 24'h006000" in cpu and "== 24'h014000" not in cpu
 assert all(f".system_spi_0_io_data_{i}_read(1'b1)" in top for i in range(4))
 assert re.search(r'enable\s*<=\s*0',apb)
 logs=[ROOT/f'tmp/frame-execution/board-boot-observe{i}/simulation.log' for i in (1,2)]
 assert 'cycles=1000000 pc=f9000c40 apb=0 enable=0' in logs[0].read_text()
 assert 'addr=f8014004 write=0' in logs[1].read_text()
 area=json.loads((board/'video_ideal_audit.json').read_text())
 csi=(board/'rtl/csi_rx_controller.sv').read_text();norm=lambda s:re.sub(r'\s+','',s)
 paired=[]
 for name in ('efx_csi2_rx_top','efx_csi2_rx'):
  f=EF/'ipm/ip/efx_csi2/source_efinity'/f'{name}.sv';g=EF/'ipm/ip/efx_csi2/source_modelsim'/f'{name}.sv'
  same=norm(f.read_text()) in norm(csi);assert same
  paired.append(dict(module=name,installed_synthesis_whitespace_normalized_exact_substring=True,synthesis_sha256=sha(f),official_simulation_model=str(g),simulation_sha256=sha(g)))
 result=dict(status='GLOBAL_MEASUREMENT_BLOCKED_BEFORE_VALID_FRAME',full_board_frame_ms=None,full_board_fps=None,
  boot=dict(rom_spi_base='0xf8014000',actual_rtl_spi_base='0xf8006000',observed_poll_address='0xf8014004',cycles_observed=1000000,
   enable=0,apb_transactions=0,spi_inputs_tied_high=True,firmware_fixed=False,rom_lui_hits=lui_hits,
   evidence=[dict(path=str(f),sha256=sha(f)) for f in [*files,*logs,board/'rtl/ti60f225_oob_top.v',board/'assembly/EfxSapphireSoc.v']]),
  area=area['resources'],top_level_area=[x for x in area['hierarchy'] if x['depth']==1],
  cold_video_release=dict(cycles=2**26,assumed_clock_hz=74250000,seconds=(2**26)/74250000,measured=False),
  csi_package_pairing=paired,
  scope='Frozen current implementation, missing gesture/exit/effect/true-FPS leafs remain outside implemented graph; not a complete baseline',
  production_rtl_modified=False)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ('status','full_board_frame_ms','full_board_fps','production_rtl_modified')}))

if __name__=='__main__':main()
