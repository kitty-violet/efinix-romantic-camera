"""Same original DDR Gray gates; finite old probe replaced by camera snapshot."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha
def main():
 source=HERE/'audit_command_progress_gray.py';code=source.read_text()
 injections=[
  ('from snapshot_clock_cdc import inventory','from camera_checked_cdc import inventory'),
  ("here/'snapshot_clock_cdc.py'","here/'camera_checked_cdc.py'"),
  ("('system_to_jtag','process_clk_150','jtag_inst1_TCK'),", "('camera_cfg_to_system','CLK_5M','process_clk_150'),\n ('camera_pixel_to_system','i_mipi_rx_pclk','process_clk_150'),\n ('system_to_jtag','process_clk_150','jtag_inst1_TCK'),")]
 needle='code=source.read_text()';assert code.count(needle)==1
 code=code.replace(needle,needle+'\n'+''.join(' code=code.replace('+repr(a)+','+repr(b)+')\n' for a,b in injections))
 root=Path(sys.argv[sys.argv.index('--root')+1]).resolve();driver=root.parent/(root.name+'-camera-wrapper.py')
 assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),wrapper_sha256=sha(__file__),driver_sha256=sha(driver)),indent=2))
 sys.argv=[str(driver),*sys.argv[1:]];runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
