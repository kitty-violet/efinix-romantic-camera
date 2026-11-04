"""Probe PLL4's exact config/compute ratio; keep the unrelated CSI path frozen."""
import json, runpy, sys
from run_units import ROOT, HERE, sha
from control_command_repair import once

def main():
    source=HERE/'camera_related_clock_model.py';code=source.read_text()
    code=once(code,"(('i_mipi_rx_pclk','20.000','19.998'),('CLK_5M','200.000','199.980'))","(('CLK_5M','200.000','199.980'),)")
    code=once(code,'camera-related-clock-sta1','camera-related-clock-sta2')
    code=once(code,"('pixel','i_mipi_rx_pclk','process_clk_150'),",'')
    code=once(code,"for name in ('cfg','pixel','core'):","for name in ('cfg','core'):")
    driver=ROOT/'tmp/frame-execution/camera-clock-ratio-probe2.py'
    assert not driver.exists();compile(code,str(driver),'exec');driver.write_text(code)
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),hardware_access=False),indent=2))
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
