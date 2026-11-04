"""Reuse the release rejection test without touching a running implementation."""
import runpy
from run_units import ROOT,HERE,sha
from control_command_repair import once

def main():
 source=HERE/'preflight_registered_release.py';code=source.read_text()
 code=once(code,'import finish_registered_boundary_release as release','import finish_snapshot_ram_release as release')
 code=code.replace('registered-package-preflight1','snapshot-ram-package-preflight1')
 code=code.replace('must-not-exist-registered-preflight','must-not-exist-snapshot-ram-preflight')
 code=once(code,"'registered_boundary_transform_sha256'","'base_inputs_sha256'")
 driver=ROOT/'tmp/board-test/snapshot-ram-package-preflight-runner.py';assert not driver.exists()
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 runpy.run_path(str(driver),run_name='__main__')
if __name__=='__main__':main()
