"""Read-only source selection and isolated simulator compatibility probes.

Compilation success is never interpreted as functional or timing validation.
No production RTL, generated PHY or global simulator configuration is changed.
"""
import concurrent.futures, hashlib, json, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
MS=Path('E:/altera/15.0/modelsim_ase/win32aloem')
EF=Path('E:/efinity-2026.1.132-windows-x64/efinity/2026.1')
BASE=ROOT/'tmp/efinity-smoke-2026.1/sc431-v23-fifo2026-seed1-effort5/rtl'

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 out=a.root.resolve();assert out.is_relative_to(ROOT/'tmp') and not out.exists()
 out.mkdir(parents=True)
 specs={
  'current_csi':BASE/'csi_rx_controller.sv',
  'current_hdmi':BASE/'dvi_tx/dvi_encoder.v_encrypted.v',
  'current_ddr':BASE/'ddr3_controller/ddr3_top.v',
  'official_cpu_modelsim':ROOT/'tmp/efinity-smoke-2026.1/sapphire-lite-ipgen-i1-o4/ip/sapphire_lite_i1_o4/source/hardware/netlist/source_modelsim/EfxSapphireSoc.v',
  'official_csi_2026_modelsim':EF/'ipm/ip/efx_csi2/source_modelsim/efx_csi2_rx_top.sv',
  'legacy_csi_5_0_modelsim':ROOT/'官方示例/10_Ti60f225_sc431hai2hdmi_demo/Ti60f225_sc431hai2hdmi_v1/ip/csi_rx_controller/Testbench/modelsim/csi_rx_controller.sv',
 }
 def one(item):
  name,source=item;run=out/name;run.mkdir()
  assert source.is_file(),source
  vl=subprocess.run([str(MS/'vlib.exe'),'work'],cwd=run,capture_output=True,timeout=60)
  assert vl.returncode==0,vl.stdout
  cmd=[str(MS/'vlog.exe'),'-sv','-work','work','+incdir='+str(source.parent),str(source)]
  proc=subprocess.run(cmd,cwd=run,capture_output=True,timeout=120)
  log=proc.stdout+proc.stderr;(run/'compile.log').write_bytes(log)
  return dict(name=name,source=str(source),sha256=digest(source),returncode=proc.returncode,
   log_sha256=digest(run/'compile.log'),compile_only=True,
   errors=[s for s in log.decode(errors='replace').splitlines() if 'Error' in s or 'decrypt' in s][:8])
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
  results=list(pool.map(one,specs.items()))
 result=dict(scope='Compiler compatibility only; older/different IP is NOT a replacement',
  production_rtl_modified=False,simulator=str(MS),results=results)
 (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
