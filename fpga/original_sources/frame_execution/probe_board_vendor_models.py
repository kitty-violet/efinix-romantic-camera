"""Try supported simulation envelopes without modifying or decrypting IP.

The fixed Micron parameter mirror is a reference artifact; it is not the exact
board speed-bin datasheet and does not qualify physical DDR timing.
"""
import argparse,concurrent.futures,json,re,shutil,subprocess,urllib.request
from pathlib import Path
from run_units import ROOT,EF,sha
MS=Path('E:/altera/15.0/modelsim_ase/win32aloem')

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 nets=ROOT/'tmp/frame-execution/full-board-ip-netlists1'
 specs={n:([nets/n/'outflow/board_ip.map.v'],[]) for n in ('csi_rx_controller','ddr3_top','dvi_encoder')}
 # Correct the macro omission in the earlier compile-only probe. This does
 # not assert equivalence of the installed 5.16 package to the frozen 5.15 IP.
 spec=root/'csi_macro.sv';spec.write_text('`define IP_MODULE_NAME(x) x\n')
 csi=EF/'ipm/ip/efx_csi2/source_modelsim/efx_csi2_rx_top.sv'
 specs['installed_csi_with_macro']=([spec,csi],['-mfcu'])
 model=EF/'ipm/ip/efx_ddr3_soft_controller/testbench/ddr3.v'
 url='https://raw.githubusercontent.com/buttercutter/DDR/ff8625614e2ae16537d794fa74a64de935332dbd/2048Mb_ddr3_parameters.vh'
 header=root/'2048Mb_ddr3_parameters.vh';header.write_bytes(urllib.request.urlopen(url,timeout=20).read())
 s=header.read_text();assert 'Micron' in s and '`elsif sg125' in s
 specs['micron_2gb_sg125_x16']=([model],['+define+den2048Mb+sg125+x16','+incdir+'+root.as_posix()])
 def one(item):
  name,(files,args)=item;run=root/name;run.mkdir()
  subprocess.run([str(MS/'vlib.exe'),'work'],cwd=run,check=True,capture_output=True)
  p=subprocess.run([str(MS/'vlog.exe'),'-sv',*args,*map(str,files)],cwd=run,capture_output=True,timeout=180)
  b=p.stdout+p.stderr;(run/'compile.log').write_bytes(b)
  body=b.decode(errors='replace');return dict(name=name,returncode=p.returncode,compile_only=True,errors=[line for line in body.splitlines() if line.startswith('** Error')][:6],files=[dict(path=str(f),sha256=sha(f)) for f in files])
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(one,specs.items()))
 result=dict(results=results,micron_parameter_reference=dict(url=url,sha256=sha(header),scope='2012 Micron 2Gb model parameter reference, exact JT-125 electrical qualification remains open'),full_frame_pass=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
