"""Compile the complete frozen board with official mapped vendor-IP models.

No periphery is stubbed. This establishes model compatibility only; it never
claims successful startup, valid frames, physical timing or electrical proof.
"""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
sys.path.insert(0,str(HERE.parent/'pw_array_feed_abc'))
from run_native import BIN,VROOT

def prepare(root):
 base=ROOT/'tmp/frame-execution/connected-board-sharing1';nets=ROOT/'tmp/frame-execution/full-board-ip-netlists1'
 manifest=json.loads((base/'inputs.json').read_text());records=[];paths=[]
 for folder in ('rtl','assembly'):shutil.copytree(base/folder,root/folder)
 for f in base.glob('*.bin'):shutil.copy2(f,root/f.name)
 for r in manifest['inputs']:
  f=Path(r['snapshot']);assert sha(f)==r['sha256'];rel=f.relative_to(base)
  if f.suffix not in ('.v','.sv'):continue
  if any(s in str(rel) for s in ('ddr3_controller','dvi_tx')) or f.name in ('EfxSapphireSoc.v','csi_rx_controller.sv'):continue
  dst=root/rel;paths.append(dst);records.append(dict(path=str(f),snapshot=str(dst),sha256=sha(dst)))
 for module in ('csi_rx_controller','dvi_encoder','EfxSapphireSoc','ddr3_top'):
  src=nets/module/'outflow/board_ip.map.v';meta=json.loads((nets/module/'result.json').read_text())
  assert sha(src)==meta['net_sha256'];dst=root/(module+'.v');shutil.copy2(src,dst)
  paths.append(dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst),model='Efinity primitive netlist of unchanged module'))
 for name in ('efx_lut4','efx_ff','efx_add','efx_comb4','efx_srl8','efx_ram10','efx_dpram10','efx_dsp24','efx_dsp48'):
  src=EF/'sim_models/verilog'/f'{name}.v';dst=root/src.name;shutil.copy2(src,dst)
  paths.append(dst);records.append(dict(path=str(src),snapshot=str(dst),sha256=sha(dst)))
 includes=sorted(set(f.parent for f in (root/'rtl').rglob('*') if f.is_file()))+[root/'assembly']
 return paths,includes,records

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 paths,includes,records=prepare(root)
 env=os.environ.copy();env.update(VERILATOR_ROOT=str(VROOT),PATH=str(BIN)+';'+env['PATH'],MSYS2_ARG_CONV_EXCL='*')
 cmd=[str(BIN/'verilator_bin.exe'),'--lint-only','--timing','--sched-zero-delay','--top-module','ti60f225_oob_top','-Wno-fatal',*[('-I'+f.as_posix()) for f in includes],*[f.as_posix() for f in paths]]
 (root/'arguments.json').write_text(json.dumps(cmd,indent=2));(root/'inputs.json').write_text(json.dumps(records,indent=2))
 t=time.monotonic()
 with (root/'compile.log').open('wb') as log:r=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 body=(root/'compile.log').read_text(errors='replace')
 result=dict(returncode=r.returncode,seconds=time.monotonic()-t,errors=re.findall(r'^%Error[^\n]+',body,re.M),warning_codes=sorted(set(re.findall(r'^%Warning-([^:]+)',body,re.M))),model_compile_pass=r.returncode==0,full_frame_pass=False,production_rtl_modified=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
