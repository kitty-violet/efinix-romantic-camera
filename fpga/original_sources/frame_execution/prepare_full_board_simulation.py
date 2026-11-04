"""Freeze the real board with vendor simulation views, with source-pair evidence.

CSI synthesis ciphertext is matched verbatim (ignoring whitespace) to the
installed IP package's source_efinity before selecting its paired source_modelsim.
No decrypted content, replacement behavior or internal forces are introduced.
"""
import argparse,json,re,shutil,subprocess,time
from pathlib import Path
from run_units import ROOT,HERE,EF,sha
from observe_board_boot import MODEL
from board_integration.boot_image import make_program

def compact(text):return re.sub(r'\s+','',text)
def envelopes(text):return re.findall(r'`pragma protect begin_protected.*?`pragma protect end_protected',text,re.S)

def prepare(root,base,boot):
 meta=json.loads((base/'inputs.json').read_text());records=[];sources=[]
 for row in meta['inputs']:
  src=Path(row['snapshot']);assert sha(src)==row['sha256']
  if src.name in ('csi_rx_controller.sv','EfxSapphireSoc.v'):continue
  dst=root/src.relative_to(base)
  declaration={'ti60f225_oob_top.v':'wire camera_write_fault,camera_read_fault,camera_line_fault;',
               'cnn_production_frame.sv':'reg datapath_fault_q;'}.get(src.name)
  if declaration:
   live=HERE/('board_integration' if src.name=='ti60f225_oob_top.v' else '')/src.name
   if meta.get('ddr516') and src.name=='ti60f225_oob_top.v':
    from prepare_ddr516_board import adapt_top
    assert src.read_text()==adapt_top(live.read_text())
   else:
    assert compact(src.read_text()).replace(compact(declaration),'')==compact(live.read_text()).replace(compact(declaration),''),'Only declaration ordering may change'
    src=live
  if meta.get('ddr516') and src.name=='ddr516_reference.sv':
   paired=Path(meta['ddr516_generation'])/'ip/ddr516_reference/Testbench/modelsim/ddr516_reference.sv'
   assert src.read_text().split('//pragma protect',1)[0]==paired.read_text().split('//pragma protect',1)[0]
   src=paired
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
  if src.suffix in ('.v','.sv'):sources.append(dst)
 # Include paths and non-design headers are also immutable inputs.
 for src in (base/'rtl').rglob('*'):
  if src.is_file() and (src.suffix in ('.vh','.svh','.mem','.hex') or src.name in ('timescale.v','i2c_master_defines.v')):
   dst=root/src.relative_to(base)
   if not dst.exists():
    dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 for src in [base/'ddr_memory_map.vh',*base.glob('*.bin')]:
  if src.name.startswith('EfxSapphireSoc.v_toplevel_system_ramA_'):continue
  dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 expected,_=make_program();assert (boot/'boot.bin').read_bytes()==expected
 for src in boot.glob('EfxSapphireSoc*.bin'):
  dst=root/src.name;shutil.copy2(src,dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 raw=(base/'assembly/EfxSapphireSoc.v').read_text();attr='(* syn_ramstyle = "registers" *) '
 assert raw.split('module EfxCPUSp1')[0].replace(attr,'')==MODEL.read_text().split('module EfxCPUSp1')[0].replace(attr,'')
 dst=root/'EfxSapphireSoc.v';shutil.copy2(MODEL,dst);sources.append(dst)
 records.append(dict(source=str(MODEL),snapshot=str(dst),sha256=sha(dst),view='official paired CPU simulation view; public RTL matches'))
 csi=base/'rtl/csi_rx_controller.sv';text=csi.read_text();norm=compact(text)
 files=json.loads((ROOT/'tmp/frame-execution/csi-official-bundle1/result.json').read_text())['files']
 paired=[];blocks=set();wrapper=text[:text.index('endmodule')+len('endmodule')]
 simtext=[wrapper]
 for row in files:
  sim=Path(row['path']);assert sha(sim)==row['sha256'];syn=Path(str(sim).replace('source_modelsim','source_efinity'))
  s=syn.read_text();enc=envelopes(s)
  if enc:
   assert all(compact(block) in norm for block in enc),syn
   blocks.update(map(compact,enc))
  if 'module ' in s:
   module=s[s.index('module '):s.rfind('endmodule')+len('endmodule')]
   assert compact(module) in norm,syn
  assert enc or 'module ' in s
  paired.append(dict(synthesis_source=str(syn),synthesis_sha256=sha(syn),simulation_source=str(sim),simulation_sha256=sha(sim),envelopes=len(enc)))
  simtext.append(sim.read_text())
 assert set(map(compact,envelopes(text)))==blocks,'Unmatched board CSI protected block'
 dst=root/'csi_rx_controller_sim.sv';dst.write_text('\n'.join(simtext));sources.append(dst)
 records.append(dict(source='Audited official paired CSI simulation bundle',snapshot=str(dst),sha256=sha(dst)))
 for src in sorted((EF/'sim_models/verilog').glob('*.v')):
  dst=root/src.name;shutil.copy2(src,dst);sources.append(dst);records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst)))
 inc=sorted(set(f.parent for f in root.rglob('*') if f.is_file()))
 manifest=dict(base=str(base),base_inputs_sha256=sha(base/'inputs.json'),boot_sha256=sha(boot/'boot.bin'),inputs=records,
  csi_board_sha256=sha(csi),csi_simulation_sha256=sha(root/'csi_rx_controller_sim.sv'),csi_pairs=paired,
  source_files=[str(f) for f in sources],include_dirs=[str(f) for f in inc],functional_pass=False,full_board=False)
 manifest['ddr516']=meta.get('ddr516',False)
 (root/'inputs.json').write_text(json.dumps(manifest,indent=2));return sources,inc,manifest

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--boot-image',type=Path,required=True);p.add_argument('--sim-bin',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir(parents=True)
 files,inc,manifest=prepare(root,a.base.resolve(),a.boot_image.resolve())
 subprocess.run([str(a.sim_bin/'vlib.exe'),'work'],cwd=root,check=True,capture_output=True)
 cmd=[str(a.sim_bin/'vlog.exe'),'-sv','-timescale','1ns/1ps',*[('+incdir+'+str(f)) for f in inc],*[str(f) for f in files]]
 (root/'compile_args.json').write_text(json.dumps(cmd,indent=2));t=time.monotonic()
 with (root/'compile.log').open('wb') as log:r=subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=300)
 body=(root/'compile.log').read_text(errors='replace')
 result=dict(status='BOARD_COMPILE_PASS' if r.returncode==0 else 'BOARD_COMPILE_FAIL',returncode=r.returncode,seconds=time.monotonic()-t,
  csi_all_synthesis_blocks_matched=True,csi_paired_files=len(manifest['csi_pairs']),errors=[v for v in body.splitlines() if '** Error' in v],inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'compile.log'),full_board_functional_pass=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
