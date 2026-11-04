"""Compile the IP-XACT-declared official CSI ModelSim fileset, unchanged.

This distinguishes a real encryption compatibility failure from a missing
preprocessor macro or an incorrect include option. No behavioral replacements.
"""
import argparse,json,re,subprocess,xml.etree.ElementTree as ET
from pathlib import Path
from run_units import ROOT,EF,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();out=a.root.resolve()
 assert out.is_relative_to(ROOT/'tmp') and not out.exists();out.mkdir(parents=True)
 component=EF/'ipm/ip/efx_csi2/efx_csi2_rx/ipm/ip_component.xml'
 tree=ET.parse(component).getroot();ns={'x':tree.tag.split('}')[0][1:]}
 fs=next(f for f in tree.findall('.//x:fileSet',ns) if f.find('x:name',ns).text=='external_testbench_modelsim')
 paths=[(component.parent/f.find('x:name',ns).text).resolve() for f in fs.findall('x:file',ns)]
 assert len(paths)==28 and all(f.is_file() for f in paths)
 macro=out/'macro.sv';macro.write_text('`define IP_MODULE_NAME(x) x\n')
 ms=Path('E:/altera/15.0/modelsim_ase/win32aloem')
 subprocess.run([str(ms/'vlib.exe'),'work'],cwd=out,check=True,capture_output=True)
 cmd=[str(ms/'vlog.exe'),'-sv','-mfcu',*[('+incdir+'+f.as_posix()) for f in sorted(set(f.parent for f in paths))],macro.as_posix(),*[f.as_posix() for f in paths]]
 r=subprocess.run(cmd,cwd=out,capture_output=True,timeout=180);b=r.stdout+r.stderr;(out/'compile.log').write_bytes(b)
 text=b.decode(errors='replace');errors=[x for x in text.splitlines() if x.startswith('** Error')]
 result=dict(returncode=r.returncode,compile_only=True,full_model_pass=False,error_count=len(errors),errors=errors[:8],compiled_modules=re.findall(r'-- Compiling module ([^\r\n]+)',text),
  component_sha256=sha(component),files=[dict(path=str(f),sha256=sha(f)) for f in paths],arguments=cmd)
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('files','arguments','compiled_modules')}))

if __name__=='__main__':main()
