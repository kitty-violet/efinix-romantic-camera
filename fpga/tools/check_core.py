"""Compile and elaborate the readable CNN with the included Efinix models."""
from pathlib import Path
import argparse,json,os,subprocess,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--questa',type=Path,required=True);ap.add_argument('--license',type=Path);ap.add_argument('--output',type=Path,default=ROOT/'build/core-check');a=ap.parse_args()
 q=a.questa.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);env=os.environ.copy()
 if a.license:env['SALT_LICENSE_SERVER']=str(a.license.resolve())
 rtl=ROOT/'network/rtl';tree=ET.parse(rtl/'cnn_production_frame.xml')
 src=[rtl/e.attrib['name'] for e in tree.iter() if e.tag.split('}')[-1]=='design_file']
 vendor=[ROOT/'tests/vendor'/n for n in ('efx_add.v','efx_dpram10.v','efx_dsp48.v','efx_lut4.v','efx_ram10.v','efx_srl8.v')]
 commands=[('vlib',['work']),('vlog',['-sv',*map(str,vendor+src)]),('vopt',['-suppress','14408','cnn_production_frame','-o','cnn_package_check'])]
 for name,args in commands:
  with (out/(name+'.log')).open('w',encoding='utf-8') as log:
   r=subprocess.run([str(q/(name+'.exe')),*args],cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
  if r.returncode:raise RuntimeError((out/(name+'.log')).read_text()[-3000:])
 # 14408 is the Starter edition's recommended 5000-instance performance
 # notice; no missing-module, syntax, port or arithmetic errors are suppressed.
 result={'status':'READABLE_CNN_COMPILE_ELABORATE_PASS','design_files':len(src),'vendor_models':len(vendor),'suppressed_recommendation':14408}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
