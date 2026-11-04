"""Reload the exported surface CNN in an independent process and compare pixels."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import cv2
import numpy as np
from PIL import Image
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--render',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.render.read_text());out=a.render.parent
    if file_hash(data['weights'])!=data['weights_sha256']:raise ValueError('Export drift')
    files={}
    for name in ('coherent_shared_model.py','coherent_surface_model.py','infer_surface_image.py'):
        path=Path(__file__).with_name(name);dst=out/('infer_image.py' if name=='infer_surface_image.py' else name)
        shutil.copy2(path,dst);files[dst.name]=file_hash(dst)
    row=data['clips'][0];cap=cv2.VideoCapture(row['clip']);ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError('Source decode failure')
    Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(out/'export_check_input.png')
    subprocess.run([sys.executable,str(out/'infer_image.py'),'--input',str(out/'export_check_input.png'),
      '--output',str(out/'export_check_output.png')],check=True,cwd=str(out))
    expected=np.array(Image.open(out/f"{row['id']}_000_styled.png")).astype(np.int16)
    actual=np.array(Image.open(out/'export_check_output.png')).astype(np.int16)
    error=np.abs(expected-actual)
    result={'status':'PASS' if error.max()==0 else 'FAIL','pixels_max_abs_error':int(error.max()),
      'pixels_mean_abs_error':float(error.mean()),'weights_sha256':data['weights_sha256'],
      'shape':list(actual.shape),'device':'cuda','dtype':'FP32','files':files,
      'inference':'CNN and fixed resize only; training tone prior and bilateral filter are not packaged or called.'}
    write_json(out/'export_check.json',result)
    if result['status']!='PASS':raise RuntimeError(result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
