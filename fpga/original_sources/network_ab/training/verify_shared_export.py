"""Verify exported weights through an independent, locally packaged entrypoint."""
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
    model=Path(__file__).with_name('coherent_shared_model.py')
    entry=Path(r'E:\fpga\model-training-workspace\runs\coherent-shared-20261002\final-2500\infer_image.py')
    if file_hash(data['weights'])!=data['weights_sha256']:raise ValueError('Exported weights drift')
    shutil.copy2(model,out/model.name);shutil.copy2(entry,out/entry.name)
    row=data['clips'][0];cap=cv2.VideoCapture(row['clip']);ok,bgr=cap.read();cap.release()
    if not ok:raise ValueError('Cannot decode source frame')
    Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)).save(out/'export_check_input.png')
    subprocess.run([sys.executable,str(out/entry.name),'--input',str(out/'export_check_input.png'),
                    '--output',str(out/'export_check_output.png')],check=True,cwd=str(out))
    expected=np.array(Image.open(out/f"{row['id']}_000_styled.png")).astype(np.int16)
    actual=np.array(Image.open(out/'export_check_output.png')).astype(np.int16)
    error=np.abs(actual-expected)
    result={'status':'PASS' if error.max()==0 else 'FAILED','pixels_max_abs_error':int(error.max()),
        'pixels_mean_abs_error':float(error.mean()),'weights_sha256':data['weights_sha256'],
        'shape':list(actual.shape),'device':'cuda','dtype':'FP32',
        'model_sha256':file_hash(out/model.name),'entrypoint_sha256':file_hash(out/entry.name),
        'reference':f"{row['id']}_000_styled.png"}
    write_json(out/'export_check.json',result)
    if result['status']!='PASS':raise RuntimeError(result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
