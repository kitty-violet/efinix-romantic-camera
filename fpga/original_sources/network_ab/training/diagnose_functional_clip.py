"""Measure each completed clip while remaining clips render; no frame skips."""
import argparse
import json
from .audit_style_branches import RUN
from .runtime import configure_workspace,write_json


def main():
    ap=argparse.ArgumentParser();ap.add_argument('scene',choices=['new_hallway','stairs','plaza']);args=ap.parse_args()
    root=configure_workspace();run=root/'runs'/RUN
    report=json.loads((run/'videos/report.json').read_text());row=next(r for r in report['records'] if r['id']==args.scene)
    previous=json.loads((run/'reference/videos/report.json').read_text());assert any(r['id']==args.scene for r in previous['records'])
    chunk=run/'diagnostic_chunks'/args.scene;(chunk/'videos').mkdir(parents=True,exist_ok=False)
    write_json(chunk/'videos/report.json',dict(status='COMPLETE',records=[row],scope='One complete scene, no trimmed frames'))
    from . import publish_fullres_c8_stylepush as frozen
    import cv2
    cv2.setNumThreads(2)
    frozen.PREVIOUS=RUN+'/reference';frozen.diagnostics(root,chunk)
    data=json.loads((chunk/'video_diagnostics.json').read_text());assert len(data['records'])==1 and data['records'][0]['frames']==row['validation']['frames']
    data['status']='CLIP_COMPLETE';data['scope']='Only this one complete clip; each adjacent pair measured'
    write_json(chunk/'video_diagnostics.json',data)
    names=['new_hallway','stairs','plaza'];parts=[]
    for name in names:
        p=run/'diagnostic_chunks'/name/'video_diagnostics.json'
        if p.exists():
            d=json.loads(p.read_text())
            if d['status']=='CLIP_COMPLETE':parts.append(d)
    aggregate=dict(status='COMPLETE' if len(parts)==3 else 'MEASURING',records=[r for d in parts for r in d['records']],
        bitexact_area_checks=sum(d['bitexact_area_checks'] for d in parts),grid=[1280,720],flow_estimation_grid=[640,360],limitations=data['limitations'])
    write_json(run/'video_diagnostics.json',aggregate)
    print('CLIP_DIAGNOSTIC_COMPLETE',args.scene,'total',len(parts),flush=True)


if __name__=='__main__':main()
