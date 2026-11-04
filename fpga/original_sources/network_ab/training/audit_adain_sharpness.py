"""Lossless re-render and input-edge audit for the 500-step AdaIN game model."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch

from .adain_game_train import FixedStyle
from .adain_model import load_pretrained
from .data import image_tensor
from .evaluate import to_u8
from .runtime import configure_workspace, write_json


def edge_stats(reference, result, roi):
    x0,y0,x1,y1 = roi
    a = cv2.cvtColor(reference[y0:y1,x0:x1], cv2.COLOR_RGB2GRAY)
    b = cv2.cvtColor(result[y0:y1,x0:x1], cv2.COLOR_RGB2GRAY)
    ea, eb = cv2.Canny(a,80,160), cv2.Canny(b,80,160)
    kernel = np.ones((5,5),np.uint8)
    close = cv2.dilate(ea,kernel) > 0
    return {"input_canny_pixels": int((ea>0).sum()),
            "output_canny_pixels": int((eb>0).sum()),
            "output_edges_near_input_fraction": float(((eb>0)&close).sum()/max(1,(eb>0).sum())),
            "input_laplacian_variance": float(cv2.Laplacian(a,cv2.CV_64F).var()),
            "output_laplacian_variance": float(cv2.Laplacian(b,cv2.CV_64F).var())}


def main():
    root=configure_workspace()
    run=root/"runs/adain-game-20261001-01"
    checkpoint=torch.load(run/"checkpoint_000500.pt",map_location="cpu",weights_only=True)
    enc,dec=load_pretrained(root/"cache/pretrained/adain")
    dec.load_state_dict(checkpoint["decoder"],strict=True)
    enc,dec=enc.cuda(),dec.cuda().eval()
    content=json.loads((root/"data/processed/content-20260930-v2/manifest.approved.json").read_text(encoding="utf-8"))
    row=[r for r in content["images"] if r["split"]=="val"][8]
    style=json.loads((root/"data/processed/style-target-20261001-v1/dataset-manifest.json").read_text(encoding="utf-8"))
    s=next(r for r in style["records"] if r["file"].endswith("2-00540-F.png"))
    sr=root/"data/processed/style-target-20261001-v1"/s["file"]
    with torch.no_grad():
        value=image_tensor(row["path"],"cuda").unsqueeze(0)
        model=FixedStyle(enc,dec,((128*image_tensor(sr,"cuda")+128)/255).unsqueeze(0))
        out=model(value)[0]
    before,after=to_u8(value[0]),to_u8(out)
    directory=run/"sharpness_audit"
    directory.mkdir(exist_ok=True)
    path=directory/"comparison_lossless_008.png"
    Image.fromarray(np.concatenate((before,after),axis=1)).save(path)
    jpeg=np.array(Image.open(run/"validation_000500/comparison_2-00540-F_008.jpg").convert("RGB"))[:,1280:]
    report={"full_frame":edge_stats(before,after,(0,0,1280,720)),
            "hand_region":edge_stats(before,after,(250,210,740,720)),
            "jpeg_mean_absolute_error_vs_lossless":float(np.abs(jpeg.astype('float32')-after.astype('float32')).mean()),
            "lossless_comparison":str(path)}
    write_json(directory/"sharpness.json",report)
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
