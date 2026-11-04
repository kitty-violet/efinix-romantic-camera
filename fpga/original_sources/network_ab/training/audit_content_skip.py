"""Diagnose high-resolution content residual versus current game stylizer."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .audit_adain_sharpness import edge_stats
from .runtime import configure_workspace, write_json


def main():
    root=configure_workspace()
    directory=root/"runs/adain-game-20261001-01/sharpness_audit"
    pair=np.array(Image.open(directory/"comparison_lossless_008.png").convert("RGB"),dtype=np.float32)/255
    source,style=pair[:,:1280],pair[:,1280:]
    low=cv2.GaussianBlur(source,(0,0),1.25)
    high=source-low
    output=[]
    for name,value in (("blend_050",.5*source+.5*style),
                       ("blend_065",.35*source+.65*style),
                       ("blend_080",.2*source+.8*style),
                       ("highpass_075",style+.75*high)):
        image=np.clip(value*255,0,255).astype(np.uint8)
        path=directory/(name+"_008.png")
        Image.fromarray(np.concatenate(((source*255).astype(np.uint8),image),axis=1)).save(path)
        output.append({"name":name,"path":str(path),
                       "full_frame":edge_stats((source*255).astype(np.uint8),image,(0,0,1280,720)),
                       "hand_region":edge_stats((source*255).astype(np.uint8),image,(250,210,740,720))})
    write_json(directory/"content_skip_diagnostics.json",output)
    print(json.dumps([{"name":r["name"],
                       "near_original_edge_fraction":r["full_frame"]["output_edges_near_input_fraction"]}
                      for r in output],indent=2))


if __name__=="__main__":
    main()
