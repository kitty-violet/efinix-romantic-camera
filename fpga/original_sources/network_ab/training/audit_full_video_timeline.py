"""Dense, read-only game-video quality survey before selecting any new style crop."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .media import decoder_identity
from .runtime import configure_workspace, file_hash, write_json


def metrics(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    # Top telemetry and the game's far-right HUD are excluded from scoring.
    region=gray[42:485,40:790]
    parts=[]
    for yi in range(3):
        for xi in range(5):
            tile=region[yi*147:(yi+1)*147,xi*150:(xi+1)*150]
            lap=cv2.Laplacian(tile,cv2.CV_32F)
            grad_x=cv2.Sobel(tile,cv2.CV_32F,1,0,ksize=3)
            grad_y=cv2.Sobel(tile,cv2.CV_32F,0,1,ksize=3)
            parts.append({"tile": [yi,xi], "laplacian_variance":float(lap.var()),
                          "tenengrad":float((grad_x*grad_x+grad_y*grad_y).mean()),
                          "luma":float(tile.mean()),
                          "dark_fraction":float((tile<10).mean()),
                          "bright_fraction":float((tile>245).mean())})
    valid=[r for r in parts if r["dark_fraction"]<.45 and r["bright_fraction"]<.45]
    lap=sorted((r["laplacian_variance"] for r in valid),reverse=True)
    return {"tile_metrics":parts,
            "usable_tile_count":len(valid),
            "best_3_tile_laplacian":float(np.mean(lap[:3])) if len(lap)>=3 else 0.0,
            "median_tile_laplacian":float(np.median(lap)) if lap else 0.0,
            "overall_laplacian":float(cv2.Laplacian(region,cv2.CV_32F).var())}


def sample(task):
    video, second, exe, output=task
    command=[exe,"-hide_banner","-loglevel","error","-nostdin","-ss",f"{second:.3f}",
             "-i",str(video),"-map","0:v:0","-frames:v","1",
             "-vf","scale=960:540:flags=bicubic,format=rgb24",
             "-f","rawvideo","-pix_fmt","rgb24","pipe:1"]
    result=subprocess.run(command,capture_output=True,timeout=90)
    if result.returncode or len(result.stdout)!=960*540*3:
        raise RuntimeError(f"Frame decode failed {video.name} {second}s: {result.stderr[-200:]!r}")
    rgb=np.frombuffer(result.stdout,dtype=np.uint8).reshape(540,960,3)
    info=metrics(rgb)
    thumb=Image.fromarray(rgb).resize((480,270),Image.Resampling.BICUBIC)
    name=f"{video.stem}_{int(second):05d}.jpg"
    path=output/name
    thumb.save(path,quality=92)
    return {"video":video.name,"second":second,"thumbnail":str(path),
            "thumbnail_sha256":file_hash(path),**info}


def contacts(rows, output):
    pages=[]
    for start in range(0,len(rows),16):
        page=Image.new("RGB",(1920,4*300),"#181818")
        draw=ImageDraw.Draw(page)
        for slot,row in enumerate(rows[start:start+16]):
            x,y=slot%4*480,slot//4*300
            with Image.open(row["thumbnail"]) as thumb:
                page.paste(thumb,(x,y+28))
            draw.text((x+5,y+5),f"{row['video']} {row['second']:.0f}s  sharp={row['best_3_tile_laplacian']:.0f}",fill="white")
        target=output/f"contact_{start//16:03d}.jpg"
        page.save(target,quality=94)
        pages.append(str(target))
    return pages


def main():
    root=configure_workspace()
    source=root/"data/incoming/style-target-replacement"
    old=json.loads((root/"runs/style-target-review-20261001/source_inventory.json").read_text(encoding="utf-8"))
    output=root/"runs/style-target-full-timeline-20261001"
    output.mkdir(parents=True,exist_ok=False)
    thumbs=output/"thumbnails"
    thumbs.mkdir()
    exe=decoder_identity()["path"]
    tasks=[]
    sources=[]
    for item in old["sources"]:
        video=Path(item["file"])
        if video.parent.resolve()!=source.resolve() or file_hash(video)!=item["sha256"]:
            raise ValueError("Game source moved or changed")
        duration=item["duration_seconds"]
        times=[float(t) for t in range(5,int(duration-2),5)]
        tasks.extend((video,t,exe,thumbs) for t in times)
        sources.append({"file":str(video),"sha256":item["sha256"],
                        "duration_seconds":duration,"interval_seconds":5,"sample_count":len(times)})
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(sample,task) for task in tasks]
        for count,future in enumerate(as_completed(futures),1):
            rows.append(future.result())
            if count%100==0:
                print(f"scanned {count}/{len(tasks)}",flush=True)
    rows.sort(key=lambda x:(x["video"],x["second"]))
    # Highest usable regional detail within each 60-second bin; every minute is represented.
    bins={}
    for row in rows:
        key=(row["video"],int(row["second"]//60))
        if key not in bins or row["best_3_tile_laplacian"]>bins[key]["best_3_tile_laplacian"]:
            bins[key]=row
    overview=sorted(bins.values(),key=lambda x:(x["video"],x["second"]))
    pages=contacts(overview,output)
    report={"schema":"dense-style-source-quality-v1","status":"REVIEW_ONLY",
            "sources":sources,"sample_count":len(rows),"interval_seconds":5,
            "rows":rows,"best_per_minute":overview,"contact_pages":pages,
            "notes":"Regional sharpness is a ranking aid only; UI, foreground hands, objects and game effects require visual review at native pixels."}
    write_json(output/"timeline.json",report)
    print(json.dumps({"frames":len(rows),"minutes":len(overview),"pages":len(pages)},ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
