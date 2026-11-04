"""Reserve source-isolated originals; not used by the research trainer."""
import json
from pathlib import Path
from PIL import Image
from .media import decode_frames
from .fpga_video_resize import area_3to2
from .runtime import configure_workspace, file_hash, write_json


def main():
    root=configure_workspace()
    dest=root/'runs/b1-reconstruction-research-20261005/holdout'
    dest.mkdir(exist_ok=False)
    parent=root/'data/processed/content-20261001-v4-minus1ev-draft/manifest.approved.json'
    manifest=json.loads(parent.read_text(encoding='utf-8'))
    groups={}
    for row in manifest['images']:
        if row['split']=='test':groups.setdefault(row['source'],[]).append(row)
    records=[]
    for source,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda z:z['frame_index'])
        chosen=[rows[round(i*(len(rows)-1)/4)] for i in range(5)]
        byframe={z['frame_index']:z for z in chosen}
        assert file_hash(source)==chosen[0]['source_sha256']
        for index,pts,rgb in decode_frames(source,indices=sorted(byframe),hlg_exposure_stops=-1.):
            x=area_3to2(rgb)
            p=dest/(Path(source).stem+f'_{index:05d}.png')
            Image.fromarray(x).save(p)
            records.append(dict(id=p.stem,path=str(p),sha256=file_hash(p),video=source,
                source_sha256=byframe[index]['source_sha256'],frame_index=index,pts=pts,
                split='source_isolated_test_reserved_this_run'))
        print('RESERVED_TEST_SOURCE',Path(source).name,len(chosen),flush=True)
    write_json(dest/'manifest.json',dict(records=records,parent=str(parent),parent_sha256=file_hash(parent),
        policy='Two designated test videos excluded from training/selection. Historically available, not claimed never viewed.',
        resize='exact FPGA integer area720 from decoded1080, fixed HLG -1EV'))


if __name__=='__main__':main()
