"""Versioned colour-transition update; keep the earlier preview reproducible."""
import argparse,ast,json,shutil
from pathlib import Path
from . import publish_coarse_people as base
from . import coarse_people_blocks as blocks
from .runtime import file_hash,write_json

OUT=base.ROOT/'person-colour-blocks-video-20261005-v1'
SOFT=base.ROOT/'coarse-people-video-20261005-v2/plaza_coarse_people_15s_1080p.mp4'
SOFT_HASH='775a09455543a81053a27a96d9c4f8089a2fc9b13412b28fa027fee7f4afc2f5'
PRESET='strongblocks'


def chosen_surface(pred,adapter,tone_module,preset='coarse',source=None):
    return blocks.person_surface(pred,adapter,tone_module,PRESET,source)


def main(short=False):
    assert file_hash(SOFT)==SOFT_HASH
    base.OUT=OUT;base.person_surface=chosen_surface;base.PRESETS={'coarse':blocks.PRESETS[PRESET]}
    extra=[Path(__file__),Path(blocks.__file__)]
    if not short:
        cfg=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'))
        assert all(file_hash(path)==cfg['live_code_hashes'][str(path)] for path in extra)
    base.render(short)
    if short:
        cfg=json.loads((OUT/'pipeline.json').read_text(encoding='utf-8'))
        for path in extra:
            ast.parse(path.read_text(encoding='utf-8'));sha=file_hash(path)
            cfg['live_code_hashes'][str(path)]=sha
            shutil.copy2(path,OUT/'source_code'/('blocks_'+path.name))
        cfg.update(person_preset=PRESET,comparison_previous_video=str(SOFT),comparison_previous_sha256=SOFT_HASH,
            face_levels=dict(l_step=8,l_sharp=9,l_mix=.84,ab_step=6,ab_sharp=8,ab_mix=.75),
            colour_guide=dict(median=3,bilateral_d=9,lab_sigma=8,spatial_sigma=3,skin_original_guide_fraction=.5),
            skin_tv_min=.014,tv_unchanged_from_previous_soft_preview=True,
            diagnostic_previous_means='original block-separation background, not the soft-person preview',
            change='shorten within-person colour transitions and use fewer fixed colour levels; retain anatomy and fixed palette')
        write_json(OUT/'pipeline.json',cfg)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['short','render']);args=parser.parse_args();main(args.mode=='short')
