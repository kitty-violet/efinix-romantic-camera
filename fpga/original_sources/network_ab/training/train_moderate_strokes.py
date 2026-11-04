"""Close-range contour adjustment using the saved style's soft-stroke loss."""
import json
import shutil
import sys
from pathlib import Path
from . import train_moderate_contours as runner
from . import train_teacher_strokes as old
from .moderate_stroke_targets import LEVELS,target
from .runtime import file_hash,write_json


def main():
    runner.ORIGINAL_CRITERION=old.criterion
    runner.LEVELS=LEVELS;runner.target=target
    if '--name' not in sys.argv:sys.argv.extend(['--name','moderate-stroke-grains-v2'])
    runner.main()
    if sys.argv[1]=='prepare':
        out=runner.base.ROOT/sys.argv[sys.argv.index('--name')+1]
        cfg=json.loads((out/'config.json').read_text(encoding='utf-8'))
        cfg.update(lr=.0002,objective='same soft dark strokes as archive, only remove weak small components; no RGB or BCE',
                   label=dict(kind='teacher_soft_dark_stroke_component_selection',parameters=LEVELS,label_only=True))
        write_json(out/'config.json',cfg);b=json.loads((out/'binding.json').read_text(encoding='utf-8'))
        for name in ['train_moderate_strokes.py','moderate_stroke_targets.py','train_teacher_strokes.py']:
            p=Path(__file__).with_name(name);b['code'][str(p)]=file_hash(p);shutil.copy2(p,out/'source'/p.name)
        b['config']=file_hash(out/'config.json');write_json(out/'binding.json',b)


if __name__=='__main__':main()
