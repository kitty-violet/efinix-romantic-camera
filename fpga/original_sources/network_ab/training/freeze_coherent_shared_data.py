"""Freeze visual review decisions made from all 83 game candidates."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from .prepare_coherent_shared_data import contact
from .runtime import file_hash,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);a=p.parse_args()
    source=a.data/'manifest.draft.json';d=json.loads(source.read_text(encoding='utf-8'))
    selected=[1,2,3,4,5,7,8,9,13,14,15,16,17,19,20,23,24,25,26,27,28,29,
              43,47,48,49,50,53,55,57,67,69,70,71,72,73,74,75,79,81,82]
    indoor={2,3,4,5,7,8,13,14,43}
    special={20:[0,10,1000,660],43:[20,10,1040,620],67:[100,10,1230,570],
             73:[20,10,1240,610],75:[20,10,1200,560],79:[20,10,1050,660]}
    styles=[]
    for i in selected:
        row=d['style_candidates'][i]
        styles.append({**row,'candidate_index':i,'style_group':'indoor' if i in indoor else 'outdoor',
                       'safe_roi_xyxy':special.get(i,[20,10,1260,660]),
                       'new_review':'All 83 candidates visually inspected; representative scene, peripheral crop excludes visible overlays/foreground equipment where present.'})
    if len(d['pairs'])<100:raise ValueError('Insufficient real training pairs')
    d.update({'status':'REVIEWED_SHARED_TRAINING','styles':styles,
              'reviewed_from':str(source),'reviewed_from_sha256':file_hash(source),
              'review_policy':'Shared style distribution, no material pairing; exclude dominant vehicles/characters, obvious weapons/HUD, defocus, night and heavy sunset.',
              'rejected_style_indices':[i for i in range(len(d['style_candidates'])) if i not in selected]})
    for row in styles:
        if file_hash(row['path'])!=row['sha256']:raise ValueError('Target drift')
    write_json(a.data/'manifest.json',d)
    contact(styles,a.data,'selected_game_targets','path')
    print(json.dumps({'pairs':len(d['pairs']),'style_count':len(styles),'indoor_targets':len(indoor),
                     'training_sources':len({r['source'] for r in d['pairs']}),
                     'manifest':str(a.data/'manifest.json')},ensure_ascii=False))


if __name__=='__main__':main()
