"""Record the user's style-name correction without touching frozen packages."""
import json
import library as l

def main():
    d=l.D/'revisions/film-pixel-r2';d.mkdir(parents=True,exist_ok=True)
    catalog=json.loads((l.D/'inventory.json').read_text('utf-8'))
    changes=[]
    for e in catalog['styles']:
        if e['id'] not in ['R8-06','R8-07']:continue
        previous=e.get('current_display_name',e['name']);name='热成像' if e['id']=='R8-06' else '漫画'
        e.update(current_display_name=name,formal_name=name,user_name_corrected='2026-11-02',previous_display_name=previous)
        changes.append(dict(id=e['id'],previous_display_name=previous,display_name=name,original_coefficient_sha256=e['source_sha256'],new_structure_coefficient_sha256=l.sha(l.D/'models'/e['id']/'coefficients-padded.bin'),coefficient_values_changed=False))
    l.save_json(d/'inventory.json',catalog)
    l.save_json(d/'name-corrections.json',dict(status='USER_CORRECTED_NAMES',changes=changes,red_style_id='R8-06',old_frozen_zip_unchanged=True,hardware_labels_not_modified=True))
    print('NAME_CORRECTIONS_SAVED',changes,flush=True)

if __name__=='__main__':main()
