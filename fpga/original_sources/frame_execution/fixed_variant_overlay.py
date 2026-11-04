"""Compose explicit RTL variants; runners still freeze/hash every selected file."""
import json
from pathlib import Path
from run_units import HERE

def resolve_overlay(folder, seen=()):
    folder=Path(folder).resolve()
    assert folder.is_relative_to(HERE/'variants') and folder.is_dir(),folder
    assert folder not in seen and len(seen)<32,('variant inheritance cycle or excessive depth',folder)
    chosen={};config=folder/'variant.json'
    if config.exists():
        data=json.loads(config.read_text(encoding='utf-8'))
        assert set(data)=={'base'} and isinstance(data['base'],str),config
        chosen.update(resolve_overlay(folder/data['base'],(*seen,folder)))
    chosen.update({p.name:p for p in folder.glob('*.sv')})
    assert chosen,('empty variant',folder)
    return chosen
