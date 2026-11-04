"""Freeze one reviewed public RTL override against an existing whole-board map."""
import argparse,json,shutil
from pathlib import Path
from run_units import ROOT,HERE,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
 p.add_argument('--root',type=Path,required=True);p.add_argument('--override',type=Path,action='append',required=True)
 a=p.parse_args();base=a.base.resolve();root=a.root.resolve();overrides=[v.resolve() for v in a.override]
 assert root.is_relative_to(ROOT/'tmp') and not root.exists()
 assert all(v.is_relative_to(HERE/'variants') and v.suffix in ('.sv','.v') for v in overrides)
 by_name={v.name:v for v in overrides};assert len(by_name)==len(overrides)
 meta=json.loads((base/'inputs.json').read_text());rows=meta['inputs']
 for row in rows:assert sha(row['snapshot'])==row['sha256'],row['snapshot']
 for name in by_name:assert sum(Path(r['snapshot']).name==name for r in rows)==1,name
 root.mkdir();records=[]
 # Includes absent from the source manifest also stay byte-identical.
 for name in ('rtl','assembly'):shutil.copytree(base/name,root/name)
 for row in rows:
  old=Path(row['snapshot']);dst=root/old.relative_to(base)
  src=by_name.get(old.name,old)
  dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
  records.append(dict(source=str(src),snapshot=str(dst),sha256=sha(dst),base_sha256=row['sha256']))
 meta.update(inputs=records,base=str(base),base_inputs_sha256=sha(base/'inputs.json'),
  source_files=[str(root/Path(f).relative_to(base)) for f in meta['source_files']],
  override=dict(path=str(overrides[0]),sha256=sha(overrides[0])) if len(overrides)==1 else None,
  overrides=[dict(path=str(v),sha256=sha(v)) for v in overrides],board_release=False)
 (root/'inputs.json').write_text(json.dumps(meta,indent=2))
 print('BOARD_VARIANT_FROZEN',flush=True)

if __name__=='__main__':main()
