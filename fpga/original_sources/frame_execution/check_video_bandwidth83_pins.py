"""Compare generated physical pins and all original Gray endpoint names."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import sha
from video_bandwidth83_physical import REFERENCE

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
assert json.loads((root/'result.json').read_text())['status']=='PLACEMENT_ONLY_FINISHED_NOT_RELEASE'
files=[d/n for d in (REFERENCE,root) for n in ('ti60f225_oob.peri.xml','outflow/ti60f225_oob.interface.csv')]
assert sha(files[0])==sha(files[2]),'Periphery inputs changed'
old=files[1].read_text().splitlines();new=files[3].read_text().splitlines();assert len(old)==len(new)
diff=[(i,x,y) for i,(x,y) in enumerate(zip(old,new)) if x!=y]
assert len(diff)<=1 and all(x.startswith('# Date: ') and y.startswith('# Date: ') for _,x,y in diff),diff
result=dict(status='ORIGINAL_PERIPHERY_INPUT_AND_GENERATED_PINS_MATCH_EXCEPT_DATE',differences=diff,
 ignored_content='Generated Date comment only; every other CSV line equal',
 evidence={str(f):sha(f) for f in files+[Path(__file__).resolve()]},hardware_access=False)
(root/'original-periphery-output-check.json').write_text(json.dumps(result,indent=2))
sys.argv=['check_video_sensor73_gray_endpoints.py','--root',str(root)]
runpy.run_path(str(Path(__file__).with_name(sys.argv[0])),run_name='__main__')
print(result['status'])
