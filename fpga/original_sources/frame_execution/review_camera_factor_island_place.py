"""Review the whole placed chip, edited controls and retained data hotspots."""
import argparse,json,runpy,sys
from pathlib import Path
from run_units import ROOT,HERE,sha
from control_command_repair import once
from camera_factor_island_locations import verify_placed

def main():
 p=argparse.ArgumentParser();p.add_argument('--placement',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 place=a.placement.resolve();out=a.root.resolve();assert not out.exists();out.mkdir()
 result=verify_placed(place)
 source=HERE/'review_camera_factor_balanced_place.py';code=source.read_text()
 code=once(code,'No added clocks/resets or individually fixed ordinary logic.',
  'No added clocks/resets. 36 exactly matched arena FF/SRL cells retain complete prior legal tiles; remaining ordinary logic is free.')
 sys.argv=[str(source),'--placement',str(place),'--out',str(out/'global.json')]
 exec(compile(code,str(source)+':island-review','exec'),{'__name__':'__main__','__file__':str(source)})
 source=ROOT/'tmp/board-test/review-camera-factor-retained-hotspots.py';code=source.read_text()
 code=once(code,"R/'tmp/frame-execution/camera-factor-balanced-place1'",repr(place))
 code=once(code,"R/'tmp/board-test/camera-factor-balanced-retained-hotspots1.json'",repr(out/'retained.json'))
 # Path repr uses WindowsPath; supply it explicitly for generated constants.
 from pathlib import WindowsPath
 exec(compile(code,str(source)+':island-hotspots','exec'),{'__name__':'__main__','__file__':str(source),'WindowsPath':WindowsPath})
 result['evidence']={str(p):sha(p) for p in [out/'global.json',out/'retained.json',Path(__file__),place/'arena_island_placement.json']}
 (out/'result.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
