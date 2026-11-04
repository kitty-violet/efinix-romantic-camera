"""Current validated map: release29 parameter RAMs, stop after placement."""
import argparse,json,runpy,subprocess,sys
from pathlib import Path
from run_units import HERE,ROOT,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True)
 p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();board=a.board.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution') and not root.exists()
 subprocess.run([sys.executable,str(HERE/'place_video_service_epoch.py'),'--board',str(board),'--root',str(root),'--prepare-only'],check=True,timeout=90)
 driver=root.parent/(root.name+'-wrapper.py');receipt=driver.with_suffix('.json')
 code=driver.read_text();old='from fixed_cnn_ram_locations import'
 assert code.count(old)==2
 code=code.replace(old,'from floating_cnn_parameter_ram_locations import')
 code=code.replace('same DSP cascade and CNN RAM anchors, no general logic fixed.',
  'same288 DSP cascade and84 arena RAM anchors;29 parameter RAMs and ordinary logic free.')
 compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
 m=json.loads(receipt.read_text());assert m['hard_anchors']==401
 m.update(hard_anchors=372,driver_sha256=sha(driver),
  parameter_ram_adapter_sha256=sha(__file__),
  location_generator_sha256=sha(HERE/'floating_cnn_parameter_ram_locations.py'))
 receipt.write_text(json.dumps(m,indent=2)+'\n')
 sys.argv=m['argv'];runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
