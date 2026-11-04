"""Resume a prepared offline placement after the tool DB sandbox denial only."""
import argparse,json,re,shutil,subprocess,time,traceback,sys,textwrap
from pathlib import Path
from types import SimpleNamespace
from run_units import ROOT,EF,sha
from windows_process_tree_safe import stop_launched_tree

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);arg=p.parse_args();root=arg.root.resolve()
 assert root.is_relative_to(ROOT/'tmp/frame-execution')
 physical=json.loads((root/'physical_inputs.json').read_text());base=Path(physical['base'])
 meta=json.loads((base/'inputs.json').read_text());mapped=json.loads((base/'result.json').read_text())
 assert sha(base/'inputs.json')==physical['base_inputs_sha256']==mapped['inputs_sha256']
 for row in physical['inputs']:assert sha(row['snapshot'])==row['sha256']
 assert sha(root/'outflow/ti60f225_oob.map.v')==physical['base_map_sha256']==mapped['map_sha256']
 old=json.loads((root/'result.json').read_text())
 assert len(old['stages'])==1 and old['stages'][0]['flow']=='interface' and old['stages'][0]['exit_code']==1
 log=(root/'outflow/ti60f225_oob.log').read_text()
 assert "PermissionError: [WinError 5]" in log and 'pt\\\\db' in log
 assert not (root/'outflow/ti60f225_oob.place.rpt').exists()
 backup=root/'interface-denied-evidence';backup.mkdir()
 for rel in ['result.json','stages.json','interface.log','interface.cmd','outflow/ti60f225_oob.log']:
  f=root/rel
  if f.exists():shutil.copy2(f,backup/Path(rel).name)
 (backup/'resume.json').write_text(json.dumps(dict(runner_sha256=sha(__file__),physical_inputs_sha256=sha(root/'physical_inputs.json'),reason='Existing vendor device DB denied by sandbox; no design or tool installation changes',hardware_access=False),indent=2))
 driver=root.parent/(root.name+'-place-driver.py');code=driver.read_text();start=code.index('stages=[]\ntry:')
 assert sha(driver)==physical['runner_sha256']
 a=SimpleNamespace(pnr_budget=physical['pnr_budget_seconds'],logic_limit=physical['logic_limit'],packing_only=False)
 scope=dict(globals(),a=a,base=base,root=root,meta=meta,mapped=mapped,__file__=str(driver))
 exec(compile(code[start:],str(driver)+':resume-after-denial','exec'),scope)
 result=json.loads((root/'result.json').read_text());assert all(x['exit_code']==0 for x in result['stages'])
 wrapper=root.parent/(root.name+'-wrapper.py');s=wrapper.read_text()
 tail=s[s.index('    from fixed_cnn_ram_locations import verify_placed'):s.index("if __name__=='__main__'")]
 exec(compile(textwrap.dedent(tail),str(wrapper)+':review','exec'),dict(globals(),root=root))
if __name__=='__main__':main()
