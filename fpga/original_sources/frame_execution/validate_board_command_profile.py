"""Check command accounting with independent known windows and corrupt evidence."""
import argparse,copy,json
from pathlib import Path
from run_units import ROOT,sha
from summarize_board_command_profile import analyze

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp');root.mkdir(exist_ok=False)
 commands=[dict(program_index=3,node=0),dict(program_index=8,node=1)]
 rows=[dict(frame=7,program_index=3,begin_cycle=100,end_cycle=130,elapsed_cycles=30),
       dict(frame=7,program_index=8,begin_cycle=140,end_cycle=190,elapsed_cycles=50),
       dict(frame=8,program_index=3,begin_cycle=205,end_cycle=235,elapsed_cycles=30)]
 good=analyze(rows,commands,{7:100})
 assert good[0]['command_window_cycles']==80 and good[0]['outside_command_window_cycles']==20
 assert good[0]['nodes']==[dict(node=0,commands=1,window_cycles=30),dict(node=1,commands=1,window_cycles=50)]
 assert not good[1]['committed'] and good[1]['outside_command_window_cycles'] is None
 cases=[]
 def rejected(name,records,commits):
  try:analyze(records,commands,commits)
  except AssertionError:cases.append(name);return
  raise AssertionError('Missed corrupt evidence: '+name)
 rejected('missing_command',[rows[0]],{7:100})
 rejected('duplicate_command',[rows[0],rows[0]],{7:100})
 rejected('missing_frame',rows,{7:100,9:100})
 rejected('underreported_counter',rows,{7:79})
 bad=copy.deepcopy(rows);bad[1].update(begin_cycle=120,end_cycle=170)
 rejected('overlap',bad,{7:100})
 bad=copy.deepcopy(rows);bad[0]['elapsed_cycles']=29
 rejected('wrong_elapsed',bad,{7:100})
 result=dict(status='COMMAND_PROFILE_ACCOUNTING_PASS',positive_complete_and_inflight=True,rejected=cases,
  analyzer_sha256=sha(Path(__file__).with_name('summarize_board_command_profile.py')),test_sha256=sha(__file__),real_board=False)
 (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
