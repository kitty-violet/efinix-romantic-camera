"""Attribute measured command windows; keep their waits distinct from pure MAC work."""
import argparse,collections,csv,json,re
from pathlib import Path
from run_units import sha

def analyze(records,commands,commits):
 expected=[c['program_index'] for c in commands];by_pc={c['program_index']:c for c in commands}
 assert len(by_pc)==len(expected),'Duplicate descriptor program index'
 frames=collections.defaultdict(list)
 for row in records:
  row={key:int(value) for key,value in row.items()}
  assert row['end_cycle']>=row['begin_cycle'] and row['elapsed_cycles']==row['end_cycle']-row['begin_cycle']
  frames[row['frame']].append(row)
 output=[]
 for frame,rows in sorted(frames.items()):
  indices=[r['program_index'] for r in rows]
  assert indices==expected[:len(indices)],f'Command missing/duplicated/reordered in frame {frame}'
  assert all(b['begin_cycle']>=a['end_cycle'] for a,b in zip(rows,rows[1:])),f'Overlapping windows {frame}'
  complete=frame in commits
  if complete:assert indices==expected,f'Committed frame has incomplete command evidence {frame}'
  nodes={}
  for row in rows:
   node=by_pc[row['program_index']]['node'];dst=nodes.setdefault(node,dict(commands=0,window_cycles=0))
   dst['commands']+=1;dst['window_cycles']+=row['elapsed_cycles']
  total=sum(x['window_cycles'] for x in nodes.values());hardware=commits.get(frame)
  if complete:assert total<=hardware,f'Command windows exceed frame counter {frame}'
  output.append(dict(frame=frame,committed=complete,completed_command_count=len(rows),
   command_window_cycles=total,production_counter_cycles=hardware,
   outside_command_window_cycles=hardware-total if complete else None,
   nodes=[dict(node=node,**values) for node,values in sorted(nodes.items())]))
 assert set(commits)<=set(frames),'Committed frame absent from timing CSV'
 return output

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();root=a.root.resolve()
 run=json.loads((root/'run_result.json').read_text())
 raw=b''.join(int(word,16).to_bytes(16,'little') for word in (root/'compute.hex').read_text().split())
 assert len(raw)==8789*64
 # Bit positions are the actual frozen frame_program_issue interface:
 # debug_program_index=cdata[28:14], layer/node=cdata[33:29].
 commands=[]
 for start in range(0,len(raw),64):
  word=int.from_bytes(raw[start:start+64],'little')
  commands.append(dict(program_index=(word>>14)&32767,node=(word>>29)&31))
 with (root/'cnn_command_events.csv').open(newline='') as stream:records=list(csv.DictReader(stream))
 commits={}
 for line in run['events']:
  if 'CNN_COMMIT' not in line:continue
  fields=dict(re.findall(r'(\w+)=([^\s,]+)',line));frame=int(fields['frame'])
  assert frame not in commits;commits[frame]=int(fields['cycles'])
 frames=analyze(records,commands,commits)
 result=dict(status='COMMAND_PROFILE_RECONCILED' if commits else 'NO_COMMITTED_FRAME_PROFILE',frames=frames,
  inputs_sha256=run['inputs_sha256'],run_sha256=sha(root/'run_result.json'),csv_sha256=sha(root/'cnn_command_events.csv'),
  descriptor_hex_sha256=sha(root/'compute.hex'),
  note='Command windows include their internal data/weight/result stalls. DMA overlaps them; do not call these pure MAC cycles or add all spans to end-to-end latency.')
 (root/'command_profile.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
