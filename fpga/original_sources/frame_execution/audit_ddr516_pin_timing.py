"""Independent ACT/PRE timing audit of observed DDR command pins."""
import argparse,csv,json
from pathlib import Path
from run_units import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
 root=a.root.resolve();path=root/'pin_commands.csv';active={};last_read={};intervals=[];violations=[];refreshes=0;commands=0;partial_rows=0
 with path.open() as f:
  for row in csv.DictReader(f):
   if not all(row.values()):
    partial_rows+=1;continue # an interrupted run may have a partial final record
   t=float(row['time_ns']);bank=int(row['bank']);addr=int(row['address'],16);cmd=row['ras_cas_we'];commands+=1
   if cmd=='011':active[bank]=t;last_read.pop(bank,None)
   elif cmd=='101':last_read[bank]=t
   elif cmd=='010':
    for b in list(active):
     if addr&1024 or bank==b:
      interval=dict(bank=b,activate_ns=active.pop(b),precharge_ns=t,read_ns=last_read.get(b))
      interval['act_to_pre_ns']=round(t-interval['activate_ns'],3);intervals.append(interval)
      if interval['act_to_pre_ns']<34.930:violations.append(interval)
   elif cmd=='001':refreshes+=1
 assert intervals
 result=dict(status='PIN_TRAS_TIMING_PASS' if not violations else 'PIN_TRAS_TIMING_FAIL',
  scope='observed ACT/PRE only; not calibration/data/full DDR pass',commands=commands,refreshes=refreshes,
  closed_banks=len(intervals),partial_rows=partial_rows,minimum_act_to_pre_ns=min(x['act_to_pre_ns'] for x in intervals),
  minimum_read_row_act_to_pre_ns=min((x['act_to_pre_ns'] for x in intervals if x['read_ns'] is not None),default=None),
  micron_tras_min_ns=35.0,configured_tras_ns=37.5,violations=violations,
  shortest_intervals=sorted(intervals,key=lambda x:x['act_to_pre_ns'])[:5],pin_csv_sha256=sha(path),board_release=False)
 (root/'pin_timing_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
