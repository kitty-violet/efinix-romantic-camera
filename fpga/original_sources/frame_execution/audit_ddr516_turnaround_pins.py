"""Audit observed command-pin write/read transitions, independent of data checks."""
import argparse,csv,json
from collections import Counter
from decimal import Decimal
from pathlib import Path
from run_units import sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
 p.add_argument('--ck-ns',type=Decimal,default=Decimal('2.5'))
 p.add_argument('--write-latency',type=int,default=5)
 p.add_argument('--burst-length',type=int,default=8)
 p.add_argument('--twtr-ck',type=int,default=4);a=p.parse_args()
 path=a.root/'pin_commands.csv';last_write=None;last_access=None
 minimum=None;violations=[];commands=Counter();bank_pairs=Counter();turns=0
 active_rows={};minimum_by_relationship={};relationship_counts=Counter()
 # Trace has 1 ps precision. Do not interpret trace rounding as a CK shortage.
 tolerance=Decimal('0.002');required=a.write_latency+a.burst_length//2+a.twtr_ck
 with path.open() as f:
  for row in csv.DictReader(f):
   assert all(row.values()),'Partial pin trace cannot pass an audit'
   stamp=Decimal(row['time_ns']);cmd=row['ras_cas_we'];bank=int(row['bank'])
   commands[cmd]+=1
   if cmd=='011':active_rows[bank]=int(row['address'],16)
   if cmd=='100':last_write=(stamp,bank,row['address'],active_rows.get(bank))
   if cmd=='101' and last_write is not None:
    gap=stamp-last_write[0]
    interval=dict(write_ns=str(last_write[0]),read_ns=str(stamp),write_bank=last_write[1],
     read_bank=bank,write_address=last_write[2],read_address=row['address'],gap_ns=str(gap),
     gap_ck=float(gap/a.ck_ns),required_ck=required)
    relationship=('same_bank_' if last_write[1]==bank else 'other_bank_')+(
     'unknown_row' if last_write[3] is None or active_rows.get(bank) is None else
     'same_row' if last_write[3]==active_rows[bank] else 'other_row')
    interval.update(write_row=last_write[3],read_row=active_rows.get(bank),relationship=relationship)
    if relationship not in minimum_by_relationship or gap<Decimal(minimum_by_relationship[relationship]['gap_ns']):
     minimum_by_relationship[relationship]=interval
    if minimum is None or gap<Decimal(minimum['gap_ns']):minimum=interval
    if gap+tolerance<required*a.ck_ns:violations.append(interval)
    if last_access=='100':
     turns+=1;bank_pairs[str(last_write[1])+'->'+str(bank)]+=1
     relationship_counts[relationship]+=1
   if cmd in ('100','101'):last_access=cmd
 assert turns and minimum,'No WRITE-to-READ coverage'
 result=dict(status='PIN_TWTR_FAIL' if violations else 'PIN_TWTR_PASS',
  scope='Observed command-pin tWTR only; not DDR data, calibration or whole-board pass',
  clock_ns=str(a.ck_ns),write_latency=a.write_latency,burst_length=a.burst_length,
  twtr_ck=a.twtr_ck,additive_latency=0,commands=dict(commands),write_read_turns=turns,
  bank_pairs=dict(bank_pairs),minimum=minimum,relationship_counts=dict(relationship_counts),
  minimum_by_relationship=minimum_by_relationship,violations=violations,pin_sha256=sha(path))
 (a.root/'pin_turnaround_audit.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result),flush=True)
 if violations:raise SystemExit(1)
if __name__=='__main__':main()
