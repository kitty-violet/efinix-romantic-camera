"""OFP4 decoding only; importing this module never opens a device."""
import time,json
import read_control_ddr_progress as previous
from read_control_snapshot import SnapshotPending
MAGIC=0x4f465034
SCAN_BITS=770
def decode(raw):
 if not isinstance(raw,int) or isinstance(raw,bool) or not 0<=raw<1<<SCAN_BITS:raise ValueError('OFP4 scan length/value')
 value=raw>>2
 if value&0xffffffff!=MAGIC or value>>40&255!=4:raise ValueError('OFP4 signature/version')
 base=value&((1<<448)-1)
 base=(base&~(0xffffffff|(255<<40)))|previous.MAGIC|(3<<40)
 result=previous.decode(base<<2)
 result.update(format='OFP4',raw_hex=f'{value:0192x}',adapter=None,clock_probe=None)
 if result['valid']:
  a=value>>448&((1<<128)-1);c=value>>576
  def flags(v):
   return dict(raw=f'{v:08x}',read_state=v&7,write_state=v>>3&7,
    read_error=bool(v>>6&1),write_error=bool(v>>7&1),reset_n=bool(v>>8&1),
    source_arvalid=bool(v>>9&1),source_awvalid=bool(v>>10&1),source_wvalid=bool(v>>11&1),
    downstream_rvalid=bool(v>>12&1),downstream_bvalid=bool(v>>13&1),
    downstream_rlast=bool(v>>14&1),source_wlast=bool(v>>15&1),wstrb=v>>16&15,
    rresp=v>>20&3,bresp=v>>22&3,rid_nonzero=bool(v>>24&1),bid_nonzero=bool(v>>25&1),
    source_rready=bool(v>>26&1),source_bready=bool(v>>27&1),one_write_left=bool(v>>28&1),
    downstream_awready=bool(v>>29&1),downstream_arready=bool(v>>30&1),source_wready=bool(v>>31&1))
  result['adapter']=dict(live=flags(a>>96),captured=bool(a>>95&1),
   captured_error_mask=a>>88&3,tick=a>>64&0xffffff,address=f'{a>>32&0xffffffff:08x}',
   preceding_cycle=flags(a&0xffffffff),capture_scope='One cycle preceding observed sticky error; not an inferred root cause')
  result['clock_probe']=dict(counts={n:c>>(24*i)&0xffffff for i,n in enumerate(('core200','tdqss400','tac400','twd400'))},
   ticks=c>>96&0xffffff,divider=256,counter_modulus=1<<24,comparison_window_limit_s=2,
   divided_levels=c>>120&15,ddr_reset_n=bool(c>>124&1),cke=bool(c>>125&1),
   system_reset_n=bool(c>>126&1),age_or_event_saturated=bool(c>>127&1),
   reset_age=c>>128&65535,last_low=c>>144&65535,last_high=c>>160&65535,
   reset_rises=c>>176&255,reset_falls=c>>184&255,
   scope='Counts relative to process_clk150, not absolute oscillator or analog phase proof')
 return result

def frequency_report(records):
 if len(records)<2 or any(r.get('clock_probe') is None for r in (records[0],records[-1])):
  return dict(status='INSUFFICIENT_FRESH_SNAPSHOTS')
 a,b=(r['clock_probe'] for r in (records[0],records[-1]));mod=1<<24
 ticks=(b['ticks']-a['ticks'])%mod
 if not 32<=ticks<2*150000000/256:return dict(status='INVALID_COMPARISON_WINDOW',ticks=ticks)
 counts={n:(b['counts'][n]-a['counts'][n])%mod for n in a['counts']}
 return dict(status='RELATIVE_CLOCK_RATE_OBSERVED',ticks=ticks,counts=counts,
  assumed_process_clock_mhz=150,mhz_relative_to_process150={n:150*c/ticks for n,c in counts.items()},
  measurement_scope='Relative divided edge counts only; no analog phase or absolute reference proof')

def fresh(transport,history,save,seconds=1):
 deadline=time.monotonic()+seconds
 def scan(selector):
  r=decode(transport.scan(selector,SCAN_BITS));history.append(dict(selector=selector,**r));save();return r
 old=scan(3)
 for _ in range(8):
  if not old['busy'] or time.monotonic()>=deadline:break
  old=scan(3)
 if old['busy']:raise SnapshotPending('Previous OFP4 request pending')
 scan(2)
 for _ in range(8):
  if time.monotonic()>=deadline:break
  now=scan(3)
  if now['valid'] and not now['busy'] and now['sequence']==(old['sequence']+1)&255:return now
 raise SnapshotPending('No fresh OFP4 snapshot; valid prefix retained')

if __name__=='__main__':
 import read_first_fault as worker
 original=worker.worker
 def with_rates(url,report):
  result=original(url,report);result['relative_clock_rates']=frequency_report(result['snapshots'])
  report.write_text(json.dumps(result,indent=2),encoding='utf8');return result
 worker.fresh=fresh;worker.worker=with_rates;worker.__file__=__file__;worker.main()
