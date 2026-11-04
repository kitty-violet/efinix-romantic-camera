"""OFP5 read helper; device access occurs only when explicitly run as a script."""
import time
from read_startup_cause import decode,SCAN_BITS
from read_control_snapshot import SnapshotPending
def fresh(transport,history,save,seconds=1):
 deadline=time.monotonic()+seconds
 def scan(selector):
  r=decode(transport.scan(selector,SCAN_BITS));history.append(dict(selector=selector,**r));save();return r
 old=scan(3)
 for _ in range(8):
  if not old['busy'] or time.monotonic()>=deadline:break
  old=scan(3)
 if old['busy']:raise SnapshotPending('Previous OFP5 request pending')
 scan(2)
 for _ in range(8):
  if time.monotonic()>=deadline:break
  now=scan(3)
  if now['valid'] and not now['busy'] and now['sequence']==((old['sequence']+1)&255):return now
 raise SnapshotPending('No fresh OFP5 snapshot; valid prefix retained')
if __name__=='__main__':
 import read_first_fault as worker
 worker.fresh=fresh;worker.__file__=__file__;worker.main()
