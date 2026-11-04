"""Bounded read-only helpers; caller owns the released binding and transport."""
import time
from read_control_snapshot import SnapshotPending
from read_video_service_pages import decode

def path_status(row):
 if row.get('page_kind')!='path' or not row.get('measurement_valid'):return None
 narrow=row.get('diagnostic_profile') in ('video-service-pages-v5','video-service-pages-v6')
 if row.get('chunk_index')!=(14 if narrow else 3):return None
 words=row['chunk_words']
 if narrow:
  if len(words)!=1 or words[0]>>6:raise ValueError('Path word schema')
  word=words[0]
 else:
  if len(words)!=4 or words[3]!=0x50415431:raise ValueError('Path signature')
  word=words[2]
 return dict(mode=word&3,inspect_enabled=bool(word&4),inspect_result=bool(word&8),inspect_bank=(word>>4)&3,chunk_sequence=row['sequence'])

def fresh(transport,positions,camera_slots,history,save,*,now=time.monotonic):
 deadline=now()+1.0
 def scan(selector):
  if now()>=deadline:raise SnapshotPending('Service snapshot deadline')
  raw=transport.scan(selector,1026)
  row=decode(raw,positions,camera_slots)
  history.append(dict(selector=selector,**row));save()
  return row
 old=scan(3)
 for _ in range(8):
  if not old['busy']:break
  old=scan(3)
 if old['busy']:raise SnapshotPending('Previous service snapshot still pending')
 scan(2)
 for _ in range(8):
  row=scan(3)
  if row['valid'] and not row['busy'] and row['sequence']==((old['sequence']+1)&255):return row
 raise SnapshotPending('Fresh service snapshot unavailable; no command replay')

def prepared(transport,positions,camera_slots,history,save,*,now=time.monotonic):
 # Each accepted chunk contains both readiness and a signature atomically.
 word_schema=len(positions) in (87,96)
 wanted_part=12 if word_schema else 3
 deadline=now()+30.0
 for _ in range(112 if len(positions)==96 else 96 if word_schema else 24):
  if now()>=deadline:raise SnapshotPending('Prepared observation deadline')
  row=fresh(transport,positions,camera_slots,history,save,now=now)
  if row.get('page_kind')!='process' or row.get('chunk_index')!=wanted_part:continue
  if not row.get('measurement_valid') or not row.get('chunk_atomic'):raise ValueError('Invalid prepared observation')
  words=row['chunk_words']
  if word_schema:
   if len(positions)==96:
    if len(words)!=1 or words[0]>>13 or row.get('diagnostic_profile') not in ('video-service-pages-v5','video-service-pages-v6'):raise ValueError('Prepared word schema')
   elif len(words)!=1 or words[0]>>16!=0x5352 or words[0]&0xe000:raise ValueError('Prepared word signature')
  elif len(words)!=4 or words[3]!=0x53525631:raise ValueError('Prepared page signature')
  return bool(words[0]&1)
 raise SnapshotPending('One complete phase sweep lacked prepared status')
