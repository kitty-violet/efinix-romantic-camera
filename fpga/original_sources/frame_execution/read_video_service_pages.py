"""Pure decoder for the retained JTAG transport; does not access hardware."""
from read_continuous_progress import decode as legacy_decode

PROFILE='video-service-pages-v1'
def decode_v1(raw,positions,camera_slots):
 if not isinstance(raw,int) or isinstance(raw,bool) or not 0<=raw<1<<1026:raise ValueError('scan length')
 if len(positions)!=567 or len(set(positions))!=567 or any(not 0<=p<960 for p in positions):raise ValueError('manifest positions')
 value=raw>>2
 if value&0xffffffff!=0x4f465035 or value>>40&255!=5:raise ValueError('OFP5 signature/version')
 valid=bool(value>>32&1);busy=bool(value>>33&1)
 if busy!=bool((value>>34^value>>35)&1):raise ValueError('handshake mismatch')
 out=dict(diagnostic_profile=PROFILE,valid=valid,busy=busy,sequence=value>>56&255)
 if not valid:return out
 payload=value>>64
 packet=sum(((payload>>p)&1)<<i for i,p in enumerate(positions))
 tag=packet>>512&0xffffffff
 if tag&0xfffffff8!=0x56530000:
  result=legacy_decode(raw,camera_slots);result['diagnostic_profile']=PROFILE;result['page_kind']='legacy';return result
 phase=tag&7
 if phase<2 or packet>>552&255!=1 or packet>>563:raise ValueError('service page schema/reserved')
 out.update(page_kind={3:'process',5:'health',7:'path'}.get(phase,'display'),page_phase=phase,
  page_sequence=packet>>544&255,pixel_cache_valid=bool(packet>>560&1),
  pixel_cache_busy=bool(packet>>561&1),pixel_cache_stale=bool(packet>>562&1))
 w=[packet>>(i*32)&0xffffffff for i in range(16)]
 if phase%2==0:
  if not out['pixel_cache_valid']:out['display']=None;return out
  if w[15]!=0x56494431:raise ValueError('display page magic')
  out['display']=dict(ticks=w[14],physical_frames=w[13],complete_frames=w[12],distinct_frames=w[11],
   repeated_frames=w[10],blank_frames=w[9],underflows=w[8],marker_errors=w[7],fifo_errors=w[6],
   first_fault=dict(x=w[5]&65535,y=w[5]>>16,valid=bool(w[4]>>24&1),code=w[4]>>16&255,
    frame_id=w[4]&65535,tick=w[3],epoch=w[2]),fifo_low=w[1]&8191,fifo_high=w[1]>>16&8191,status=w[0],
   measurement_valid=not out['pixel_cache_stale'])
 elif phase==5:
  if w[15]!=0x484c5431:raise ValueError('health page magic')
  out['health']=dict(status=w[14],reset_epochs=w[13]>>16,calibration_losses=w[13]&65535,
   source_area_frames=w[12],source_interval_cycles=w[11],commit_stall_cycles=w[10],lock_loss_events=w[9]&65535,
   first_tick=w[8],first_context=w[4:8],live_context=w[:4],
   first_valid=bool(w[14]>>17&1),first_faults=w[14]>>9&255)
 elif phase==7:
  if w[15]!=0x50415431:raise ValueError('path page magic')
  mode=w[14]&3
  out['path']=dict(mode=['LIVE_A','INPUT_PREVIEW','DDR_PATTERN','FROZEN_A'][mode],inspect_enabled=bool(w[14]&4),
   inspect_result=bool(w[14]&8),inspect_bank=w[14]>>4&3,capture_id=w[13]&65535,source_id=w[12]&65535,
   reader_address=w[11],reader_line_offset=w[10],capture_xy=w[9],reader_xy=w[8],scaler_xy=w[7],
   read_arbiter=w[6],write_arbiter=w[5],pipeline_flags=w[4],rejected_mode_changes=w[3],selected_rgb=w[2]&0xffffff,
   h_total=w[1]>>16,v_total=w[1]&65535,maximum_read_no_progress_cycles=w[0]>>16,
   maximum_write_no_progress_cycles=w[0]&65535,gaps_include_backpressure=True,gaps_saturate_at=65535,
   counts_are_ai_results=mode in (0,3))
 else:
  if w[15]!=0x53525631:raise ValueError('process page magic')
  out['process']=dict(ticks=w[14],recoveries=w[13],recovery_status=w[12],captured=w[11],skipped=w[10],
   orphan_pixels=w[9],compute_admitted=w[8],committed=w[7],repeated_reads=w[6],dropped_results=w[5],
   pixel_cache_sequence=w[4],pixel_cache_timeouts=w[3],rejected_buttons=w[2],
   owner_states=[w[1]>>(i*3)&7 for i in range(3)],status=w[0])
 return out


def decode(raw,positions,camera_slots):
 """v2/v3 return one atomic chunk; a collected page is not simultaneous."""
 if len(positions)==567:return decode_v1(raw,positions,camera_slots)
 if len(positions)==192:return decode_compact(raw,positions,camera_slots)
 if len(positions)==96:return decode_narrow(raw,positions,camera_slots)
 if len(positions) not in (183,87) or len(set(positions))!=len(positions) or any(type(p) is not int or not 0<=p<960 for p in positions):raise ValueError('manifest positions')
 width,parts,version=(32,16,3) if len(positions)==87 else (128,4,2)
 if not isinstance(raw,int) or isinstance(raw,bool) or not 0<=raw<1<<1026:raise ValueError('scan length')
 value=raw>>2
 if value&0xffffffff!=0x4f465035 or value>>40&255!=5:raise ValueError('OFP5 signature/version')
 valid=bool(value>>32&1);busy=bool(value>>33&1)
 if busy!=bool((value>>34^value>>35)&1):raise ValueError('handshake mismatch')
 out=dict(diagnostic_profile='video-service-pages-v'+str(version),valid=valid,busy=busy,sequence=value>>56&255)
 if not valid:return out
 payload=value>>64
 packet=sum(((payload>>p)&1)<<i for i,p in enumerate(positions))
 tag=packet>>width&0xffffffff
 if tag&(~(parts*8-1)&0xffffffff)!=0x56530000:
  result=legacy_decode(raw,camera_slots);result['diagnostic_profile']=out['diagnostic_profile'];result['page_kind']='legacy';return result
 phase=tag&7;part=tag>>3&(parts-1)
 if phase not in (2,3,4,5) or packet>>(width+40)&255!=version or packet>>(width+51):raise ValueError('service page schema/reserved')
 out.update(page_kind={2:'display',3:'process',4:'health',5:'path'}[phase],page_phase=phase,chunk_index=part,
  page_sequence=packet>>(width+32)&255,pixel_cache_valid=bool(packet>>(width+48)&1),pixel_cache_busy=bool(packet>>(width+49)&1),
  pixel_cache_stale=bool(packet>>(width+50)&1),chunk_words=[packet>>(i*32)&0xffffffff for i in range(width//32)],
  chunk_bits=width,chunks_per_page=parts,chunk_atomic=True,whole_page_atomic=False)
 out['measurement_valid']=phase!=2 or out['pixel_cache_valid'] and not out['pixel_cache_stale']
 return out


def decode_compact(raw,positions,camera_slots):
 if positions!=list(range(192)):raise ValueError('compact manifest positions')
 if not isinstance(raw,int) or isinstance(raw,bool) or not 0<=raw<1<<1026:raise ValueError('scan length')
 value=raw>>2;header=value&((1<<64)-1)
 if header&0xffffffff!=0x4f465035 or header>>40&255!=5:raise ValueError('OFP5 signature/version')
 valid=bool(header>>32&1);busy=bool(header>>33&1)
 if busy!=bool((header>>34^header>>35)&1):raise ValueError('handshake mismatch')
 out=dict(diagnostic_profile='video-service-pages-v4',valid=valid,busy=busy,sequence=header>>56&255,transport_header=header)
 if not valid:return out
 packet=value>>64
 if packet>>192:raise ValueError('compact padding not zero')
 if packet>>168&255!=4 or packet>>179:raise ValueError('compact schema/reserved')
 tag=packet>>128&0xffffffff
 out.update(page_sequence=packet>>160&255,chunk_words=[packet>>(i*32)&0xffffffff for i in range(4)],chunk_bits=128,chunk_atomic=True,whole_page_atomic=False)
 if tag&0xfffffff8==0x4c4f0000:
  part=tag&7
  if packet>>176:raise ValueError('legacy chunk flags')
  if part==7 and (out['chunk_words'][2] or out['chunk_words'][3]):raise ValueError('legacy chunk padding')
  out.update(page_kind='legacy_chunk',chunk_index=part,chunks_per_page=8,measurement_valid=True)
  return out
 phase=tag&7;part=tag>>3&3
 if tag&0xffffffe0!=0x56530000 or phase not in (2,3,4,5):raise ValueError('compact page tag')
 out.update(page_kind={2:'display',3:'process',4:'health',5:'path'}[phase],page_phase=phase,chunk_index=part,chunks_per_page=4,
  pixel_cache_valid=bool(packet>>176&1),pixel_cache_busy=bool(packet>>177&1),pixel_cache_stale=bool(packet>>178&1))
 out['measurement_valid']=phase!=2 or out['pixel_cache_valid'] and not out['pixel_cache_stale']
 return out


def decode_narrow(raw,positions,camera_slots):
 if positions!=list(range(96)):raise ValueError('narrow manifest positions')
 if not isinstance(raw,int) or isinstance(raw,bool) or not 0<=raw<1<<1026:raise ValueError('scan length')
 value=raw>>2;header=value&((1<<64)-1)
 if header&0xffffffff!=0x4f465035 or header>>40&255!=5:raise ValueError('OFP5 signature/version')
 valid=bool(header>>32&1);busy=bool(header>>33&1)
 if busy!=bool((header>>34^header>>35)&1):raise ValueError('handshake mismatch')
 out=dict(diagnostic_profile='video-service-pages-v5',valid=valid,busy=busy,sequence=header>>56&255,transport_header=header)
 if not valid:return out
 packet=value>>64
 if packet>>96:raise ValueError('narrow padding not zero')
 version=packet>>72&255
 if version not in (5,6) or packet>>83:raise ValueError('narrow schema/reserved')
 out['diagnostic_profile']='video-service-pages-v'+str(version)
 tag=packet>>32&0xffffffff;word=packet&0xffffffff
 out.update(page_sequence=packet>>64&255,chunk_words=[word],chunk_bits=32,chunk_atomic=True,whole_page_atomic=False)
 if tag&0xffffffe0==0x4c4f0000:
  part=tag&31
  if packet>>80:raise ValueError('legacy word flags')
  if part>=30 and word:raise ValueError('legacy word padding')
  out.update(page_kind='legacy_chunk',chunk_index=part,chunks_per_page=32,measurement_valid=True)
  return out
 phase=tag&7;part=tag>>3&15
 if tag&0xffffff80!=0x56530000 or phase not in ((2,3,4,5,6) if version==6 else (2,3,4,5)):raise ValueError('narrow page tag')
 out.update(page_kind={2:'display',3:'process',4:'health',5:'path',6:'input_isp'}[phase],page_phase=phase,chunk_index=part,chunks_per_page=16,
  pixel_cache_valid=bool(packet>>80&1),pixel_cache_busy=bool(packet>>81&1),pixel_cache_stale=bool(packet>>82&1))
 out['measurement_valid']=phase!=2 or out['pixel_cache_valid'] and not out['pixel_cache_stale']
 return out


class PageAssembler:
 """Collect four/sixteen chunks in order; preserve their individual tags.

 Do not use a mixed-time page as a point-in-time multi-field snapshot. First
 fault records remain stable until explicitly cleared; live words may differ.
 """
 def __init__(self,camera_slots=None):self.pages={};self.camera_slots=camera_slots or []
 def add(self,chunk):
  kind=chunk.get('page_kind')
  if kind not in ('display','process','health','path','input_isp','legacy_chunk') or 'chunk_index' not in chunk:return None
  if not chunk.get('valid') or not chunk.get('measurement_valid'):
   self.pages.pop(kind,None);return None
  part=chunk['chunk_index']
  if part==0:self.pages[kind]=[]
  parts=self.pages.get(kind)
  if parts is None:return None
  if parts and chunk.get('diagnostic_profile')!=parts[0].get('diagnostic_profile'):
   self.pages.pop(kind,None);return None
  if part!=len(parts):
   self.pages.pop(kind,None);return None
  parts.append(chunk.copy())
  if len(parts)!=chunk.get('chunks_per_page',4):return None
  del self.pages[kind]
  words=[w for c in parts for w in c['chunk_words']]
  if kind=='legacy_chunk':
   # The camera/CNN subpage spans chunks6/7. Their schema choice must be
   # identical even though live counters are not a simultaneous snapshot.
   if chunk['diagnostic_profile'] in ('video-service-pages-v5','video-service-pages-v6'):
    first=parts[0]['page_sequence']
    # One complete96-request sweep has one fixed legacy epoch. Reject
    # dropped/mixed sweeps, including when the8-bit sequence wraps.
    phases=7 if chunk['diagnostic_profile']=='video-service-pages-v6' else 6
    if any((c['page_sequence']-first)&255 != phases*(i//2)+i%2 for i,c in enumerate(parts)):return None
   else:
    s6,s7=parts[6]['page_sequence'],parts[7]['page_sequence']
    if (s7-s6)&255!=1 or (s6^s7)&0x18:return None
   payload=sum(w<<(i*32) for i,w in enumerate(words));assert payload>>960==0
   raw=((payload<<64)|chunk['transport_header'])<<2
   fields=legacy_decode(raw,self.camera_slots)
   return dict(page_kind='legacy',fields=fields,whole_page_atomic=False,chunk_atomic=True,
    chunks=[{k:c[k] for k in ('chunk_index','page_sequence','sequence','measurement_valid')} for c in parts],
    note=str(len(parts))+' individually sampled legacy chunks; fields spanning chunks are not atomic. Use direct APB reads or a coherent service status word for control decisions.')
  if kind=='input_isp':
   if words[15]!=0x49535031 or any(words[4:15]):raise ValueError('input ISP schema/padding')
   flags=words[1]
   fields=dict(accepted_crc32=f'{words[3]:08x}',read_index=words[2]>>24,
    read_rgb=[words[2]>>(8*i)&255 for i in range(3)],write_crc_internal=f'{words[0]:08x}',
    rejected=flags>>16,count=flags&1023,valid=bool(flags&1024),updating=bool(flags&2048),
    swap_rb=bool(flags&4096),read_pending=bool(flags&8192),read_request=bool(flags&16384))
   return dict(page_kind=kind,fields=fields,whole_page_atomic=False,chunk_atomic=True,
    chunks=[{k:c[k] for k in ('chunk_index','page_sequence','sequence','measurement_valid')} for c in parts],
    note='Stopped configuration state only; require stable matching index/status around readback.')
  phase={'display':2,'process':3,'health':5,'path':7}[kind]
  # Reuse the field layout decoder, with an explicitly constructed v1 envelope.
  packet=sum(w<<(i*32) for i,w in enumerate(words))|((0x56530000|phase)<<512)|(1<<552)|(1<<560)
  raw=((packet<<64)|0x4f465035|(5<<40)|(1<<32))<<2
  fields=decode_v1(raw,list(range(567)),[])[kind]
  return dict(page_kind=kind,fields=fields,whole_page_atomic=False,chunk_atomic=True,
   chunks=[{k:c[k] for k in ('chunk_index','page_sequence','sequence','measurement_valid')} for c in parts],
   note='Each '+str(chunk.get('chunk_bits',128))+'-bit chunk was sampled separately; live fields are not one simultaneous snapshot.')
