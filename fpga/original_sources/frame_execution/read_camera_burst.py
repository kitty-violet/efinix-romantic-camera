"""Decode only the new release's six capture progress bits; no hardware IO."""
from read_camera_checked import decode as decode_config
from read_camera_recovery import context
PROGRESS=('down_valid_seen','capture_started','aw_accepted','w_accepted','frame_done','frame_failed')
def decode(raw,slots):
 result=decode_config(raw,slots)
 if not result['valid']:return result
 wire=(raw>>2)>>896;v=sum(((wire>>slot)&1)<<i for i,slot in enumerate(slots))
 if v>>60&15!=14:
  result['camera']['schema']='C7-config-page';return result
 result.pop('camera');flags=v>>92&7
 result['camera_stream']=dict(schema='C7-stream-capture-progress',
  first_boundary=context(v&0xffffffff,True) if flags&1 else None,
  first_length=context(v>>32&0xfffffff,False) if flags&2 else None,
  first_complete_source_words=v>>64&0xfffff if flags&4 else None,
  observed_frame_end_events_mod256=v>>84&255,
  capture_progress={name:bool(v>>(95+i)&1) for i,name in enumerate(PROGRESS)},
  capture_active=bool(v>>101&1),owner_states=[v>>(102+3*i)&7 for i in range(3)],
  first_faults_held_until_reset=True,word_count_is_packer_at_offending_cycle=True,
  frame_activity_may_merge_edges_shorter_than_cdc_latency=True)
 return result
