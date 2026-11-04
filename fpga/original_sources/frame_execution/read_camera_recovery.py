"""Decode alternating configuration and stream-first-fault snapshots."""
from read_camera_checked import decode as decode_config
def context(value,wide):
 pins=value>>20
 out=dict(packer_words=value&0xfffff,previous_vsync=bool(pins>>1&1),vsync=bool(pins>>2&1),
  hsync=bool(pins>>3&1),data_enable=bool(pins>>4&1),pixel_valid=bool(pins>>5&1),
  frame_end=bool(pins>>6&1),frame_start=bool(pins>>7&1))
 if wide:out.update(active_vsync_level=bool(pins>>8&1),polarity_known=bool(pins>>9&1),
  frame_active=bool(pins>>10&1),packer_ready=bool(pins>>11&1))
 return out
def decode(raw,slots):
 result=decode_config(raw,slots)
 if not result['valid']:return result
 wire=(raw>>2)>>896;v=sum(((wire>>slot)&1)<<i for i,slot in enumerate(slots))
 if v>>60&15!=14:
  result['camera']['schema']='C7-config-page';return result
 if v>>95&63:raise ValueError('C7 stream reserved bits nonzero')
 result.pop('camera');flags=v>>92&7
 result['camera_stream']=dict(schema='C7-stream-page',
  first_boundary=context(v&0xffffffff,True) if flags&1 else None,
  first_length=context(v>>32&0xfffffff,False) if flags&2 else None,
  first_complete_source_words=v>>64&0xfffff if flags&4 else None,
  observed_frame_end_events_mod256=v>>84&255,
  capture_active=bool(v>>101&1),owner_states=[v>>(102+3*i)&7 for i in range(3)],
  first_faults_held_until_reset=True,word_count_is_packer_at_offending_cycle=True,
  frame_activity_may_merge_edges_shorter_than_cdc_latency=True)
 return result
