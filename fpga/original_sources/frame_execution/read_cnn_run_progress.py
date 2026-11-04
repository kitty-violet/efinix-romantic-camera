"""Decode D run-progress page, preserving C7 config/stream before first run."""
from read_camera_burst import decode as previous_decode
def decode(raw,slots):
 result=previous_decode(raw,slots)
 if not result['valid']:return result
 wire=(raw>>2)>>896;v=sum(((wire>>slot)&1)<<i for i,slot in enumerate(slots))
 if v>>60&15!=13:return result
 result.pop('camera',None);result.pop('camera_stream',None)
 result['cnn_run']=dict(schema='D-run-progress-111',cycles=v&0x3ffffff,
  compute_completed=v>>26&0x3fff,dma_completed=v>>40&0x3fff,
  compute_state=v>>54&7,header_state=v>>57&3,dma_waiting=bool(v>>59&1),
  nodes_completed=v>>64&0x1fffffff,owner_states=[v>>(93+3*i)&7 for i in range(3)],
  frames_admitted=v>>102&31,control_enable=bool(v>>107&1),busy=bool(v>>108&1),
  program_crc_verified=bool(v>>109&1),issuer_error=bool(v>>110&1),
  sample_is_live_or_stopped_counter=True,completed_frame_latency_not_implied=True)
 return result
