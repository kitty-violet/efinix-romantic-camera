"""Bounded continuous-run diagnostics; full commit count is read via APB."""
from read_cnn_observer_trim import decode as original
PROFILE='cnn-continuous-progress-v1'

def decode(raw,slots):
 result=original(raw,slots)
 result['diagnostic_profile']=PROFILE
 if 'cnn_run' in result:
  run=result['cnn_run']
  run['wire_dma_completed']=run['dma_completed']
  run['dma_completed']^=1<<13
  run['schema']='D-continuous-progress-v1'
  run['admission_saturated']=run['frames_admitted']==31
  run['admission_count_is_lower_bound']=run['admission_saturated']
  run['full_committed_frames_source']='APB +0x5c; diagnostic admission clears after stop'
 return result
