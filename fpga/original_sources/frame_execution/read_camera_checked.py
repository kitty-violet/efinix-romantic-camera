"""Offline C6 camera payload decoder; use only with a bound camera release."""
import read_startup_cause

def decode(raw,slots):
 if len(slots)!=111 or len(set(slots))!=111 or not all(0<=s<128 for s in slots):raise ValueError('camera slot map')
 base=read_startup_cause.decode(raw)
 if not base['valid']:return base
 # OFP5 value = 64-bit header + 960-bit held payload; original cell window.
 wire=(raw>>2)>>896
 v=sum(((wire>>slot)&1)<<i for i,slot in enumerate(slots))
 config=v&((1<<64)-1);phase=config>>60;error=config>>56&15
 base.pop('cell_probes',None)
 errors=['RAW marker','frame ID','feeder','debayer','RGB pair overflow','reserved',
         'line reservation/FIFO','RAW frame length','RAW overflow','RAW packer protocol','ingress FIFO']
 raw_errors=v>>99&0x7ff
 base['camera']=dict(schema='C6-packed-111',phase=phase,
   state={0:'waiting_or_not_yet_terminal',1:'chip_id_high',2:'chip_id_low',3:'configuration_write',4:'configuration_readback',15:'verified'}.get(phase,'invalid'),
   error_code=error,error={0:'none',1:'slave_nack',2:'timeout',3:'arbitration_lost',4:'chip_id_mismatch',5:'readback_mismatch',6:'illegal_state'}.get(error,'unknown'),
   config_verified=phase==15 and error==0,transaction_index=config>>48&255,
   register=f'0x{config>>32&65535:04x}',actual=config>>24&255,expected=config>>16&255,
   chip_id=f'0x{config&65535:04x}',
   first_complete_line_words=v>>64&65535,first_complete_frame_lines=v>>80&65535,
   seen_pixel_valid=bool(v>>96&1),seen_hsync=bool(v>>97&1),seen_vsync_transition=bool(v>>98&1),
   raw_error_bits=raw_errors,raw_errors=[name for i,name in enumerate(errors) if raw_errors>>i&1],boundary_error=bool(v>>110&1),
   counts_are_zero_until_captured=True,sticky_bits_independently_synchronized=True)
 return base
