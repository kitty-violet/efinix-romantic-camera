"""Keep the legacy C/E/D page choice stable across its two compact chunks."""
from video_service_compact_native import extension as compact
from video_service_serializer_native import P,net
from camera_joint_native import OUT

def extension(D,removed,edits,supplied):
 code,receipt=compact(D,removed,edits,supplied)
 request=net(P+'request_sync');epoch=net('service_page_sequence',3)
 handshake={('LUT__159763','I1'),(P+'acknowledge~FF','D')}
 targets=set(D['loads'][request])-handshake
 assert len(targets)==118 and handshake<=set(D['loads'][request])
 # Prove that every changed signal ends only at a legacy observation D pin.
 # No functional camera/CNN/controller state can receive this changed choice.
 todo=[t for n,p in targets for q,ts in D['g'][n][1].items() if q in OUT[D['g'][n][0]] for t in ts]
 seen=set();ends=set()
 while todo:
  t=todo.pop()
  if t in seen:continue
  seen.add(t)
  for n,p in D['loads'].get(t,set()):
   k,pins=D['g'][n]
   if k=='EFX_FF':
    assert n.startswith(P+'held_payload[') and p=='D',(n,p)
    ends.add(n);continue
   assert k in ('EFX_LUT4','EFX_ADD'),(n,k)
   todo.extend(q for pin,ts in pins.items() if pin in OUT[k] for q in ts)
 assert len(ends)==111
 for n,p in targets:
  assert n not in removed and request in D['g'][n][1][p]
  assert request not in edits.get(n,{}).get(p,{})
  edits.setdefault(n,{}).setdefault(p,{})[request]=epoch
 gate='LUT__246809';assert (gate,'I0') in targets
 old=net('cnn_run_seen');assert D['g'][gate][1]['I1']==[old]
 edits.setdefault(gate,{}).setdefault('I1',{})[old]=net('service_page_sequence',4)
 receipt.update(legacy_camera_selector='service_page_sequence[3]',legacy_progress_qualifier='service_page_sequence[4]',
  selector_loads=sorted(targets),selector_cone_only_observation=True,selector_endpoints=sorted(ends),
  legacy_camera_chunks=[6,7],legacy_camera_cycle='C/E/C/D; same choice across both chunks, independent of CNN starting between reads',
  camera_counter_fields_whole_page_atomic=False)
 return code,receipt
