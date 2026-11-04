"""Same OFP5 atomic snapshots, fixed sparse capture -> inactive RAM bank."""
from video_service_serializer_native import extension as serial_cut,P,B,net
from native_local_drivers import render_token
from camera_joint_native import OUT

def extension(D,removed,edits,supplied):
 _,proof=serial_cut(D,removed,edits,supplied)
 g,ps,drivers,loads=(D[k] for k in ('g','ps','drivers','loads'))
 snapshots={n for n in g if n.startswith(P+'snapshot[')}
 held={n for n in g if n.startswith(P+'held_payload[')}
 assert len(snapshots)==len(held)==567
 # No retained functional path may depend on the removed TCK copy.
 for n in snapshots:
  assert g[n][0]=='EFX_FF'
  q=g[n][1]['Q'][0]
  assert all(c in removed for c,_ in loads[q]),(n,loads[q])
 removed.update(snapshots)
 for n in snapshots:edits.pop(n,None)
 publish=net(P+'n2000')
 # Remove the previous scan-dependent publication gate. Banking now protects
 # the ongoing scan, while sequence/ack are free to publish a completed bank.
 for n,pin in loads[publish]:
  if n in removed:continue
  assert edits[n][pin].pop(publish)==net('service_snapshot_publish')
  if not edits[n][pin]:edits[n].pop(pin)
  if not edits[n]:edits.pop(n)
 for n in held:
  pins=g[n][1]
  assert pins['CE']==[net(P+'n5')] and ps[n]['CE_POLARITY']==(1,1)
  edits.setdefault(n,{}).setdefault('CE',{})[net(P+'n5')]=net('service_snapshot_capture')
 ack=P+'acknowledge~FF';assert g[ack][1]['CE']==[net(P+'n5')]
 edits.setdefault(ack,{}).setdefault('CE',{})[net(P+'n5')]=net('service_snapshot_store_done')
 payload=[]
 for row in proof['payload']:
  if row['token'] is None:expr=row['expression']
  else:
   t=tuple(row['token']);n=drivers[t];pins=g[n][1]
   assert n in snapshots and pins['D'][0] in drivers
   assert drivers[pins['D'][0]] in held,(n,pins['D'])
   invert=row['invert'] ^ (ps[n]['D_POLARITY'][1]==0)
   expr=('~' if invert else '')+render_token(pins['D'][0])
  payload.append(expr)
 v=lambda n:render_token(net(n))
 header='{'+f"{v(P+'sequence_number')},{v(P+'raw_sync')},8'd5,4'd0,{v(P+'request_toggle')},{v(P+'sequence_number')}[0],({v(P+'request_toggle')}^{v(P+'sequence_number')}[0]),{v(P+'valid')},32'h4f465035"+'}'
 code='wire service_serial_tdo,service_snapshot_capture,service_snapshot_store_done;\n'
 code+='video_snapshot_serializer service_serializer(\n'
 code+=f'.source_clk(process_clk_150),.source_request({v(P+"n5")}),.source_bank({v(P+"request_sync")}),\n'
 code+='.source_data({'+','.join(reversed(payload))+'}),.source_capture(service_snapshot_capture),.source_done(service_snapshot_store_done),\n'
 code+=f'.tck(jtag_inst1_TCK),.capture({v(P+"n2987")}),.shift({v(B+"n8")}),.header({header}),.tdo(service_serial_tdo));\n'
 proof.update(removed_snapshot_cells=sorted(snapshots),fixed_sparse_source_capture=567,
  inactive_ram_banks=2,store_words=60,atomic_payload=True,
  snapshot_publication_delayed_until_scan_safe=False,
  publish_after_final_ram_write=True,no_new_ram=False,
  producer_handshake_unchanged=False,protocol_format_unchanged=True)
 return code,proof
