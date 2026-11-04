"""Page all legacy fields plus service fields through one256-bit scan image.

The unchanged request/ack synchronizers still latch data on the source edge
and publish on the TCK edge. Capturing a scan copies header+published payload
together; later publications cannot change a scan already in progress.
"""
from video_service_serializer_native import extension as serial_cut,P,B,net
from native_local_drivers import render_token

def extension(D,removed,edits,supplied):
 _,proof=serial_cut(D,removed,edits,supplied)
 g,ps,drivers,loads=(D[k] for k in ('g','ps','drivers','loads'))
 snapshots={n for n in g if n.startswith(P+'snapshot[')}
 held={n for n in g if n.startswith(P+'held_payload[')}
 assert len(snapshots)==len(held)==567
 for n in snapshots:
  assert g[n][0]=='EFX_FF'
  assert all(c in removed for c,_ in loads[g[n][1]['Q'][0]]),(n,loads[g[n][1]['Q'][0]])
 for n in held:
  assert g[n][0]=='EFX_FF'
  assert all(c in snapshots for c,_ in loads[g[n][1]['Q'][0]]),(n,loads[g[n][1]['Q'][0]])
  assert g[n][1]['CE']==[net(P+'n5')] and ps[n]['CE_POLARITY']==(1,1)
  assert g[n][1]['SR']==[('literal',"1'b0",None)] and ps[n]['SR_POLARITY']==(1,1)
 removed.update(snapshots|held)
 for n in snapshots|held:edits.pop(n,None)
 publish=net(P+'n2000')
 for n,pin in loads[publish]:
  if n in removed:continue
  assert edits[n][pin].pop(publish)==net('service_snapshot_publish')
  if not edits[n][pin]:edits[n].pop(pin)
  if not edits[n]:edits.pop(n)
 raw=[]
 for row in proof['payload']:
  if row['token'] is None:expr=row['expression']
  else:
   t=tuple(row['token']);s=drivers[t];assert s in snapshots
   h=drivers[g[s][1]['D'][0]];assert h in held
   invert=row['invert']^(ps[s]['D_POLARITY'][1]==0)^(ps[h]['D_POLARITY'][1]==0)
   expr=('~' if invert else '')+render_token(g[h][1]['D'][0])
  raw.append(expr)
 v=lambda n:render_token(net(n))
 header='{'+f"{v(P+'sequence_number')},{v(P+'raw_sync')},8'd5,4'd0,{v(P+'request_toggle')},{v(P+'sequence_number')}[0],({v(P+'request_toggle')}^{v(P+'sequence_number')}[0]),{v(P+'valid')},32'h4f465035"+'}'
 code='wire service_serial_tdo;\nwire [1023:0] service_legacy_padded;\n'
 code+='assign service_legacy_padded={64\'d0,'+','.join(reversed(raw))+'};\n'
 code+='wire [2:0] service_legacy_part;assign service_legacy_part={service_page_part,service_page_phase[0]};\n'
 code+='wire [191:0] service_compact_packet;\n'
 code+='assign service_compact_packet=service_page_phase<2 ? {16\'d0,8\'d4,service_page_sequence,29\'h989e000,service_legacy_part,service_legacy_padded[service_legacy_part*128+:128]} : {9\'d0,service_observation_packet[182:176],8\'d4,service_observation_packet[167:0]};\n'
 code+='video_snapshot_serializer service_serializer(\n'
 code+=f'.source_clk(process_clk_150),.source_capture({v(P+"n5")}),.source_data(service_compact_packet),\n'
 code+=f'.tck(jtag_inst1_TCK),.publish({v(P+"n2000")}),.capture({v(P+"n2987")}),.shift({v(B+"n8")}),.header({header}),.tdo(service_serial_tdo));\n'
 proof.update(removed_snapshot_cells=sorted(snapshots),removed_held_cells=sorted(held),source_word_bits=192,
  published_word_bits=192,scan_register_bits=256,external_scan_bits=1024,tail_bits_zero=768,
  legacy_chunks=8,legacy_chunk_bits=128,service_chunks=4,service_chunk_bits=128,schema_version=4,
  whole_legacy_snapshot_atomic=False,each_chunk_atomic=True,producer_handshake_unchanged=True,
  snapshot_publication_delayed_until_scan_safe=False,protocol_format_unchanged=False,
  constant_legacy_padding_bits=64)
 return code,proof
