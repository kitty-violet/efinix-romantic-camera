"""Remove only the duplicate serial shifter; retain all observation payloads.

The 960-bit parallel bus is reconstructed by exhaustive capture-edge Boolean
cofactoring of the real native shifter inputs, including folded constants and
FF polarity. It is not guessed from the old assembly or assumed all-zero gaps.
"""
import itertools,re
from native_local_drivers import render_token
from camera_joint_native import OUT,used_inputs

P='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
B='cpu/system_hardJtag_debug_logic_jtagBridge/'
def net(n,b=None):return ('net',n,b)

def extension(D,removed,edits,supplied):
 g,ps,drivers,loads=(D[k] for k in ('g','ps','drivers','loads'))
 cap=net(P+'n2987');shift=net(B+'n8');ack=net(P+'n2000')
 forced={n for n,(k,p) in g.items() if n.startswith(P) and
          (n.startswith(P+'shifter[') or n.startswith(P+'dff_37/'))}
 forced.add(B+'diagnostic_tdo~FF')
 assert len(forced)==1022,len(forced)
 assert not forced&removed
 known={cap:1};memo={};active=set()
 def leaves(t):
  if t in known or t[0]=='literal':return set()
  n=drivers.get(t)
  if n is None or g[n][0] not in ('EFX_LUT4','EFX_ADD'):return {t}
  assert n not in active,('comb_loop',n);active.add(n)
  result=set().union(*(leaves(q) for v in used_inputs(g[n][0],g[n][1],ps[n]).values() for q in v))
  active.remove(n);return result
 def evaluate(t,values):
  if t in known:return known[t]
  if t in values:return values[t]
  if t[0]=='literal':assert t[1] in ("1'b0","1'b1");return int(t[1][-1])
  n=drivers[t];k,p=g[n]
  if k=='EFX_LUT4':
   idx=sum(evaluate(p.get('I'+str(i),[('literal',"1'b0",None)])[0],values)<<i for i in range(4))
   return ps[n]['LUTMASK'][1]>>idx&1
  assert k=='EFX_ADD',n
  x=evaluate(p['I0'][0],values)^(1-ps[n]['I0_POLARITY'][1]);y=evaluate(p['I1'][0],values)^(1-ps[n]['I1_POLARITY'][1])
  total=x+y+evaluate(p['CI'][0],values);return (total>>int(p['CO'][0]==t))&1
 payload=[];truth_cases=0
 for i in range(960):
  n=P+f'shifter[{64+i}]~FF';k,p=g[n];assert k=='EFX_FF'
  assert p['SR']==[('literal',"1'b0",None)]
  t=p['D'][0];terms=sorted(leaves(t),key=str);assert len(terms)<=5,(n,terms)
  snapshots=[q for q in terms if drivers.get(q,'').startswith(P+'snapshot[')]
  assert len(snapshots)<=1,(n,snapshots)
  table=[]
  for bits in itertools.product((0,1),repeat=len(terms)):
   values=dict(zip(terms,bits));v=evaluate(t,values)^(1-ps[n]['D_POLARITY'][1]);truth_cases+=1
   table.append((values,v))
  unique={v for _,v in table}
  if len(unique)==1:expr="1'b"+str(next(iter(unique)));token=None;invert=False
  else:
   assert len(snapshots)==1,(n,'not a held snapshot or constant');token=snapshots[0]
   invert=all(v==1-values[token] for values,v in table)
   assert invert or all(v==values[token] for values,v in table),(n,'non-snapshot dependency')
   expr=('~' if invert else '')+render_token(token)
  payload.append(dict(bit=i,expression=expr,token=token,invert=invert))
 # Delete backward-only shifter combinational logic; snapshot/header and the
 # public capture/shift handshakes are explicit boundaries.
 boundaries={cap,shift,ack}|{tuple(r['token']) for r in payload if r['token'] is not None}
 candidates=set(forced);todo=list(forced)
 while todo:
  n=todo.pop()
  for pin,terms in g[n][1].items():
   if pin in OUT[g[n][0]] or pin=='CLK':continue
   for t in terms:
    parent=drivers.get(t)
    if t not in boundaries and parent and parent not in candidates|removed and g[parent][0] in ('EFX_LUT4','EFX_ADD'):
     candidates.add(parent);todo.append(parent)
 while True:
  keep={n for n in candidates-forced for pin,terms in g[n][1].items() if pin in OUT[g[n][0]]
        for t in terms if t in boundaries or any(q not in candidates|removed for q,_ in loads[t])}
  if not keep:break
  candidates-=keep
 tdo=net(B+'diagnostic_tdo');outgoing=[]
 for n in candidates:
  for pin,terms in g[n][1].items():
   if pin not in OUT[g[n][0]]:continue
   for t in terms:
    if any(c not in candidates|removed for c,_ in loads[t]):outgoing.append(t)
 assert set(outgoing)=={tdo},outgoing
 removed.update(candidates)
 for n in candidates:edits.pop(n,None)
 # Deferring the shared publish pulse preserves snapshot/sequence/header
 # agreement; the producer continues holding data until this ack is consumed.
 for n,pin in loads[ack]:
  assert n not in removed
  assert ack not in edits.get(n,{}).get(pin,{})
  edits.setdefault(n,{}).setdefault(pin,{})[ack]=net('service_snapshot_publish')
 supplied[tdo]='service_serial_tdo'
 v=lambda n:render_token(net(n))
 snapshot='{'+', '.join(row['expression'] for row in reversed(payload))+'}'
 # Sequence bit zero is the original mapper's shared seen_ack register.
 header='{'+f"{v(P+'sequence_number')}, {v(P+'raw_sync')},8'd5,4'd0,{v(P+'request_toggle')}, {v(P+'sequence_number')}[0], ({v(P+'request_toggle')}^{v(P+'sequence_number')}[0]), {v(P+'valid')},32'h4f465035"+'}'
 code='\nwire service_serial_tdo,service_snapshot_publish,service_snapshot_publish_allowed;\n'
 code+=f'assign service_snapshot_publish={v(P+"n2000")} && service_snapshot_publish_allowed;\n'
 code+=f'video_snapshot_serializer service_serializer(.tck(jtag_inst1_TCK),.capture({v(P+"n2987")}),.shift({v(B+"n8")}),.header({header}),.snapshot({snapshot}),.tdo(service_serial_tdo),.publish_allowed(service_snapshot_publish_allowed));\n'
 return code,dict(removed_serial_cells=sorted(candidates),capture_truth_cases=truth_cases,payload=payload,
    old_scan_bits=1024,new_scan_bits=1024,payload_fields_preserved=True,snapshot_publication_delayed_until_scan_safe=True,
    no_new_ram=True,producer_handshake_unchanged=True,serial_equivalence_test_required=True)
