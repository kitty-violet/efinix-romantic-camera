"""Same-edge observation CE folding; no functional control or clock changes.

Ti60 packing experiment, not a resource claim. Keep native FF parameters and
replace CE with its active constant; a native LUT implements the exact data
feedback, including D/CE polarity. Four-state CE equivalence is required only
for the initialized binary handshake; data and held Q may be X/Z.
"""
from native_local_drivers import render_token
P='cpu/system_hardJtag_debug_logic_jtagBridge/observation/'
def extension(D,removed,edits):
 g,ps=D['g'],D['ps'];rows=[];code=[]
 names=sorted(n for n in g if n.startswith((P+'held_payload[',P+'snapshot[')))
 assert len(names)==1134
 for i,n in enumerate(names):
  kind,pins=g[n];par=ps[n]
  assert kind=='EFX_FF' and n not in removed
  assert pins['SR']==[('literal',"1'b0",None)] and par['SR_POLARITY'][1]==1
  assert 'CE' not in edits.get(n,{})
  original=pins['D'][0];data=edits.get(n,{}).get('D',{}).get(original,original)
  ce=pins['CE'][0];q=pins['Q'][0];dp=par['D_POLARITY'][1];cp=par['CE_POLARITY'][1]
  assert dp in (0,1) and cp in (0,1)
  mask=0
  for k in range(16):
   d=k&1;held=(k>>1)&1;en=(k>>2)&1
   value=d if en==cp else held^(1-dp)
   mask|=value<<k
  name=f'service_enable_feedback_{i:04d}';out=('net',name+'_data',None)
  edits.setdefault(n,{}).setdefault('D',{})[original]=out
  edits[n].setdefault('CE',{})[ce]=('literal',"1'b"+str(cp),None)
  code.append('wire '+name+'_data;')
  ports={"I0":data,"I1":q,"I2":ce,"I3":('literal',"1'b0",None),"O":out}
  code.append('EFX_LUT4 '+name+' ('+','.join('.'+p+'('+render_token(t)+')' for p,t in ports.items())+') /* Native same-edge CE fold */;')
  code.append('defparam '+name+'.LUTMASK=16\'h'+format(mask,'04x')+';')
  rows.append(dict(ff=n,lut=name,ports=ports,mask=mask,d_polarity=dp,ce_polarity=cp,data_before=data,enable_before=ce))
 return '\n'.join(code),dict(rows=rows,latency_added=0,fields_preserved=True,ff_parameters_unchanged=True,hardware_access=False,board_release=False)
