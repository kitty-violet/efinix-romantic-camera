"""Resolve cuts by retained driver identity, never by repacked LUT input order."""
def bind_cut(source,mapped,cuts,external):
 outputs={'EFX_LUT4':('O',),'EFX_FF':('Q',),'EFX_ADD':('O','CO'),'EFX_SRL8':('Q','Q7'),'EFX_DSP24':('O','CASCOUT'),'EFX_DSP48':('O','CASCOUT'),'EFX_RAM10':('RDATA',),'EFX_DPRAM10':('RDATA',)}
 drivers={t:(n,pin,j) for n,(kind,pins) in source.items() for pin in outputs.get(kind,()) for j,t in enumerate(pins.get(pin,[])) if t[0]=='net'}
 out={};rows=[]
 for t,value in cuts.items():
  if t in drivers:
   n,pin,j=drivers[t];assert n in mapped and mapped[n][0]==source[n][0]
   assert len(mapped[n][1][pin])==len(source[n][1][pin]);mt=mapped[n][1][pin][j]
  else:
   # Primary clock/reset cuts can be bound by an unchanged non-LUT input.
   candidates=[]
   for n,(kind,pins) in source.items():
    if kind=='EFX_LUT4' or n not in mapped:continue
    for pin,ts in pins.items():
     if pin in outputs.get(kind,()):continue
     for j,x in enumerate(ts):
      if x==t:
       assert len(mapped[n][1][pin])==len(ts);candidates.append((mapped[n][1][pin][j],n,pin))
   assert candidates,('unbound primary',t);mt,n,pin=candidates[0]
   assert all(x==mt for x,_,_ in candidates),('primary not bijective',t,candidates[:6])
  if mt not in external:continue
  assert mt not in out or out[mt]==value,('cut alias requires explicit proof',t,mt)
  out[mt]=value;rows.append(dict(source=t,mapped=mt,driver=n,port=pin,value=value))
 assert external<=set(out),('actual missing cuts',external-set(out))
 return out,rows
