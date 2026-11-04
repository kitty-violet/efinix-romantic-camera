"""Independent original-tensor oracle for the shared DW/PW resident cluster."""
import sys
from pathlib import Path
from run_units import ROOT,HERE
sys.path.insert(0,str(HERE.parent/'pw_array_feed'))
from feed_reference import generate,parameters
sys.path.insert(0,str(ROOT/'network_ab'))
from coefficient_layout import build_layout
from contract_reference import requantize
from validate_dw_block import byte
def vectors(root,full=False,task_shapes=None):
 generate(root,'dense')
 ops={x['name']:x for x in build_layout()['operators']};params=parameters('dense')
 shapes=[('up2_high_dw',9,2,79,6),('down1.pw',9,2,79,6),
 ('up1_high_dw',1,1,0,0),('rgb_head',1,1,0,0),
 ('res1_conv1.dw',33,3,287,177),('res1_conv1.pw',33,3,287,177),
 ('down2.pw',9,4,78,9),('up1_low_pw',9,4,76,12),
 ('up2_low_pw',32,4,0,0),('up2_high_dw',32,4,0,0)]
 if full:shapes += [('res1_conv1.dw',80,4,80,3),('res1_conv1.pw',80,4,80,3),
  ('up1_high_dw',320,4,0,0),('down1.pw',320,4,0,0),('rgb_head',320,4,0,0),
  ('res1_conv2.dw',84,2,236,178),('res1_conv2.pw',84,2,236,178)]
 shapes.append(('down1.pw',80,2,160,3))
 shapes=[(*s,0) for s in shapes]
 shapes += [('up1_high_dw',9,2,0,0,1),('up1_high_dw',9,3,631,357,1),('up2_high_dw',1,1,1279,719,1),('up2_high_dw',33,3,77,13,1)]
 if full:shapes += [('up1_high_dw',320,4,0,0,1),('up1_high_dw',320,4,320,356,1),('up2_high_dw',320,4,320,4,1),('up2_high_dw',320,4,960,716,1)]
 if task_shapes is not None:shapes=list(task_shapes)
 shapes += [('stem',1,1,0,0,0,1280,720),('stem',9,2,1271,718,0,1280,720),('stem',33,3,287,177,0,1280,720),('stem',80,4,80,4,0,1280,720)]
 if full:shapes += [('stem',w,h,ox,oy,0,1280,720) for w,h,ox,oy in [(320,4,0,0),(320,4,320,4),(320,4,960,716),(81,4,1199,100),(7,1,5,2),(8,2,640,300)]]
 cases=[];gold=[];headers=[];details=[];views=[];stems=[]
 for c,shape in enumerate(shapes):
  name,w,h,ox,oy,nn=shape[:6]
  stem=name=='stem';stems.append(int(stem))
  op=ops[name];pw=op['kind']=='pw1x1';gi=max(1,(op['input_channels']+7)//8);go=op['padded_channels']//8
  mask=(1<<min(8,op['output_channels']))-1;par=params[name];origin=(ox if pw else max(0,ox-1))>>nn
  sw=(640 if name=='up1_high_dw' else 1280) if nn else 320
  sw,sh=shape[6:8] if len(shape)==8 else (sw,sw*9//16);views.append(nn)
  span=((w+3 if pw else w+5)//4)*gi;outspan=((w+3)//4)*go
  if nn:span=(((min(sw-1,ox+w)>>1)-origin+1+3)//4)*gi
  pcount=[0]*3;alloc={};planes=[];bases=[];opl=[];oba=[];outcount=[0]*3
  for y in range(6):
   sy=(oy+y if pw else max(0,min(sh-1,oy+y-1)))>>nn;plane=(sy+c)%3
   if sy not in alloc:alloc[sy]=pcount[plane]*span;pcount[plane]+=1
   planes.append(plane);bases.append(alloc[sy])
  for y in range(4):
   plane=(oy+y+c+1)%3;opl.append(plane);oba.append(2*span+16+outcount[plane]*outspan);outcount[plane]+=1
  assert max(oba)+outspan<=1024
  # Disjoint input/output planes allow a real result-queue fill while writes
  # are held, instead of stalling RAM reads before MAC backpressure is reached.
  if name=='down1.pw' and w==80 and h==2 and ox==160:
   planes=[0,1,0,1,0,1];bases=[(y//2)*span for y in range(6)]
   opl=[2]*4;oba=[16+y*outspan for y in range(4)]
  cases.append([c,len(gold),int(pw),w,h,gi,go,mask,ox,origin,sw,span,op['weights']['offset'],
   sum(p<<(2*y) for y,p in enumerate(planes)),sum(v<<(10*y) for y,v in enumerate(bases)),
   sum(p<<(2*y) for y,p in enumerate(opl)),sum(v<<(10*y) for y,v in enumerate(oba))])
  bias=sum((v&0xffffffff)<<(32*i) for i,v in enumerate(par.bias))
  mult=sum((v&0xffffffff)<<(32*i) for i,v in enumerate(par.multiplier))
  relu=((1<<len(par.bias))-1) if par.relu else 0;headers.append((bias,mult,relu))
  for y in range(h):
   for x in range(w):
    for group in range(go):
     vals=[]
     for l in range(8):
      oc=group*8+l
      if oc>=op['output_channels']:vals.append(0);continue
      dot=0
      if pw:
       for ch in range(gi*8):
        b=planes[y]*4+x%4;a=bases[y]+x//4*gi+ch//8
        dot+=byte(b,a,ch%8,c)*par.weights[oc][ch]
      else:
       for ky in range(3):
        for kx in range(3):
         lx=(max(0,min(sw-1,ox+x+kx-1))>>nn)-origin;b=planes[y+ky]*4+lx%4;a=bases[y+ky]+lx//4*gi+group
         if stem:
          for ic in range(3):dot+=byte(b,a,ic,c)*par.weights[oc][(3*ky+kx)*3+ic]
         else:dot+=byte(b,a,l,c)*par.weights[oc][3*ky+kx]
      val,overflow=requantize(dot,par.bias[oc],par.multiplier[oc],relu=par.relu);assert not overflow;vals.append(val)
     gold.append(sum((v&255)<<(8*l) for l,v in enumerate(vals)))
  details.append(dict(case=c,operator=name,width=w,rows=h,mode='stem' if stem else 'pw' if pw else 'dw',origin=origin,span=span,nn_view=nn,source_width=sw))
 (root/'golden.hex').write_text(''.join(f'{v:016x}\n' for v in gold))
 (root/'cases.txt').write_text(str(len(cases))+'\n'+''.join(' '.join(str(v) if i<13 else f'{v:x}' for i,v in enumerate(row))+'\n' for row in cases))
 (root/'case_stem.hex').write_text(''.join(f'{v:x}\n' for v in stems))
 (root/'case_view.hex').write_text(''.join(f'{v:x}\n' for v in views))
 for i,name in enumerate(['case_bias.hex','case_multiplier.hex','case_relu.hex']):
  (root/name).write_text(''.join(f'{h[i]:0{256 if i<2 else 8}x}\n' for h in headers))
 return dict(cases=len(cases),output_words=len(gold),scalar_checks=len(gold)*8,details=details)
