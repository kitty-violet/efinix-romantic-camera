"""Original-tensor golden for packed RGB, direct stride and real PW/skip fusion."""
from fractions import Fraction
from cluster_stride_vectors import vectors as original
from residual_q17_align import encode_alignment, fixtures, residual_value
from validate_dw_block import byte


def vectors(root, full=False, specialized=False):
    shapes = [('rgb_head',w,h,x,y,0,1280,720) for w,h,x,y in
              [(1,1,0,0),(7,2,1273,718),(8,3,3,1),(9,4,11,5),(33,4,77,12),(320,4,960,716)]]
    shapes += [('res1_conv2.pw',w,h,x,y,0,320,180) for w,h,x,y in
               [(1,1,0,0),(7,2,313,178),(8,3,5,4),(9,4,17,12),(80,4,80,4),(84,2,236,178)]]
    shapes += [('res1_conv1.dw',33,4,17,5,0,320,180),
               ('down1.pw',80,2,160,3,0,320,180),
               ('up1_high_dw',33,4,603,356,1,640,360),
               ('up2_low_pw',9,3,7,5,0,640,360)]
    oracle = original(root,full,task_shapes=shapes)
    rows = [line.split() for line in (root/'cases.txt').read_text().splitlines()[1:]]
    if specialized:
        # Preserve the independent input tensor/oracle and held-write fixture,
        # but use the production cyclic output-plane contract.
        for row in rows:
            first=int(row[15],16)&3
            row[15]=format(sum(((first+y)%3)<<(2*y) for y in range(4)),'x')
        (root/'cases.txt').write_text(str(len(rows))+'\n'+'\n'.join(' '.join(r) for r in rows)+'\n')
    gold = [int(s,16) for s in (root/'golden.hex').read_text().splitlines()]
    records = [encode_alignment(p) for p in fixtures()]
    columns = {n: [] for n in ('fuse','main_scaled','skip_planes','skip_bases','skip_delta','align','post')}
    fused = []
    for detail,row in zip(oracle['details'],rows):
        ci=int(row[0]); offset=int(row[1]);w=int(row[3]);h=int(row[4]);groups=int(row[6])
        use=detail['operator']=='res1_conv2.pw'
        record=records[ci%len(records)]
        planes=[(y+ci+2)%3 for y in range(4)]
        bases=[620+(y//3)*128 for y in range(4)]
        delta=(ci%2)*2 if specialized else ci%4
        packed=lambda values,bits:sum(v<<(bits*i) for i,v in enumerate(values))
        values=(int(use),int(record['main_scaled']),packed(planes,2),packed(bases,10),delta,record['multiplier'],record['post_multiplier'])
        for name,value in zip(columns,values):columns[name].append(value)
        if not use:continue
        for y in range(h):
            for x in range(w):
                for g in range(groups):
                    index=offset+(y*w+x)*groups+g
                    previous=gold[index]; result=0
                    for lane in range(8):
                        value=(previous>>(lane*8))&255
                        main=value-256 if value>=128 else value
                        skip=byte(planes[y]*4+(x+delta)%4,bases[y]+((x+delta)//4)*groups+g,lane,ci)
                        value=residual_value(main,skip,record['parameters'])
                        result|=(value&255)<<(lane*8)
                    gold[index]=result
        fused.append(dict(case=ci,width=w,rows=h,groups=groups,alignment=record,skip_delta=delta))
    for name,values in columns.items():
        (root/f'case_{name}.hex').write_text(''.join(f'{v:x}\n' for v in values))
    (root/'golden.hex').write_text(''.join(f'{v:016x}\n' for v in gold))
    oracle['fused_cases']=fused
    return oracle
