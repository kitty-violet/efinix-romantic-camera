"""Source RGB -> every scheduled layer -> SRAM categorical decoder.

Only input RGB is written by the host. Intermediate activations are read back
for checking, never replaced by oracle values. This is a bounded short graph
test, not a whole board, full frame, timing or production ABI release.
"""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
from .runtime import configure_workspace,contract_guard,file_hash,write_json
from .region_detail_target import ROOT,RUN
from .validate_region_native import coefficient_image,pack
from .build_region_rtl import build,run
from .audit_region_class_schedule import graph
from .probe_boundary_budget import HW,tile
from .probe_region_class_int8 import execute
from .train_b1_distillation import read_image
from .train_luma_regions import LEVELS


TASKS=r'''
 reg row_mode=0,rcv=0,rdr=0;wire rcr,rdv,rbusy,rerror;
 reg [8:0] rpixels;reg [1:0] rsource,routput;reg [9:0] rbase,robase;
 wire [23:0] rv,rw,rr;wire [239:0] ra;wire [1535:0] rd;
 wire [31:0] rcycles,rreads,rwrites,racks;
 region_row_decode reduction(.clk(clk),.reset(reset),.cancel(cancel),
 .cmd_valid(rcv),.cmd_ready(rcr),.cmd_pixels(rpixels),.cmd_source_plane(rsource),.cmd_output_plane(routput),.cmd_source_base(rbase),.cmd_output_base(robase),
 .done_valid(rdv),.done_ready(rdr),.busy(rbusy),.error(rerror),.av(rv),.ar(ar),.aw(rw),.aa(ra),.ad(rd),
 .sv(sv&{24{row_mode}}),.sr(rr),.sd(sd),.sw(sw),.cycles(rcycles),.reads(rreads),.writes(rwrites),.acks(racks));
 reg [63:0] words[0:2559],source_words[0:NSOURCE-1],golden[0:NGOLD-1];
 integer caseid=0,kind=0,g=0,weight_offset=0,waits=0,checks=0,blocks=0,held_checks=0,elapsed=0,frame_cycles=0;
 integer width=0,height=0,groups_in=0,off=0;
 reg [31:0] rng=32'h74123;reg [23:0] held_req=0,exposed_request_hold=0,exposed_response_hold=0;
 reg [239:0] held_addr;reg [1535:0] held_data;reg [23:0] held_write;
 function [31:0] next_rng(input [31:0] a);reg [31:0] z;begin z=a^(a<<13);z=z^(z>>17);next_rng=z^(z<<5);end endfunction
 task fail(input string reason);begin $display("FAIL GRAPH command=%0d group=%0d %s",caseid,g,reason);$fatal(1);end endtask
 always @(negedge clk)begin
  if(run_mode && random_stall)begin
   rng=next_rng(rng);req_gate=rng[23:0]|(rng[23:0]<<1)|exposed_request_hold;
   rng=next_rng(rng);rsp_gate=rng[23:0]|(rng[23:0]>>1)|exposed_response_hold;
   rng=next_rng(rng);permit=rng[4:0]|(rng[4:0]<<1);
  end else begin req_gate=24'hffffff;rsp_gate=24'hffffff;permit=31;end
 end
 always @(posedge clk)begin
  if(run_mode && !reset && !flush)begin
   elapsed=elapsed+1;
   for(integer p=0;p<24;p=p+1)if(held_req[p])begin
    if(!v[p] || ca[p*10+:10]!==held_addr[p*10+:10] || cw[p]!==held_write[p] || (cw[p] && cd[p*64+:64]!==held_data[p*64+:64]))fail("held request changed");held_checks=held_checks+1;
   end
   held_req=v&~cr;exposed_request_hold=av&~ar;exposed_response_hold=sv&rsp_gate&~sr;
  end else begin held_req=0;exposed_request_hold=0;exposed_response_hold=0;end
  held_addr=ca;held_data=cd;held_write=cw;
 end
 task load_row(input integer plane,base,n,offset);
  integer pn;begin run_mode=0;row_mode=0;
   for(integer x=0;x<n;x=x+4)begin
    @(negedge clk);tv=0;tw=0;ta=0;td=0;
    for(integer l=0;l<4;l=l+1)if(x+l<n)begin
     pn=(plane*4+l)*2;tv[pn]=1;tw[pn]=1;ta[pn*10+:10]=10'(base+x/4);td[pn*64+:64]=source_words[offset+x+l];
    end
    @(posedge clk);if((ar&tv)!==tv)fail("load not ready");
   end
   @(negedge clk);tv=0;tw=0;repeat(2)@(negedge clk);
  end
 endtask
 task check_row(input integer plane,base,n,ng,offset);
  integer pn,idx;reg [23:0] ports;reg [1535:0] expected;
  begin run_mode=0;row_mode=0;
   for(integer x=0;x<n;x=x+4)for(integer ch=0;ch<ng;ch=ch+1)begin
    @(negedge clk);tv=0;tw=0;ta=0;ports=0;expected=0;
    for(integer l=0;l<4;l=l+1)if(x+l<n)begin
     pn=(plane*4+l)*2;tv[pn]=1;ports[pn]=1;ta[pn*10+:10]=10'(base+x/4*ng+ch);
     idx=offset+(x+l)*ng+ch;expected[pn*64+:64]=golden[idx];
    end
    @(posedge clk);if((ar&tv)!==tv)fail("check not ready");@(negedge clk);tv=0;
    for(integer p=0;p<24;p=p+1)if(ports[p])begin
     if(!sv[p] || sw[p] || sd[p*64+:64]!==expected[p*64+:64])begin
      $display("MISMATCH plane=%0d base=%0d x=%0d ch=%0d p=%0d actual=%h expected=%h",plane,base,x,ch,p,sd[p*64+:64],expected[p*64+:64]);fail("numeric");end
     checks=checks+8;
    end
   end
   @(negedge clk);
  end
 endtask
 task compute;
  begin run_mode=0;row_mode=0;pcv=1;do @(posedge clk);while(!pcr);@(negedge clk);pcv=0;
   run_mode=1;elapsed=0;
   for(g=0;g<(kind==0 ? groups_in:1);g=g+1)begin start_group(g);finish_group(0);end
   run_mode=0;blocks=blocks+1;frame_cycles=frame_cycles+elapsed;
   $display("GRAPH_BLOCK cmd=%0d pixels=%0d rows=%0d cycles=%0d",caseid,width,height,elapsed);
  end
 endtask
 task decode_row(input integer plane,base,n,output_plane);
  begin run_mode=0;row_mode=1;rsource=2'(plane);routput=2'(output_plane);rbase=10'(base);robase=960;rpixels=9'(n);
   @(negedge clk);rcv=1;do @(posedge clk);while(!rcr);@(negedge clk);rcv=0;
   waits=0;while(!rdv)begin @(negedge clk);waits=waits+1;if(waits>2000 || rerror)fail("row decoder");end
   if(rreads!=32'(n*4) || rwrites!=32'(n) || racks!=32'(n))fail("row accounting");
   frame_cycles=frame_cycles+int'(rcycles);rdr=1;@(negedge clk);rdr=0;row_mode=0;
  end
 endtask
'''


def create(a):
    root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False)
    package=json.loads(a.package.read_text());ops=coefficient_image(package,root)
    sys.path.insert(0,str(HW));from fourway_frame_program import compile_tile
    plan=graph();commands=[];alloc=[]
    for ix,x in enumerate(a.x):
        p=compile_tile(plan,dict(name='research_luma'),tile(plan,ix,x,64),0,4,aligned=True,plane_shift=True,prefetch_rows=4,retain_commit=True,compact_lifetime=True)
        # Cyclic rotation preserves addresses/lifetimes/port conflicts and
        # meets the B1 reader's row-5 mux restriction at both NN boundaries.
        rotated=[]
        for q in p['commands']:
            q=dict(q)
            for key in ['inputs','outputs','skip']:
                if key in q:q[key]=[dict(r,plane=(r['plane']+2)%3) for r in q[key]]
            if 'row' in q:q['row']=dict(q['row'],plane=(q['row']['plane']+2)%3)
            if q['kind']=='compute':
                if q['skip']:
                    normalized=[]
                    for r in q['skip']:
                        delta=q['output_x']-r['origin'];assert delta>=0 and delta%4 in (0,2)
                        shift=delta//4*4
                        normalized.append(dict(r,base=r['base']+shift//4*r['groups'],origin=r['origin']+shift,width=r['width']-shift,delta=delta%4))
                    q['skip']=normalized
                rows=q['inputs'];ps=[r['plane'] for r in rows]
                ordinal=[sum(ps[k]!=ps[k-1] for k in range(1,j+1)) for j in range(len(ps))]
                slots=[2*p+int(o>=3) for p,o in zip(ps,ordinal)]
                assert not any(s%2 for s in slots[:3]) and not(len(slots)>5 and slots[5] in (0,4)),q
            rotated.append(q)
        cut=[];count=0
        for q in rotated:
            cut.append(q);count+=q['kind']=='store'
            if count==a.rows:break
        assert count==a.rows;commands.extend(cut);alloc.append(dict(x=x,high=p['finite_high_water'],commands=len(cut)))
    manifest=json.loads((ROOT/'runs'/RUN/'luma/data.json').read_text());src=next(r for r in manifest['validation'] if r['id']==a.image)['source']
    # Top 128 lines retain all receptive-field dependencies of the checked
    # first eight output lines; assert every scheduled source/output fits.
    image=read_image(src);values=execute(package,(image[:128].astype(np.int64)-128).transpose(2,0,1)[None],False)
    source=[];gold=[];body=[];details=[]
    def rowwords(v):
        assert v.ndim==2
        return [pack([int(c)&255 for c in z],8) for pixel in v for z in np.pad(pixel,(0,(-len(pixel))%8)).reshape(-1,8)]
    def checks(r,v):
        off=len(gold);gold.extend(rowwords(v));return f"check_row({r['plane']},{r['base']},{r['width']},{r['groups']},{off});"
    for cid,q in enumerate(commands):
        body.append(f'caseid={cid};');details.append(q)
        if q['kind']=='load':
            r=q['row'];assert r['y']<128;off=len(source);source.extend(rowwords(values['input'][0,:,r['y'],r['origin']:r['origin']+r['width']].T))
            body.append(f"load_row({r['plane']},{r['base']},{r['width']},{off});");continue
        if q['kind']=='store':
            r=q['row'];v=values['rgb'][0,:,r['y'],r['origin']:r['origin']+r['width']].T
            body.append(checks(r,v));y=LEVELS[0][0]+np.asarray(LEVELS[0])[v[:,:16].argmax(1)];co=np.asarray(LEVELS[1])[v[:,16:24].argmax(1)];cg=np.asarray(LEVELS[2])[v[:,24:].argmax(1)]
            rgb=np.stack((y+co-cg,y+cg,y-co-cg),1).clip(0,255)-128;dst=dict(r,plane=(r['plane']+1)%3,base=960,groups=1)
            if a.dma:
                off=len(gold);gold.extend(rowwords(rgb));address=0x800000+(r['y']*1280+r['origin'])*3
                body.append(f"dma_row({r['plane']},{r['base']},{r['width']},{off},{address});")
            else:
                body.append(f"decode_row({r['plane']},{r['base']},{r['width']},{dst['plane']});");body.append(checks(dst,rgb))
            continue
        name='rgb' if q['parameter_node']=='rgb_head' else q['parameter_node'];p=package['parameters'][name];n=next(n for n in package['plan']['nodes'] if n['id']==name)
        rr=q['inputs'];oo=q['outputs'];ss=q['skip'];fused=q['operator']=='pw_fused';res=package['residual'][q['node']] if fused else dict(main_scaled=False,multiplier=0,post_multiplier=0)
        delta=q['output_x']-ss[0]['origin'] if ss else 0;assert all(q['output_x']-s['origin']==delta for s in ss)
        cfg=dict(kind=int(q['operator'] in ('pw1x1','pw_fused')),width=q['pixels'],height=q['rows'],groups_in=q['output_groups'],weight_offset=ops[name]['weight_offset'],
            cfg_mode_pw=int(q['operator'] in ('pw1x1','pw_fused')),cfg_nn_view=int(q['nn']),cfg_stem=int(q['operator']=='conv3x3'),cfg_stride2=int(q['stride2']),
            cfg_pixels=q['pixels'],cfg_rows=q['rows'],cfg_in_groups=q['input_groups'],cfg_groups=q['output_groups'],cfg_mask=q['mask'],cfg_output_x=q['output_x'],cfg_source_width=q['source_width'],cfg_source_origin=rr[0]['origin'],
            cfg_pw_fuse=int(fused),cfg_main_scaled=int(res['main_scaled']),cfg_skip_delta=delta,align_multiplier=res['multiplier'],post_multiplier=res['post_multiplier'])
        for k,v in cfg.items():body.append(f'{k}={v};')
        for k,rows,field,bits,width in [('cfg_row_planes',rr,'plane',2,12),('cfg_row_bases',rr,'base',10,60),('out_planes',oo,'plane',2,8),('out_bases',oo,'base',10,40),('cfg_skip_planes',ss,'plane',2,8),('cfg_skip_bases',ss,'base',10,40)]:
            body.append(f"{k}={width}'h{pack([r[field] for r in rows],bits):x};")
        body.extend([f"pcb=1024'h{pack([v&0xffffffff for v in p['bias_int32']],32):x};",f"pcm=1024'h{pack(p['multiplier_q17'],32):x};",f"pcrelu=32'h{((1<<n['channels'])-1) if n['activation']=='relu' else 0:x};",'compute();'])
        v=values['rgb' if q['node']=='rgb_head' else q['node']][0]
        for r in oo:
            assert r['y']<v.shape[1]-16
            body.append(checks(r,v[:,r['y'],r['origin']:r['origin']+r['width']].T))
    for filename,arr in [('source.hex',source),('golden.hex',gold)]:
        (root/filename).write_text(''.join(f'{v:016x}\n' for v in arr))
    frozen=ROOT/'runs'/RUN/'rtl-variable-groups/frozen';original=(frozen/'tb_compute_column_cluster.sv').read_text()
    prefix=original[:original.index(' reg[63:0] words')].replace('module tb_compute_column_cluster;','module tb_region_graph;')
    for old,new in [('assign av=run_mode ?','assign av=row_mode ? rv : run_mode ?'),('assign aw=run_mode ?','assign aw=row_mode ? rw : run_mode ?'),('assign aa=run_mode ?','assign aa=row_mode ? ra : run_mode ?'),('assign ad=run_mode ?','assign ad=row_mode ? rd : run_mode ?'),('assign at=run_mode ?','assign at=row_mode ? 384\'d0 : run_mode ?'),('assign sr=run_mode ?','assign sr=row_mode ? rr : run_mode ?')]:
        assert old in prefix;prefix=prefix.replace(old,new)
    prefix=prefix.replace('.cancel(flush),','.cancel(flush || cancel),')
    coretasks=original[original.index(' task start_group'):original.index(' task verify;')]
    load=original[original.index('  repeat(4)@(negedge clk);reset=0;'):original.index('  for(c=0;c<ncases;c=c+1)begin')]
    load=load[:load.index('  imagefd=')]+'  verify_valid=1;do @(posedge clk);while(!verify_ready);@(negedge clk);verify_valid=0;\n'
    load=load.replace('for(k=0;k<2560;k=k+1)','for(integer k=0;k<2560;k=k+1)')
    initial=''' initial begin
  $readmemh("load_words.hex",words);$readmemh("source.hex",source_words);$readmemh("golden.hex",golden);
'''+load+"\nrandom_stall=1;\n"+'\n'.join(body)+f'\nif(held_checks==0 || blocks!={sum(q["kind"]=="compute" for q in commands)})fail("coverage");\n'+'''
  $display("PASS REGION_GRAPH blocks=%0d scalar_checks=%0d held=%0d active_cycles=%0d",blocks,checks,held_checks,frame_cycles);$finish;
 end
 initial begin #200000000;fail("global watchdog");end
endmodule
'''
    files=[]
    for f in frozen.iterdir():
        if f.suffix not in ('.v','.sv') or f.name.startswith('tb_'):continue
        dst=root/f.name;shutil.copy2(f,dst);files.append(dst)
    for name in ['region_luma_decode.sv','region_row_decode.sv']:
        dst=root/name;shutil.copy2(Path(__file__).with_name(name),dst);files.append(dst)
    tasks=TASKS
    if a.dma:
        from .region_graph_dma_tb import adapt
        tasks=adapt(tasks)
        f=ROOT/'runs'/RUN/'rtl-region-dma-v2/frame_arena_row_dma.sv';dst=root/f.name;shutil.copy2(f,dst);files.append(dst)
        initial=initial.replace('$display("PASS REGION_GRAPH',f'if(total_output_bytes!={len(a.x)*a.rows*64*3})fail("RGB byte coverage");\n  $display("PASS REGION_GRAPH_DMA bytes=%0d",total_output_bytes);\n  $display("PASS REGION_GRAPH')
    tb=root/'tb_region_graph.sv';tb.write_text(prefix+tasks.replace('NSOURCE',str(len(source))).replace('NGOLD',str(len(gold)))+coretasks+initial);files.append(tb)
    write_json(root/'inputs.json',dict(package=str(a.package),package_sha256=file_hash(a.package),image=src,image_sha256=file_hash(src),
        source_only_host_writes=True,integer_oracle='independent int64 numpy',commands=details,allocations=alloc,golden_words=len(gold),files={str(f):file_hash(f) for f in files}))
    print('GRAPH_PREPARED',len(commands),len(gold),flush=True);build(root,files,'tb_region_graph')
    seconds=run([root/'obj/Vtb_region_graph.exe'],root,root/'simulation.log',180)
    text=(root/'simulation.log').read_text();assert 'PASS REGION_GRAPH' in text
    write_json(root/'result.json',dict(status='BOUNDED_COMPLETE_GRAPH_DMA_PASS' if a.dma else 'BOUNDED_COMPLETE_GRAPH_PASS',seconds=seconds,lines=text.splitlines(),full_frame=False,board=False,fit=False,timing=False))
    print(text[-1500:],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--package',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--x',type=int,action='append',default=None);p.add_argument('--rows',type=int,default=8);p.add_argument('--image',default='val_stairs_090');p.add_argument('--dma',action='store_true');a=p.parse_args();a.x=a.x or [0,512]
    assert 1<=a.rows<=8 and all(0<=x<=1216 and x%4==0 for x in a.x)
    configure_workspace();contract_guard();create(a)


if __name__=='__main__':main()
