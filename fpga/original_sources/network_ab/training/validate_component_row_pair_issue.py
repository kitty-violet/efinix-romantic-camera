"""Versioned C3 six-bank issuer; every field checked against full program."""
import argparse,json,re,shutil
from pathlib import Path
from .train_luma_regions import ROOT,RUN
from .runtime import file_hash,write_json
from .compile_region_program import offsets
from .build_region_rtl import build,run
from .prepare_region_maps import project


def derivative(s,m):
    cs,ds=m['streams'];cp,cb=offsets(cs['schema']);dp,db=offsets(ds['schema']);bits=m['index_bits'];width=m['stripe_width']
    s=s.replace('[13:0]',f'[{bits-1}:0]').replace('output wire [14:0] debug_program_index',f'output wire [{bits-1}:0] debug_program_index')
    s=s.replace('output wire dstore,drgb,','output wire dstore,drgb,dregion,')
    for prefix,pos in [('c',cp),('d',dp)]:
        pattern=rf'wire \[(\d+):0\] {prefix}(\d+)={prefix}data\[\d+\+:\d+\];'
        def replace(match):
            off,n=pos['f'+match[2]];return f'wire [{n-1}:0] {prefix}{match[2]}={prefix}data[{off}+:{n}];'
        s,n=re.subn(pattern,replace,s);assert n==len([k for k in pos if k.startswith('f')])
    s=s.replace('cdata[28:14]',f'cdata[{2*bits-1}:{bits}]')
    s=s.replace("cdata[511:504]!=8'h52",f"cdata[511:504]!=8'h{cs['version']:02x}").replace('cdata[503:447]',f'cdata[503:{cb}]')
    s=s.replace("ddata[127:124]!=4'h2",f"ddata[127:124]!=4'h{ds['version']:x}").replace('ddata[123:121]',f'ddata[123:{db}]')
    s=s.replace('c1>=29',f"c1>={m['nodes']}").replace('d1>=29',f"d1>={m['nodes']}").replace('>=19229',f">={m['commands']}")
    expected=str(width) if 1280%width==0 else f"(rgb_x==11'd{1280//width*width} ? 9'd{1280%width}:9'd{width})"
    s=s.replace('!d0 || !d6 || d2!=256',f'!d0 || !d6 || !dregion || d2!={expected}')
    s=s.replace("rgb_x+11'd256","rgb_x+{2'd0,d2}").replace("nodes_completed==29'h1fffffff",f"nodes_completed==29'h{m['complete_node_mask']:x}")
    s=s.replace('assign dstore=d0;',f"assign dregion=ddata[{dp['region'][0]}];assign dstore=d0;")
    s=s.replace('assign cfg_storage_phase=cdata[442];assign cfg_out_phase=cdata[443+:4];',f"assign cfg_storage_phase=cdata[{cp['storage_phase'][0]}];assign cfg_out_phase=cdata[{cp['out_phase'][0]}+:4];")
    names={236:'f17',251:'f18',266:'f19',29:'f1',49:'f5'}
    s=re.sub(r'next_data\[(\d+)\+:(\d+)\]',lambda x:f"next_data[{cp[names[int(x[1])]][0]}+:{x[2]}]",s)
    return s


def main():
    p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args();m=json.loads((a.program/'program.json').read_text());out=ROOT/'runs'/RUN/a.name;out.mkdir(exist_ok=False)
    hw=ROOT.parent/'hw/efinity/prototypes/frame_execution/variants/row_pair_arena';original=hw/'frame_program_issue.sv';dst=out/original.name;dst.write_text(derivative(original.read_text(),m));cs,ds=m['streams'];nc,nd=cs['count'],ds['count'];n=m['commands'];bits=m['index_bits']
    original_tb=hw/'tb_frame_program_issue.sv';tb=original_tb.read_text()
    for old,new in [('8789',str(nc)),('8788',str(nc-1)),('10440',str(nd)),('10439',str(nd-1)),('19228',str(n-1))]:tb=tb.replace(old,new)
    tb=tb.replace('module tb_frame_program_issue;','module tb_component_issue;\n wire dregion;').replace('[13:0]',f'[{bits-1}:0]').replace("14'd",f"{bits}'d").replace('wire [14:0] debug_program_index',f'wire [{bits-1}:0] debug_program_index')
    tb=tb.replace('.drgb(drgb),','.drgb(drgb),.dregion(dregion),').replace("drgb!==1'(df(6))","dregion!==1'(df(0)) || drgb!==1'(df(6))")
    tb=tb.replace('ticks>500000','ticks>3000000').replace("nodes_completed!=29'h1fffffff",f"nodes_completed!=29'h{m['complete_node_mask']:x}")
    tb=tb.replace('cmem[0][447]=1',f"cmem[0][{cs['used_bits']}]=1").replace('dmem[0][121]=1',f"dmem[0][{ds['used_bits']}]=1")
    # C3 version's high bit is already 1; force an actual version error.
    tb=tb.replace('restart();cmem[0][511]=1;',"restart();cmem[0][511]=0;")
    tbpath=out/'tb_component_issue.sv';tbpath.write_text(tb)
    phases=[0]*n;dma=[]
    for meta in m['streams']:
        path=a.program/(meta['name']+'.bin');assert file_hash(path)==meta['sha256'];raw=path.read_bytes();pos,_=offsets(meta['schema'])
        for off in range(0,len(raw),meta['record_bytes']):
            word=int.from_bytes(raw[off:off+meta['record_bytes']],'little');v={k:(word>>p)&((1<<b)-1) for k,(p,b) in pos.items()};i=v['f31']
            if meta['name']=='compute':phases[i]=v['storage_phase']|(v['out_phase']<<1)
            else:dma.append(i)
        shutil.copy2(a.program/(meta['name']+'.hex'),out/(meta['name']+'.hex'))
    shutil.copy2(a.program/'fields.hex',out/'expected_program.hex');(out/'expected_phase.hex').write_text(''.join(f'{x:02x}\n' for x in phases));(out/'dma_commands.hex').write_text(''.join(f'{x:08x}\n' for x in dma))
    write_json(out/'inputs.json',dict(program=str(a.program),program_sha256=file_hash(a.program/'program.json'),source=str(original),source_sha256=file_hash(original),tb_source=str(original_tb),tb_source_sha256=file_hash(original_tb),derived={str(f):file_hash(f) for f in [dst,tbpath]},production_modified=False))
    build(out,[dst,tbpath],'tb_component_issue');sec=run([out/'obj/Vtb_component_issue.exe','+VECTORS='+out.as_posix(),'+DESCRIPTORS='+out.as_posix()],out,out/'simulation.log',120)
    body=(out/'simulation.log').read_text();assert 'PASS PROGRAM_ISSUE' in body;write_json(out/'result.json',dict(status='C3_FULL_PROGRAM_ROW_PAIR_ISSUER_PROTOCOL_PASS',seconds=sec,lines=[s for s in body.splitlines() if s.startswith('PASS')],actual_compute=False,actual_DDR=False,board=False));print(body[-1800:],flush=True)
    project(ROOT/'runs'/RUN/(a.name+'-map'),'frame_program_issue',[dst])


if __name__=='__main__':main()
