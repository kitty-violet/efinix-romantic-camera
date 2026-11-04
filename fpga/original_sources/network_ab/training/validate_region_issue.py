"""Isolated issuer for the explicit new 16-bit region descriptor ABI.

This checks EVERY field of the complete frame program against software ABI
decoding, two randomized protocol runs, and error suppression. Engines in
this test are protocol models, not CNN/DDR/board implementations.
"""
import argparse,json,re,shutil
from pathlib import Path
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .compile_region_program import CSCHEMA,DSCHEMA,offsets
from .build_region_rtl import build,run
from .prepare_region_maps import project


def source(text,manifest):
    cs,ds=manifest['streams'];cpos,cbits=offsets(cs['schema']);dpos,dbits=offsets(ds['schema']);nodes=manifest['nodes'];count=manifest['commands']
    bits=manifest.get('index_bits',16);width=manifest.get('stripe_width',64)
    text=text.replace('input wire [13:0] compute_records,dma_records',f'input wire [{bits-1}:0] compute_records,dma_records')
    text=text.replace('output reg [13:0] compute_completed,dma_completed',f'output reg [{bits-1}:0] compute_completed,dma_completed')
    text=text.replace('output wire [14:0] debug_program_index',f'output wire [{bits-1}:0] debug_program_index')
    text=text.replace('reg [13:0] ctotal,dtotal',f'reg [{bits-1}:0] ctotal,dtotal')
    text=text.replace('output wire dstore,drgb,','output wire dstore,drgb,dregion,')
    for prefix,pos in [('c',cpos),('d',dpos)]:
        pat=rf'wire \[(\d+):0\] {prefix}(\d+)={prefix}data\[\d+\+:\d+\];'
        def field(m):
            key='f'+m[2];start,bits=pos[key];return f'wire [{bits-1}:0] {prefix}{m[2]}={prefix}data[{start}+:{bits}];'
        text,n=re.subn(pat,field,text);assert n==len(pos)-(2 if prefix=='c' else 2),(prefix,n)
    text=text.replace('cdata[13:0]',f'cdata[{bits-1}:0]').replace('ddata[13:0]',f'ddata[{bits-1}:0]').replace('cdata[28:14]',f'cdata[{2*bits-1}:{bits}]')
    text=text.replace('cdata[511:428]',f'cdata[511:{cbits}]').replace('ddata[127:120]',f'ddata[127:{dbits}]')
    text=text.replace('c1>=29',f'c1>={nodes}').replace('d1>=29',f'd1>={nodes}').replace('>=19229',f'>={count}')
    expected_width=str(width) if 1280%width==0 else f"(rgb_x==11'd{1280//width*width} ? 9'd{1280%width}:9'd{width})"
    text=text.replace('!d0 || !d6 || d2!=256',f'!d0 || d6 || !dregion || d2!={expected_width}')
    text=text.replace("rgb_x+11'd256","rgb_x+{2'd0,d2}").replace("nodes_completed==29'h1fffffff",f"nodes_completed==29'h{manifest['complete_node_mask']:x}")
    text=text.replace('assign dstore=d0;',f"assign dregion=ddata[{dpos['region'][0]}];assign dstore=d0;")
    # Header prefetch refers to the same new schema as active commands.
    old_offsets={226:'f17',241:'f18',256:'f19',29:'f1',49:'f5'}
    text=re.sub(r'next_data\[(\d+)\+:(\d+)\]',lambda m:f"next_data[{cpos[old_offsets[int(m[1])]][0]}+:{m[2]}]",text)
    return text


def main():
    p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    manifest=json.loads((a.program/'program.json').read_text());root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False)
    hw=ROOT.parent/'hw/efinity/prototypes/frame_execution';original=hw/'frame_program_issue.sv';dst=root/original.name
    dst.write_text(source(original.read_text(),manifest));cs,ds=manifest['streams'];n=manifest['commands'];c=cs['count'];d=ds['count']
    fields=[[0]*32 for _ in range(n)];dma_indices=[]
    for e,schema,meta in [(0,cs['schema'],cs),(1,ds['schema'],ds)]:
        raw=(a.program/(meta['name']+'.bin')).read_bytes();assert file_hash(a.program/(meta['name']+'.bin'))==meta['sha256'];pos,bits=offsets(schema)
        for off in range(0,len(raw),meta['record_bytes']):
            word=int.from_bytes(raw[off:off+meta['record_bytes']],'little');v={k:(word>>o)&((1<<w)-1) for k,(o,w) in pos.items()};assert word>>bits==0
            index=v['f31'];f=fields[index];assert not any(f)
            for key,value in v.items():
                if key.startswith('f'):f[int(key[1:])]=value
            if e:dma_indices.append(index)
            else:f[0]=2;assert index==v['program_index']
        shutil.copy2(a.program/(meta['name']+'.hex'),root/(meta['name']+'.hex'))
    assert all(f[2]>0 for f in fields)
    (root/'program.hex').write_text(''.join(f'{sum(v<<(32*i) for i,v in enumerate(f)):0256x}\n' for f in fields))
    (root/'dma_commands.hex').write_text(''.join(f'{i:08x}\n' for i in dma_indices))
    original_tb=hw/'tb_frame_program_issue.sv';tb=original_tb.read_text()
    for old,new in [('8789',str(c)),('8788',str(c-1)),('10440',str(d)),('10439',str(d-1)),('19228',str(n-1))]:tb=tb.replace(old,new)
    tb=tb.replace('module tb_frame_program_issue;','module tb_region_issue;\n wire dregion;')
    bits=manifest.get('index_bits',16)
    tb=tb.replace('[13:0]',f'[{bits-1}:0]').replace("14'd",f"{bits}'d").replace('wire [14:0] debug_program_index',f'wire [{bits-1}:0] debug_program_index')
    tb=tb.replace('.drgb(drgb),','.drgb(drgb),.dregion(dregion),')
    tb=tb.replace('drgb!==1\'(df(6))',"dregion!==1'(df(0)) || drgb!==1'(df(6))")
    tb=tb.replace('ticks>500000','ticks>3000000').replace("nodes_completed!=29'h1fffffff",f"nodes_completed!=29'h{manifest['complete_node_mask']:x}")
    tbpath=root/'tb_region_issue.sv';tbpath.write_text(tb)
    write_json(root/'inputs.json',dict(program=str(a.program),program_sha256=file_hash(a.program/'program.json'),source=str(original),source_sha256=file_hash(original),
        tb_source=str(original_tb),tb_source_sha256=file_hash(original_tb),derived={str(f):file_hash(f) for f in [dst,tbpath]},production_modified=False))
    build(root,[dst,tbpath],'tb_region_issue');seconds=run([root/'obj/Vtb_region_issue.exe','+VECTORS='+root.as_posix(),'+DESCRIPTORS='+root.as_posix()],root,root/'simulation.log',90)
    text=(root/'simulation.log').read_text();assert 'PASS PROGRAM_ISSUE' in text
    write_json(root/'result.json',dict(status='FULL_FRAME_ABI_ISSUER_PROTOCOL_PASS',seconds=seconds,lines=[l for l in text.splitlines() if l.startswith('PASS')],
        actual_compute=False,actual_DDR=False,four_state=False,board=False))
    project(ROOT/'runs'/RUN/(a.name+'-map'),'frame_program_issue',[dst]);project(ROOT/'runs'/RUN/(a.name+'-reference-map'),'frame_program_issue',[original])
    print(text[-1500:],flush=True)


if __name__=='__main__':main()
