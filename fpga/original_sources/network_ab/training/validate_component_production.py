"""Actual C3 six-bank fetch/issue/header/MAC/RQ/DMA chain; isolated copy.

Only original RGB enters the datapath. Independent CPU int64 oracle, actual
trained package and per-byte output coverage. Short tests intentionally cancel.
"""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image
from .train_luma_regions import ROOT,RUN
from .runtime import file_hash,write_json
from .probe_boundary_budget import HW
from .build_region_rtl import build,run
from .validate_region_production import BODY
from .probe_region_class_int8 import execute


def main():
    p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--rows',type=int,default=8);p.add_argument('--full-frame',action='store_true');a=p.parse_args()
    assert 1<=a.rows<=720;root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False);base=ROOT/'runs'/RUN;ref=ROOT.parent/'tmp/frame-execution/row-pair-production-frame1'
    m=json.loads((a.program/'program.json').read_text());assert m['status']=='C3_ROW_PAIR_COMPILER_ONLY_PASS';cs,ds=m['streams'];bits=m['index_bits'];width=m['stripe_width'];rows=a.rows
    if a.full_frame:width=1280;rows=720
    paths={};records=[];proof=json.loads((ref/'result.json').read_text());assert proof['status']=='CONTINUOUS_720P_RTL_NUMERIC_MEASUREMENT_PASS' and file_hash(ref/'inputs.json')==proof['inputs_sha256']
    for r in json.loads((ref/'inputs.json').read_text())['inputs']:
        f=Path(r['snapshot']);assert file_hash(f)==r['sha256']
        if not f.name.startswith('tb_'):paths[f.name]=f
    assert all(name in paths for name in ['next_header_prefetch.sv','resident_read_arbiter2.sv','axi_read_merge2.sv','cnn_frame_datapath.sv','cnn_production_frame.sv'])
    paths.update({'frame_program_issue.sv':base/'rtl-component72-pair-issue-v1/frame_program_issue.sv','frame_program_fetch.sv':base/'rtl-component72-pair-fetch-v1/frame_program_fetch.sv','row_pair_skip_fuse4.sv':base/'rtl-component-row-pair-skip-v1/row_pair_skip_fuse4.sv','row_pair_dma.sv':base/'rtl-component-row-pair-dma-v1/row_pair_dma.sv','component_pixel_decode.sv':Path(__file__).with_name('component_pixel_decode.sv')})
    for name,f in paths.items():
        dst=root/name;shutil.copy2(f,dst);records.append(dict(source=str(f),source_sha256=file_hash(f),snapshot=str(dst)))
    f=root/'row_pair_compute_cluster.sv';s=f.read_text();old='cfg_in_groups!=4 || cfg_out_groups!=4';assert s.count(old)==1;s=s.replace(old,'(cfg_in_groups!=1 && cfg_in_groups!=2 && cfg_in_groups!=4) || cfg_in_groups!=cfg_out_groups');f.write_text(s)
    f=root/'cnn_frame_datapath.sv';s=f.read_text();assert s.count('input wire dstore,drgb,')==1;s=s.replace('input wire dstore,drgb,','input wire dstore,drgb,dregion,').replace('.cmd_rgb(drgb),','.cmd_rgb(drgb),.cmd_region(dregion),');f.write_text(s)
    f=root/'cnn_production_frame.sv';s=f.read_text().replace('[13:0] compute_completed',f'[{bits-1}:0] compute_completed').replace('[14:0] debug_program_index',f'[{bits-1}:0] debug_program_index')
    s=s.replace("14'd8789",f"{bits}'d{cs['count']}").replace("14'd10440",f"{bits}'d{ds['count']}").replace('wire  drgb;','wire  drgb;wire dregion;').replace('.drgb(drgb),','.drgb(drgb),.dregion(dregion),');f.write_text(s)
    sys.path.insert(0,str(HW));from build_production_cosim import ports_of
    _,ports=ports_of(s);tb='module tb_component_production;\n'
    for d,w,n in ports:tb+=f'{"reg" if d=="input" else "wire"} [{w-1}:0] {n}'+('=0;\n' if d=='input' else ';\n')
    tb+='cnn_production_frame dut(.*);always #5 clk=~clk;\n'
    for name in ['load_words.hex','compute.hex','dma.hex']:shutil.copy2(a.program/name,root/name)
    source_meta=json.loads((base/'rtl-graph-dma-v1/inputs.json').read_text());source=Path(source_meta['image']);assert file_hash(source)==source_meta['image_sha256'];original=np.asarray(Image.open(source).convert('RGB'))
    package_path=Path(m['package']);assert file_hash(package_path)==m['package_sha256'];package=json.loads(package_path.read_text());assert package['schema']=='component-regions-c3-int8-q17-v1'
    # Full 720 image oracle even for a short prefix: no crop boundary assumption.
    values=execute(package,(original.astype(np.int64)-128).transpose(2,0,1)[None],False)['rgb'][0,:,:rows,:width].copy();values[0]+=128
    y,co,cg=[np.asarray(levels,np.int64)[np.abs(values[i,...,None]-np.asarray(levels,np.int64)).argmin(-1)] for i,levels in enumerate(package['levels'])]
    gold=np.stack((y+co-cg,y+cg,y-co-cg),-1).clip(0,255).astype(np.uint8);reference=root/'golden-int64.png';Image.fromarray(gold).save(reference)
    for name,arr in [('source',original),('golden',gold)]:(root/(name+'.hex')).write_text(''.join(f'{int(b)^128:02x}\n' for b in arr.reshape(-1)),encoding='ascii')
    body=BODY.replace('COMPUTE_CRC',f"32'h{cs['crc32']:08x}").replace('DMA_CRC',f"32'h{ds['crc32']:08x}").replace('dut.cfg_row_planes','dut.cfg_row_banks')
    replacements={'34199':str(cs['count']-1),'28799':str(ds['count']-1),'04216600':f"{cs['address']+cs['count']*cs['record_bytes']:08x}",'04300000':f"{ds['address']:08x}",'04370800':f"{ds['address']+ds['count']*ds['record_bytes']:08x}",'1535':str(width*rows*3-1),'1536':str(width*rows*3),'192':str(width*3),'pixels=512':f'pixels={width*rows}','64x8':f'{width}x{rows}','y>=8':f'y>={rows}','output_rows<8':f'output_rows<{rows}','ticks>500000':'ticks>2000000','#6000000;':'#21000000;'}
    for old,new in replacements.items():body=body.replace(old,new)
    if a.full_frame:
        total=720*1280*3;body=body.replace(f'reg [{total-1}:0] seen=0;',f'reg seen[0:{total-1}];integer active_start=0;').replace('  reset=1;repeat(5)',f'  for(integer j=0;j<{total};j=j+1)seen[j]=0;\n  reset=1;repeat(5)').replace('if(!cancel && (error || frame_valid))','if(!cancel && error)').replace('ticks>2000000','ticks>15000000').replace('#21000000;','#151000000;').replace('  start_valid=1;do','  active_start=ticks;start_valid=1;do')
        begin=body.index('  while(output_rows<720)');end=body.index('  $finish;',begin)
        body=body[:begin]+f'''  while(!frame_valid)@(negedge clk);
  if(output_bytes!={total} || held_checks==0 || compute_completed!={cs['count']} || dma_completed!={ds['count']})fail("full coverage/counts");
  if(nodes_completed!=29'h{m['complete_node_mask']:x} || rgb_pixels!=921600)fail("full graph commit");
  for(integer j=0;j<{total};j=j+1)if(!seen[j])fail("missing RGB byte");
  if(read_left!=0 || write_left!=0 || rvalid || bvalid || busy)fail("published with pending transactions");
  $display("PASS REAL_COMPONENT_PRODUCTION_FRAME pixels=921600 bytes=%0d rows=%0d compute=%0d dma=%0d ticks=%0d active_ticks=%0d held=%0d read_bursts=%0d",output_bytes,output_rows,compute_completed,dma_completed,ticks,ticks-active_start,held_checks,reads_seen);
'''+body[end:]
    else:body=body.replace('REAL_REGION_PRODUCTION_PREFIX','REAL_COMPONENT_PRODUCTION_PREFIX')
    tbpath=root/'tb_component_production.sv';tbpath.write_text(tb+body);files=[root/n for n in paths]+[tbpath]
    write_json(root/'inputs.json',dict(program=str(a.program),program_sha256=file_hash(a.program/'program.json'),package_sha256=file_hash(package_path),source=str(source),source_sha256=file_hash(source),gold=str(reference),gold_sha256=file_hash(reference),inputs=records,compiled_hashes={str(f):file_hash(f) for f in files},production_modified=False,no_intermediate_injection=True,rows=rows,width=width,oracle_height=720,full_frame=a.full_frame))
    build(root,files,'tb_component_production');sec=run([root/'obj/Vtb_component_production.exe'],root,root/'simulation.log',3300 if a.full_frame else 600);body=(root/'simulation.log').read_text();label='FRAME' if a.full_frame else 'PREFIX';assert 'PASS REAL_COMPONENT_PRODUCTION_'+label in body
    write_json(root/'result.json',dict(status='C3_REAL_ROW_PAIR_FETCH_ISSUER_HEADER_MAC_RQ_DMA_'+label+'_PASS',seconds=sec,log=body,full_frame=a.full_frame,actual_DDR=False,fit=False,timing=False,board=False));print(body,flush=True)


if __name__=='__main__':main()
