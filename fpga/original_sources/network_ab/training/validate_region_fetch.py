"""Bounded 16-bit descriptor fetch/CRC, using the existing two-slot hardware."""
import argparse,json,shutil
from pathlib import Path
from .region_detail_target import ROOT,RUN
from .runtime import file_hash,write_json
from .build_region_rtl import build,run
from .prepare_region_maps import project


def main():
    p=argparse.ArgumentParser();p.add_argument('--program',type=Path,required=True);p.add_argument('--name',required=True);a=p.parse_args()
    m=json.loads((a.program/'program.json').read_text());cs,ds=m['streams'];nc,nd=cs['count'],ds['count'];root=ROOT/'runs'/RUN/a.name;root.mkdir(exist_ok=False)
    hw=ROOT.parent/'hw/efinity/prototypes/frame_execution';original=hw/'frame_program_fetch.sv';s=original.read_text()
    bits=m.get('index_bits',16);daddr=f"{ds['address']:08x}";coaddr=f"{m['coefficient_address']:08x}"
    s=s.replace('[13:0]',f'[{bits-1}:0]').replace("{18'd0,compute_records}",f"{{{32-bits}'d0,compute_records}}").replace("{18'd0,dma_records}",f"{{{32-bits}'d0,dma_records}}")
    s=s.replace("{14'd0,dfetched,4'd0}",f"{{{28-bits}'d0,dfetched,4'd0}}").replace("{12'd0,cfetched,6'd0}",f"{{{26-bits}'d0,cfetched,6'd0}}")
    s=s.replace('8789',str(nc)).replace('10440',str(nd)).replace('04100000',daddr).replace('04200000',coaddr)
    dst=root/original.name;dst.write_text(s);original_tb=hw/'tb_frame_program_fetch.sv';tb=original_tb.read_text()
    for old,new in [('8789',str(nc)),('8788',str(nc-1)),('10440',str(nd)),('10439',str(nd-1)),('19229',str(nc+nd)),('45596',str(nc*4+nd))]:tb=tb.replace(old,new)
    tb=tb.replace('module tb_frame_program_fetch;','module tb_region_fetch;').replace('[13:0]',f'[{bits-1}:0]')
    tb=tb.replace('CCRC=0,DCRC=0',f"CCRC=32'h{cs['crc32']:08x},DCRC=32'h{ds['crc32']:08x}")
    tb=tb.replace('04100000',daddr).replace('04200000',coaddr).replace('ticks>1500000','ticks>4000000')
    needle='  reset_case(8);launch();'
    assert needle in tb
    tb=tb.replace(needle,f'''  reset_case(7);crecords={nc+1};launch();repeat(3)@(negedge clk);
  if(!error || busy || accepted!=0 || fault_code!=2)$fatal(1,"compute overflow extent");
  reset_case(7);drecords={nd+1};launch();repeat(3)@(negedge clk);
  if(!error || busy || accepted!=0 || fault_code!=2)$fatal(1,"DMA overflow extent");
'''+needle)
    tb=tb.replace('invalid_extent=1','invalid_extent=3')
    tbpath=root/'tb_region_fetch.sv';tbpath.write_text(tb)
    for meta in (cs,ds):
        assert file_hash(a.program/(meta['name']+'.bin'))==meta['sha256'];shutil.copy2(a.program/(meta['name']+'.hex'),root/(meta['name']+'.hex'))
    write_json(root/'inputs.json',dict(program=str(a.program),program_sha256=file_hash(a.program/'program.json'),source=str(original),source_sha256=file_hash(original),tb_source=str(original_tb),tb_sha256=file_hash(original_tb),derived={str(f):file_hash(f) for f in [dst,tbpath]},production_modified=False))
    build(root,[dst,tbpath],'tb_region_fetch');seconds=run([root/'obj/Vtb_region_fetch.exe','+VECTORS='+root.as_posix()],root,root/'simulation.log',90)
    text=(root/'simulation.log').read_text();assert 'PASS PROGRAM_FETCH' in text
    write_json(root/'result.json',dict(status='FULL_FRAME_DESCRIPTOR_FETCH_CRC_RECOVERY_PASS',seconds=seconds,lines=[l for l in text.splitlines() if l.startswith('PASS')],actual_DDR=False,four_state=False,board=False))
    project(ROOT/'runs'/RUN/(a.name+'-map'),'frame_program_fetch',[dst]);project(ROOT/'runs'/RUN/(a.name+'-reference-map'),'frame_program_fetch',[original]);print(text[-2000:],flush=True)


if __name__=='__main__':main()
