"""Time-bounded real-vendor board startup, optionally with a public RTL shadow.

This probe never declares complete frames or extrapolates an idle-only budget.
"""
import argparse
import json
import os
from pathlib import Path
import re
from probe_board_dpi_cost import BIN, SIM, run
from run_units import ROOT, sha


def validate_probe_log(log, duration_us):
    assert f'BOUNDED_PUBLIC_STOP {duration_us}' in log
    assert not re.search(r'\*\* (?:Fatal|Error)| ERROR: |PUBLIC_DIFF .*t=',log)
    # A short window can end before an uninitialized CPU is exercised. A
    # missing ROM must reject even if the HDL checker has not yet diverged.
    assert not re.search(r'Failed to open readmem file',log), 'Missing ROM invalidates comparison'


def fused_wrapper(text,meta):
    """One delta-coalesced dispatcher; native time never runs ahead of Questa."""
    ni=meta['packing']['input_bits'];nc=meta['packing']['clock_bits'];no=meta['packing']['output_bits']
    first=text.index('chandle h;');last=text.index('// DDR response data',first)
    old=text[first:last]
    inputs=re.search(r'wire \[\d+:0\] inputs=.*?;',old)[0]
    clocks=re.search(r'wire \[\d+:0\] clocks=.*?;',old)[0]
    out_expr=re.search(r'assign (\{[^;]+?\})=\(seq_num>combo_num\)',old)[1]
    new=f'''chandle h;
{inputs}
{clocks}
// Remember four-state idle inputs exactly. Two-state storage would turn an
// unchanged X into a false change at every unrelated clock event.
logic [{ni-1}:0] last_inputs;logic [{nc-1}:0] last_clocks;
bit [{no-1}:0] temp,native_out;
bit initialized=0,last_tick=0;reg tick=0;
longint unsigned ignored,due=64'hffffffffffffffff,scheduled=64'hffffffffffffffff,wakeup=64'hffffffffffffffff;
assign {out_expr}=native_out;
initial h=board_public_create();
always @(inputs or clocks or tick)begin
 #0;
 if(!initialized || inputs!==last_inputs)begin
  ignored=board_public_combo(h,$time,inputs,temp,due);last_inputs=inputs;
 end
 if(!initialized || clocks!==last_clocks || tick!=last_tick || due==$time)begin
  ignored=board_public_seq(h,$time,clocks,temp,due);last_clocks=clocks;last_tick=tick;
 end
 initialized=1;native_out<=temp;
 if(due!=64'hffffffffffffffff && due!=scheduled)begin
  if(due<$time)$fatal(1,"Board fused native time reversed");
  scheduled=due;wakeup<=#(due-$time)due;
 end
end
always @(wakeup)if(wakeup==$time && wakeup==scheduled)tick=~tick;
task warm_video;board_public_warm(h);tick=~tick;endtask
'''
    return text[:first]+new+text[last:]


def zero_copy_wrapper(text, meta):
    """Use Questa's documented fixed int-array DPI fast argument path.

    Bit positions and native ABI stay identical. Only host argument storage
    changes; padding is explicit and every existing HDL check is retained.
    """
    ni=meta['packing']['input_bits'];nc=meta['packing']['clock_bits'];no=meta['packing']['output_bits']
    iw=(ni+31)//32;cw=(nc+31)//32;ow=(no+31)//32
    for kind,width,count in [('combo',ni,iw),('seq',nc,cw)]:
        old=f'input bit [{width-1}:0] data,output bit [{no-1}:0] result'
        new=f'input int unsigned data[0:{count-1}],inout int unsigned result[0:{ow-1}]'
        assert old in text;text=text.replace(old,new)
    text=text.replace(f'bit [{no-1}:0] combo_out,seq_out,temp;',
        f'int unsigned combo_out[0:{ow-1}],seq_out[0:{ow-1}],temp[0:{ow-1}];\n'
        f'int unsigned inputs_array[0:{iw-1}],clocks_array[0:{cw-1}];\n'
        f'wire [{iw*32-1}:0] inputs_padded={iw*32}\'(inputs);\n'
        f'wire [{cw*32-1}:0] clocks_padded={cw*32}\'(clocks);\n'
        f'wire [{ow*32-1}:0] combo_packed,seq_packed;\n'
        f'for(genvar k=0;k<{ow};k++)begin assign combo_packed[k*32+:32]=combo_out[k];assign seq_packed[k*32+:32]=seq_out[k];end\n')
    text=text.replace('(seq_num>combo_num)?seq_out:combo_out;',f'(seq_num>combo_num)?seq_packed[{no-1}:0]:combo_packed[{no-1}:0];')
    text=text.replace('always_comb combo_num=board_public_combo(h,$time,inputs,combo_out,combo_due);',
        f'always_comb begin for(integer k=0;k<{iw};k++)inputs_array[k]=inputs_padded[k*32+:32];combo_num=board_public_combo(h,$time,inputs_array,combo_out,combo_due);end')
    text=text.replace('seq_num<=board_public_seq(h,$time,clocks,temp,seq_due);',
        f'for(integer k=0;k<{cw};k++)clocks_array[k]=clocks_padded[k*32+:32];\n seq_num<=board_public_seq(h,$time,clocks_array,temp,seq_due);')
    return text


def qualification(name):
    if name.startswith('cut__cpu_alu__'):
        return 'reference.cpu.system_cores_0_logic_cpu.execute_arbitration_isValid'
    if name.startswith('probe__cpu__system_cores_0_logic_cpu_dBus_cmd_payload_'):
        return 'probe__cpu__system_cores_0_logic_cpu_dBus_cmd_valid'
    for prefix in ('cut__u_ddr3_top__s_axi_', 'cut__cpu__axiA_', 'probe__s_axi_', 'probe__raw_axi_'):
        if name.startswith(prefix):
            field=name[len(prefix):]
            for channel,fields in [('aw',('awid','awaddr','awlen','awsize','awburst','awlock','awcache','awprot')),
                                   ('ar',('arid','araddr','arlen','arsize','arburst','arlock','arcache','arprot')),
                                   ('w',('wdata','wstrb','wlast')),('r',('rid','rdata','rresp','rlast')),('b',('bid','bresp'))]:
                if field in fields:return prefix+channel+'valid'
    if name=='cut__cpu__io_apbSlave_0_PRDATA':return 'cut__cpu__io_apbSlave_0_PSEL && cut__cpu__io_apbSlave_0_PENABLE'
    if name.startswith('cut__dvi_encoder_m0__i_') and name[-5:] in ('bdata','gdata','rdata'):return 'cut__dvi_encoder_m0__i_de'
    if name in ('probe__hdmi_tx_bdata','probe__hdmi_tx_gdata','probe__hdmi_tx_rdata'):return 'probe__hdmi_tx_de'
    if name.startswith('probe__pair_pixel_') and not name.endswith('valid'):return 'probe__pair_pixel_valid'
    return "1'b1"


def main():
    p=argparse.ArgumentParser();p.add_argument('--native',type=Path,required=True)
    p.add_argument('--duration-us',type=int,default=100)
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--compare',action='store_true')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--batch',action='store_true')
    p.add_argument('--zero-copy',action='store_true')
    p.add_argument('--vector-pads',type=Path)
    p.add_argument('--fused',action='store_true')
    a=p.parse_args();root=a.native.resolve()
    assert root.is_relative_to(ROOT/'tmp') and 1<=a.duration_us<=10000 and 1<=a.timeout<=1800
    meta=json.loads((root/'partition.json').read_text())
    assert (root/'board_public.dll').is_file()
    name=('compare' if a.compare else 'profile' if a.profile else 'startup')+str(a.duration_us)
    if a.batch:name+='batch'
    if a.zero_copy:name+='array'
    if a.vector_pads:name+='pads'
    if a.fused:name+='fused'
    assert not(a.fused and a.zero_copy)
    assert not (root/(name+'.log')).exists(), 'Each experiment needs its own evidence name'
    env=os.environ.copy();env.update(PATH=str(BIN)+';'+env['PATH'],SALT_LICENSE_SERVER='E:/quatra/LR-195771_License.dat')
    wrapper=(root/'board_public_wrapper.sv').read_text()
    if a.zero_copy:wrapper=zero_copy_wrapper(wrapper,meta)
    if a.fused:wrapper=fused_wrapper(wrapper,meta)
    if a.vector_pads:
        pads=a.vector_pads.resolve();proof=json.loads((pads/'result.json').read_text())
        assert proof['status']=='DDR_GPIO_VECTOR_FOUR_STATE_EQUIVALENCE_PASS'
        assert sha(root/'EFX_GPIO_V3.v')==proof['source_sha256']
        assert sha(pads/'ddr_gpio_vector_sim.sv')==proof['vector_sha256']
        run([SIM/'vlog.exe','-sv',pads/'ddr_gpio_vector_sim.sv',pads/'ddr516_periphery_vector.sv'],root,name+'_pads_compile',env,30)
    else:
        run([SIM/'vlog.exe','-sv',root/'ddr516_periphery_model.sv'],root,name+'_pads_compile',env,30)
    if a.compare:
        declarations=[];connections=[];checks=[]
        for d,w,n in meta['resolved_ports']:
            connections.append('.'+n+'('+('ref_'+n if d=='output' else n)+')')
            if d=='output':
                declarations.append(f'wire [{w-1}:0] ref_{n};')
                # All known bits must match; undefined idle payloads are counted
                # separately, and the original qualified-X guards stay active.
                checks.append(f'if({qualification(n)})begin if(!$isunknown(ref_{n}))begin if({n}!==ref_{n})$fatal(1,"PUBLIC_DIFF {n} t=%t got=%h ref=%h",$time,{n},ref_{n});known_comparisons++;end else unknown_samples++;end')
        compare='\n'.join(declarations)+'\nboard_public_reference reference('+','.join(connections)+');\n'
        compare+='longint unsigned known_comparisons=0,unknown_samples=0;\n'
        compare+='always @(posedge process_clk_150 or negedge process_clk_150)begin #2;if($time>1000000)begin\n'+'\n'.join(checks)+'\nend end\n'
        compare+='final $display("PUBLIC_DIFF_COUNTS known=%0d unknown_idle_or_reset=%0d",known_comparisons,unknown_samples);\n'
        wrapper=wrapper.replace('task warm_video;',"task warm_video;reference.dly_cnt=27'h4000000;")
        wrapper=wrapper.replace('endmodule',compare+'\nendmodule')
        (root/'board_public_compare.sv').write_text(wrapper)
        reference=(root/'board_public_native.sv').read_text().replace('module board_public_native','module board_public_reference',1)
        widths={n:w for d,w,n in meta['resolved_ports']}
        reference=re.sub(r'output wire \[\$bits\([^\n]+?\)-1:0\] (\w+)',
                         lambda m:f'output wire [{widths[m[1]]-1}:0] {m[1]}',reference)
        extra_sources=[]
        if meta.get('cpu_alu_only'):
            cpu=(root/'EfxSapphireSoc_public.v').read_text()
            # Reference CPU uses its own original protected ALU, not feedback
            # from the native CPU. Otherwise a bad partition could self-mask.
            cpu=cpu[:re.search(r'\bmodule\s+EfxCPUSp1\b',cpu).start()]
            names=set(re.findall(r'\bmodule\s+(\w+)',cpu))
            cpu=re.sub(r'\b('+'|'.join(map(re.escape,names))+r')\b',lambda m:m[1]+'_public_reference',cpu)
            # Renaming Verilog identifiers must not rename the boot ROM files
            # inside $readmemb strings; both sides load the same frozen image.
            cpu=cpu.replace('EfxSapphireSoc_public_reference.v_toplevel_',
                            'EfxSapphireSoc.v_toplevel_')
            cpu=cpu.replace('$root.board_public_native.', '$root.tb_board_startup.dut.reference.')
            (root/'cpu_public_reference.v').write_text(cpu)
            extra_sources.append(str(root/'cpu_public_reference.v'))
            reference=reference.replace('EfxSapphireSoc cpu(', 'EfxSapphireSoc_public_reference cpu(')
        (root/'board_public_reference.sv').write_text(reference)
        sources=[f for f in meta['source_files'] if Path(f).name not in {'ti60f225_oob_top.v','cnn_production_frame_dpi.sv'}]
        sources += [str(root/'assembly/cnn_production_frame.sv'),str(root/'board_public_reference.sv'),str(root/'board_public_compare.sv')]
        sources += extra_sources
        run([SIM/'vlog.exe','-sv','-timescale','1ns/1ps','+define+den2048Mb+sg125+x16',
             *['+incdir+'+x for x in meta['include_dirs']],*sources],root,name+'_compile',env,120)
    else:
        selected=root/(name+'_wrapper.sv');selected.write_text(wrapper)
        run([SIM/'vlog.exe','-sv','-timescale','1ns/1ps',
             *['+incdir+'+x for x in meta['include_dirs']],selected],root,name+'_compile',env,30)
    do='onerror {quit -code 1}\nonbreak {quit -code 1}\n'
    if a.profile:do+='profile on -p\n'
    do+='set run_started [clock milliseconds]\n'
    do+=f'run {a.duration_us} us\n'
    do+='puts "PUBLIC_RUN_MILLISECONDS [expr {[clock milliseconds]-$run_started}]"\n'
    if a.profile:do+=f'profile summary -file {name}-summary.txt\nprofile report -ranked -file {name}-ranked.txt\n'
    do+=f'puts "BOUNDED_PUBLIC_STOP {a.duration_us}"\nquit -code 0\n'
    (root/(name+'.do')).write_text(do)
    wall=run([SIM/'vsim.exe','-batch' if a.batch else '-c','-warning','14408,16154','-voptargs=-warning 14408 -O5',
              '-sv_lib',root/'board_public','work.tb_board_startup','-do',name+'.do'],root,name,env,a.timeout)
    log=(root/(name+'.log')).read_text(errors='replace')
    validate_probe_log(log,a.duration_us)
    if a.compare:
        count=re.search(r'PUBLIC_DIFF_COUNTS known=(\d+) unknown_idle_or_reset=(\d+)',log)
        assert count and int(count[1])>1000
    run_ms=re.search(r'PUBLIC_RUN_MILLISECONDS (\d+)',log)
    result=dict(status='PUBLIC_BOARD_BOUNDED_COMPARISON_PASS' if a.compare else 'PUBLIC_BOARD_BOUNDED_STARTUP_PASS',
                simulated_us=a.duration_us,wall_seconds=wall,active_frame_speed=False,full_frame=False,
                run_milliseconds=int(run_ms[1]) if run_ms else None,
                batch=a.batch,fused=a.fused,zero_copy=a.zero_copy,
                wrapper_sha256=sha(root/'board_public_compare.sv' if a.compare else selected),
                vector_pads_proof_sha256=sha(a.vector_pads.resolve()/'result.json') if a.vector_pads else None,
                dll_sha256=sha(root/'board_public.dll'),partition_sha256=sha(root/'partition.json'),
                log_sha256=sha(root/(name+'.log')),events=re.findall(r'BOARD_(?:PROGRESS|BOOT_PHASE).*',log))
    (root/(name+'_result.json')).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
