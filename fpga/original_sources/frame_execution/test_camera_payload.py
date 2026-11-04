"""Bounded four-state primitive and paired router/real-RAM equivalence test."""
import argparse,json,os,re,shutil
from pathlib import Path
from run_units import ROOT,EF,sha,run
from camera_payload_owner import BASE
from native_local_drivers import graph
from probe_full_native_split_map import params
MS=Path('E:/quatra/questa_fse/win64')

def bench(text,groups):
    ps=params(text);n=len(groups)
    lines=['`timescale 1ns/1ps','module tb_payload;',
      'reg clk=0;always #5 clk=~clk;reg reset=1,cancel=0;',
      'reg [17:0] request_valid=0,response_ready=0;reg [179:0] request_address=0;reg [35:0] request_half=0;reg [2303:0] request_data=0;',
      'reg [5:0] read_valid=0,read_response_ready=0;reg [59:0] read_address=0;',
      'integer seed=91,checks=0,writes=0,reads=0,stalls=0,cancels=0,resets=0;reg [17:0] accepted=0;',
      f'reg ffclk=0;reg [2:0] ctrl=0;wire [{n-1}:0] ff_old,ff_new;',
      'function automatic symbol(input integer v);case(v)0:symbol=0;1:symbol=1;2:symbol=1\'bx;default:symbol=1\'bz;endcase endfunction',
      'function integer rnd(input integer lim);integer v;begin v=$random(seed);rnd=(v&32\'h7fffffff)%lim;end endfunction']
    for i,r in enumerate(groups):
        for tag,cell in [('old',r['owner_ff']),('new',r['new_cell'])]:
            par=','.join(f".{k}({w}'h{v:x})" for k,(w,v) in ps[cell].items())
            lines.append(f'EFX_FF #({par}) ff_{tag}_{i}(.CLK(ffclk),.D(ctrl[0]),.CE(ctrl[1]),.SR(ctrl[2]),.Q(ff_{tag}[{i}]));')
    # Two exact mapped data-mux masks, including the fused DMA-store gate.
    lines+=['reg [2:0] state=0,dat=0;wire sel=state[1] && !state[0] && state[2];wire [1:0] oldmux,newmux;',
        "EFX_LUT4 #(.LUTMASK(16'hacac)) mo0(.I0(dat[0]),.I1(dat[1]),.I2(sel),.I3(1'b0),.O(oldmux[0]));",
        "EFX_LUT4 #(.LUTMASK(16'hacac)) mn0(.I0(dat[0]),.I1(dat[1]),.I2(state[2]),.I3(1'b0),.O(newmux[0]));",
        "EFX_LUT4 #(.LUTMASK(16'h44f0)) mo1(.I0(dat[0]),.I1(dat[1]),.I2(dat[2]),.I3(sel),.O(oldmux[1]));",
        "EFX_LUT4 #(.LUTMASK(16'h44f0)) mn1(.I0(dat[0]),.I1(dat[1]),.I2(dat[2]),.I3(state[2]),.O(newmux[1]));"]
    widths={'request_ready':18,'response_valid':18,'response_was_write':18,'response_data':2304,
        'aux_valid':6,'aux_ready':6,'aux_address':60,'aux_write_half':12,'aux_data':768,
        'aux_response':6,'aux_response_ready':6,'aux_response_data':768,'aux_was_write':6,
        'busy':1,'error':1,'aux_write_collision':6,'read_ready':6,'read_response':6,'read_data':768}
    common=['clk','reset','cancel','request_valid','response_ready','request_address','request_half','request_data']
    rports=[k for k in widths if not k.startswith('read_')]
    aports=['aux_valid','aux_ready','aux_address','aux_write_half','aux_data','aux_response','aux_response_ready',
            'aux_response_data','aux_was_write','aux_write_collision','read_ready','read_response','read_data']
    for i in range(2):
        for k,w in widths.items():lines.append(f'wire [{w-1}:0] {k}_{i};')
        ports=[f'.{k}({k})' for k in common]+[f'.{k}({k}_{i})' for k in rports]
        ports += [f'.protected_read_fire(read_valid & read_ready_{i})','.protected_read_address(read_address)']
        lines.append(f'row_pair_aux_router{"_new" if i else ""} #(.READ_ONLY_SKIP(1)) router{i}('+','.join(ports)+');')
        ports=[f'.{k}({k})' for k in ['clk','reset','cancel','read_valid','read_response_ready','read_address']]+[f'.{k}({k}_{i})' for k in aports]
        lines.append(f'row_pair_arena6 #(.EXTERNAL_WRITE_COLLISION(1)) arena{i}('+','.join(ports)+');')
    lines.append('always @(posedge clk)begin accepted=request_ready_0 & request_valid;')
    for k in widths:
        if k in ('aux_data','response_data','read_data','aux_response_data'):continue
        lines.append(f'if({k}_0!=={k}_1)$fatal(1,"DIFF {k}");')
    lines+=['for(integer b=0;b<6;b++)begin',
      'if(aux_valid_0[b] && aux_ready_0[b])begin',
      'if(aux_data_0[b*128+:128]!==aux_data_1[b*128+:128])$fatal(1,"ACTIVE_PAYLOAD");',
      'if(|aux_write_half_0[b*2+:2])writes++;else reads++;end',
      'if(aux_valid_0[b] && !aux_ready_0[b])stalls++;',
      'if(aux_response_0[b] && aux_response_data_0[b*128+:128]!==aux_response_data_1[b*128+:128])$fatal(1,"RAM_B_RESPONSE");',
      'if(read_response_0[b] && read_data_0[b*128+:128]!==read_data_1[b*128+:128])$fatal(1,"RAM_A_RESPONSE");',
      'end checks++;end',
      'initial begin if($value$plusargs("SEED=%d",seed))begin end',
      'for(integer v=0;v<64;v++)begin ffclk=0;ctrl={symbol(v>>4),symbol((v>>2)&3),symbol(v&3)};#2;ffclk=1;#2;if(ff_old!==ff_new)$fatal(1,"FOURSTATE_CLONE");end',
      'for(integer s=0;s<8;s++)for(integer v=0;v<64;v++)begin state=3\'(s);dat={symbol(v>>4),symbol((v>>2)&3),symbol(v&3)};#1;',
      'if(state[1] && !(state[0] && state[2]) && oldmux!==newmux)$fatal(1,"QUALIFIED_FOURSTATE_MUX");end',
      '@(negedge clk);reset=0;',
      'for(integer t=0;t<1200;t++)begin',
      'if(t%200==0)$display("PROGRESS paired_cycle=%0d",t);',
      'reset=t>0 && t%181==0;cancel=t>0 && t%113==0;resets+=int\'(reset);cancels+=int\'(cancel);',
      'if(reset || cancel)begin request_valid=0;read_valid=0;response_ready=0;read_response_ready=0;end else begin',
      'if(error_0)$fatal(1,"UNEXPECTED_ERROR");',
      'for(integer k=0;k<18;k++)begin response_ready[k]=rnd(4)!=0;',
      'if(!request_valid[k] || accepted[k])begin request_valid[k]=rnd(4)!=0;request_address[k*10+:10]=10\'(rnd(1024));',
      'request_half[k*2+:2]=(k>=6 && k<12)?0:2\'(rnd(4));request_data[k*128+:128]={$random(seed),$random(seed),$random(seed),$random(seed)};end end',
      'for(integer b=0;b<6;b++)begin read_valid[b]=rnd(3)!=0;read_response_ready[b]=rnd(4)!=0;read_address[b*10+:10]=10\'(rnd(1024));end end',
      '@(negedge clk);end',
      'reset=1;request_valid=0;read_valid=0;@(negedge clk);reset=0;',
      '// Reject illegal skip writes, then prove cancellation recovery.',
      'request_valid[6]=1;request_half[13:12]=3;@(negedge clk);if(!error_0 || !error_1)$fatal(1,"ILLEGAL_SKIP_NOT_REJECTED");',
      'cancel=1;request_valid=0;@(negedge clk);cancel=0;@(negedge clk);if(error_0 || error_1 || busy_0 || busy_1)$fatal(1,"RECOVERY");',
      'if(writes<900 || reads<900 || stalls<100 || resets<6 || cancels<10)$fatal(1,"COVERAGE w=%d r=%d s=%d",writes,reads,stalls);',
      '$display("PASS PAYLOAD_PAIRED_RAM checks=%0d writes=%0d reads=%0d stalls=%0d reset=%0d cancel=%0d",checks,writes,reads,stalls,resets,cancels);$finish;end',
      'initial begin #1000000;$fatal(1,"TIMEOUT");end endmodule']
    return '\n'.join(lines)

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    src=a.source.resolve();root=a.root.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    proof=json.loads((src/'result.json').read_text());assert proof['status']=='PAYLOAD_SPECIALIZATION_GRAPH_PASS'
    for f,h in proof['inputs'].items():assert sha(f)==h
    for f,h in proof['files'].items():assert sha(src/f)==h
    root.mkdir();groups=json.loads((src/'groups.json').read_text());text=(src/'native_source.v').read_text()
    for n in ('efx_ff.v','efx_lut4.v','efx_dpram10.v'):shutil.copy2(EF/'sim_models/verilog'/n,root/n)
    for n in ('row_pair_aux_router.sv','row_pair_arena6.sv'):shutil.copy2(BASE/'assembly'/n,root/n)
    original=(root/'row_pair_aux_router.sv').read_text();old='assign data=select[2] ?';assert original.count(old)==1
    new=original.replace('module row_pair_aux_router #','module row_pair_aux_router_new #').replace(old,'assign data=held_owner[b][1] ?')
    (root/'row_pair_aux_router_new.sv').write_text(new);(root/'tb_payload.sv').write_text(bench(text,groups))
    sources=[x.name for x in root.iterdir() if x.suffix in ('.v','.sv')]
    meta=dict(source_result_sha256=sha(src/'result.json'),runner_sha256=sha(__file__),files={n:sha(root/n) for n in sources})
    (root/'inputs.json').write_text(json.dumps(meta,indent=2))
    for v in ('SALT_LICENSE_SERVER','MGLS_LICENSE_FILE','LM_LICENSE_FILE'):os.environ[v]='E:/quatra/LR-195771_License.dat'
    run([MS/'vlib.exe','work'],root,'vlib.log',timeout=20)
    body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*sources],root,'compile.log',timeout=35)
    assert 'Errors: 0, Warnings: 0' in body
    results=[]
    for seed in (91,137,407):
        body,secs=run([MS/'vsim.exe','-c','work.tb_payload',f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log',timeout=60)
        match=re.findall(r'^# PASS PAYLOAD_PAIRED_RAM.*$',body,re.M);assert len(match)==1 and 'Errors: 0, Warnings: 0' in body
        results.append(dict(seed=seed,seconds=secs,line=match[0],log_sha256=sha(root/f'seed{seed}.log')))
    result=dict(status='PAYLOAD_PAIRED_OFFICIAL_RAM_FOURSTATE_PASS',results=results,clone_checks=len(groups)*64,
        mux_input_cases=1024,inputs_sha256=sha(root/'inputs.json'),hardware_access=False,board_release=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)

if __name__=='__main__':main()
