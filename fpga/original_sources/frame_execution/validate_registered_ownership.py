"""Four-state integer and transaction oracles, adjusted only for new grant latency."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,run,sha
from registered_ownership_board import audit_delta,once

def router_tb():
    source=HERE/'variants/fixed_early_collision_test/tb_row_pair_aux_router.sv'
    t=source.read_text().split('\n// Three fixed clients')[0]
    lo=t.index(' wire [17:0] ref_request_ready');hi=t.index(' integer collision_checks',lo)
    t=t[:lo]+t[hi:]
    for old,new in [('ref_aux_write_half','aux_write_half'),('ref_aux_address','aux_address')]:t=t.replace(old,new)
    lo=t.index(' row_pair_aux_router_reference ref_dut');hi=t.index('  if(!reset && !cancel)begin',lo)
    t=t[:lo]+' always @(negedge clk)begin\n  #1;\n'+t[hi:]
    # Two clock edges per ownership grant: retain the original fairness and
    # conservation thresholds by offering twice as many cycles, not fewer checks.
    t=t.replace('13200','26400').replace('1200','2400').replace('%3001','%6001').replace('%2003','%4003')
    t=once(t,'  @(negedge clk);request_valid=0;checking=0;inject=1;@(negedge clk);inject=0;',
       '  do @(posedge clk);while(!request_ready[0]);\n  @(negedge clk);request_valid=0;checking=0;inject=1;@(negedge clk);inject=0;')
    return source,t

def rq_tb():
    source=ROOT/'tmp/frame-execution/bias-range-rq-reg1/tb_fixed_split_rq.sv';t=source.read_text()
    t=once(t,'delay<=8','delay<=9')
    t=once(t,'if(delay<=3 && !dut.preparing)','if(delay>=2 && delay<=4 && !dut.preparing)')
    t=once(t,'if(delay>=4 && dut.preparing)','if(delay>=5 && dut.preparing)')
    t=once(t,'cancel_phases=8','cancel_phases=9')
    t=once(t,'   if(!busy)$fatal(1,"CANCEL_NOT_ACTIVE");',
       '   if(!busy)$fatal(1,"CANCEL_NOT_ACTIVE");\n   if(delay==1 && !dut.selected)$fatal(1,"GRANT_CANCEL_NOT_ACTIVE");')
    return source,t

def main():
    p=argparse.ArgumentParser();p.add_argument('--board',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();board=a.board.resolve();root=a.root.resolve();audit_delta(board)
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    results=[]
    try:
        for kind,stem,tbfn in [('rq','fixed_split_rq',rq_tb),('router','row_pair_aux_router',router_tb)]:
            r=root/kind;r.mkdir();tb_source,tb_text=tbfn();tb=r/('tb_'+stem+'.sv');tb.write_text(tb_text,encoding='utf-8')
            rtl=board/'assembly'/('fixed_rq32_mixed.sv' if kind=='rq' else 'row_pair_aux_router.sv')
            sources=[rtl]+([EF/'sim_models/verilog'/n for n in ('efx_dsp48.v','efx_srl8.v')] if kind=='rq' else [])
            files=[]
            for f in sources:
                dst=r/f.name;shutil.copy2(f,dst);files.append(dict(source=str(f),snapshot=str(dst),sha256=sha(dst)))
            files.append(dict(source=str(tb_source),source_sha256=sha(tb_source),snapshot=str(tb),sha256=sha(tb)))
            (r/'inputs.json').write_text(json.dumps(dict(inputs=files,board=str(board),board_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__)),indent=2))
            run([MS/'vlib.exe','work'],r,'vlib.log',30)
            body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[v['snapshot'] for v in files]],r,'compile.log',60)
            assert 'Errors: 0, Warnings: 0' in body
            for seed in ([71] if kind=='rq' else [71,391,1027]):
                log=f'simulation_{seed}.log'
                body,seconds=run([MS/'vsim.exe','-c','work.tb_'+stem,f'+SEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,log,90)
                lines=re.findall(r'^# (PASS (?:FIXED_SPLIT_RQ|ROW_PAIR_AUX).*)$',body,re.M)
                assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
                row=dict(kind=kind,seed=seed,seconds=seconds,line=lines[0],log_sha256=sha(r/log))
                results.append(row);print(json.dumps(row),flush=True)
            for rec in files:assert sha(rec['snapshot'])==rec['sha256']
        result=dict(status='REGISTERED_OWNERSHIP_FOUR_STATE_SHORT_PASS',results=results,
                    board=str(board),board_inputs_sha256=sha(board/'inputs.json'),hardware_access=False)
        (root/'result.json').write_text(json.dumps(result,indent=2))
    except Exception as exc:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(exc),traceback=traceback.format_exc(),completed=results),indent=2));raise
if __name__=='__main__':main()
