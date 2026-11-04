"""Bounded four-state replay of one real-RAM fused case and five abort stages."""
import argparse,json,re,shutil,subprocess,sys,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--sparse-init',action='store_true');a=p.parse_args()
    root=a.root.resolve();ref=a.reference.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    (root/'frozen_runner.py').write_bytes(Path(__file__).read_bytes())
    try:
        result=json.loads((ref/'result.json').read_text());meta=json.loads((ref/'native_inputs.json').read_text())
        assert sha(ref/'native_inputs.json')==result['inputs_sha256']
        assert sha(ref/'simulation.log')==result['log_sha256']
        assert 'PASS RESIDUAL_ABORT_PROBE stages=5 tag_fault=1' in (ref/'simulation.log').read_text()
        snapshot=Path(meta['snapshot']);source=json.loads((snapshot/'inputs.json').read_text())
        assert sha(snapshot/'inputs.json')==meta['source_manifest_sha256']
        shutil.copytree(snapshot/'vectors',root/'vectors')
        lines=(root/'vectors/cases.txt').read_text().splitlines()[1:]
        chosen=[x for x in lines if x.split()[0]=='9'];assert len(chosen)==1
        (root/'vectors/cases.txt').write_text('1\n'+chosen[0]+'\n')
        fields=chosen[0].split();assert fields[2]=='1' and fields[4:7]==['4','4','4']
        ranges=[]
        if a.sparse_init:
            bases=int(fields[14],16);span=int(fields[11]);height=int(fields[4])
            skip_bases=int((root/'vectors/case_skip_bases.hex').read_text().splitlines()[9],16)
            delta=int((root/'vectors/case_skip_delta.hex').read_text().splitlines()[9],16)
            skip_span=((int(fields[3])+delta+3)//4)*int(fields[6])
            ranges=sorted(set([(bases>>(10*y)&1023,span) for y in range(height)]+
                             [(skip_bases>>(10*y)&1023,skip_span) for y in range(height)]))
            assert all(lo%2==0 and n%2==0 and 0<=lo<lo+n<=1024 for lo,n in ranges)
        records=[]
        for row in source['inputs']:
            src=Path(row['snapshot']);assert sha(src)==row['sha256'];dst=root/src.name
            if src.name=='tb_compute_column_cluster.sv':
                body=src.read_text();needle=' || switches<4';assert body.count(needle)==1
                # This focused test covers residual return stalls/abort. MAC
                # stalls and operator switching belong to the bound 24-case
                # native regression, not necessarily this small residual case.
                body=body.replace(needle,'')
                needle=' || mac_held_checks==0';assert body.count(needle)==1
                body=body.replace(needle,' || skip_response_hold_checks==0')
                body=body.replace(' integer checks=0,blocks=0,',
                    ''' integer skip_response_hold_checks=0;
 always @(posedge clk)if(run_mode && !reset && !flush)
  for(integer pp=0;pp<24;pp=pp+1)
   if(sv[pp] && dut.skip_locked[pp] && !sr[pp])
    skip_response_hold_checks=skip_response_hold_checks+1;
 integer checks=0,blocks=0,''',1)
                body=body.replace('  $display("COVERAGE blocks=',
                    '  $display("RESIDUAL_RETURN_HOLD checks=%0d",skip_response_hold_checks);\n  $display("COVERAGE blocks=',1)
                # Keep MAC payload-stability assertions, even though observing
                # a MAC hold is not mandatory for this particular short case.
                needle="pause_writes=(mode==1 && kind==1 && width==80 && height==2) ? 10'd512 : 10'd0;"
                assert body.count(needle)==1
                body=body.replace(needle,"pause_writes=mode==1 ? 10'd512 : 10'd0;")
                needle='if(hold_skip_responses)rsp_gate=rsp_gate & ~dut.skip_locked;'
                assert body.count(needle)==1
                body=body.replace(needle,'if(hold_skip_responses || (run_mode && pause_writes!=0))rsp_gate=rsp_gate & ~dut.skip_locked;')
                dst.write_text(body)
                if a.sparse_init:
                    predicate=' || '.join(f'(addr>={lo} && addr<{lo+n})' for lo,n in ranges)
                    helper=''' function automatic initialized_address(input integer addr);
 begin initialized_address='''+predicate+''';end endfunction
 always @(posedge clk)if(run_mode && !reset && !flush)begin
  for(integer pp=0;pp<24;pp=pp+1)
   if(av[pp] && ar[pp] && !aw[pp] && !initialized_address(int'(aa[pp*10+:10])))
    fail("read outside explicitly initialized fixture");
 end
'''
                    body=body.replace(' task initialize;',helper+' task initialize;',1)
                    needle='for(integer a=0;a<1024;a=a+2)begin'
                    assert body.count(needle)==1
                    body=body.replace(needle,'for(integer a=0;a<1024;a=a+2)if(initialized_address(a))begin',1)
                    dst.write_text(body)
            else:shutil.copy2(src,dst)
            records.append(dict(source=str(src),source_sha256=row['sha256'],snapshot=str(dst),sha256=sha(dst)))
        (root/'inputs.json').write_text(json.dumps(dict(inputs=records,reference_sha256=sha(ref/'result.json'),
          selected_case=9,sparse_init=a.sparse_init,initialized_ranges=ranges,forced_backpressure_cycles=512,
          required_stall_coverage='residual_return',mac_stall_coverage='bound_native_24_blocks',
          vectors={f.name:sha(f) for f in (root/'vectors').iterdir() if f.is_file()},runner_sha256=sha(__file__)),indent=2))
        run([MS/'vlib.exe','work'],root,'vlib.log')
        body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[r['snapshot'] for r in records]],root,'compile.log')
        assert 'Errors: 0, Warnings: 0' in body
        dump=root/'readback.hex';checker=HERE.parent/'coefficient_store/validate_readback.py'
        do='onerror {quit -code 1}\nrun 120 us\n'
        do+='if {[examine -radix unsigned sim:/tb_compute_column_cluster/verify_checkpoint] != 1} {error "Missing actual readback"}\n'
        do+='puts [exec {'+Path(sys.executable).as_posix()+'} -B {'+checker.as_posix()+'} --image-dir {'+(root/'vectors').as_posix()+'} --readback {'+dump.as_posix()+'}]\n'
        do+='force -deposit sim:/tb_compute_column_cluster/software_verified 1\nrun -all\nquit -code 0\n'
        (root/'run.do').write_text(do)
        body,seconds=run([MS/'vsim.exe','-c','work.tb_compute_column_cluster','+VECTORS='+(root/'vectors').as_posix(),'+DUMP='+dump.as_posix(),'-do','do run.do'],root,'simulation.log',timeout=240)
        assert 'PASS RESIDUAL_ABORT_PROBE stages=5 tag_fault=1' in body
        hold=re.findall(r'^# RESIDUAL_RETURN_HOLD checks=(\d+)$',body,re.M)
        assert len(hold)==1 and int(hold[0])>0
        assert 'ACTUAL_RTL_READBACK_SOFTWARE_VERIFIED' in body and 'Errors: 0, Warnings: 0' in body
        assert not re.search(r'\bFAIL\b|\*\* (Error|Fatal)',body)
        line=re.findall(r'^# (PASS COMPUTE_CLUSTER.*)$',body,re.M);assert len(line)==1 and 'blocks=2 ' in line[0]
        for row in records:assert sha(row['snapshot'])==row['sha256']
        out=dict(status='RESIDUAL_ACTIVE_ABORT_FOUR_STATE_PASS',line=line[0],abort_stages=5,seconds=seconds,
          inputs_sha256=sha(root/'inputs.json'),log_sha256=sha(root/'simulation.log'),
          residual_return_hold_checks=int(hold[0]),full_frame=False)
        (root/'result.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
