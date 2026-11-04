"""Four-state cycle comparison of packed writer versus actual expand+writer."""
import argparse,json,re,traceback
from pathlib import Path
from run_units import ROOT,HERE,MS,sha,run


def bench(body):
    body=body.replace('module tb_paired_writer_equivalence;','module tb_compact_writer_equivalence;')
    start=body.index(' column_write8 #(.PACKED_PAIR_INPUTS(1)) dut(')
    end=body.index(' column_write8_reference reference(',start)
    body=body[:start]+''' wire [3:0] compact_ir;
 assign er=ref_er;
 column_write8 dut(.clk(clk),.reset(reset),.cancel(cancel),.pair_mode(pair_mode),.pixels(pixels),.rows(rows),.groups(groups),
  .expected_mask(expected_mask),.row_planes(row_planes),.row_bases(row_bases),.in_valid(iv),.in_ready(compact_ir),.in_data(id),.in_tag(it),
  .in_mask(im),.in_bias_overflow(32'd0),.in_range_error(32'd0),.read_locked(locked),.mem_req_valid(v),.mem_req_ready(ready),
  .mem_req_address(a),.mem_req_data(d),.mem_req_tag(t),.mem_rsp_valid(sv),.mem_rsp_ready(sr),.mem_rsp_tag(st),.mem_rsp_was_write(24'haaaaaa),
  .busy(busy),.error(error),.writes(writes),.acks(acks));
'''+body[end:]
    body=body.replace('column_write8_reference reference(', 'column_write8_reference #(.PACKED_PAIR_INPUTS(1)) reference(')
    body=body.replace('{er,v,sr,busy,error,writes,acks}!=={ref_er,ref_v,ref_sr,ref_busy,ref_error,ref_writes,ref_acks}',
                      '{compact_ir,v,sr,busy,error,writes,acks}!=={ir,ref_v,ref_sr,(ref_busy|expand_busy),(ref_error|expand_error),ref_writes,ref_acks}')
    body=body.replace('if(expand_error || (error && !fault_test))','if((expand_error || error) && !fault_test)')
    body=body.replace('x=issued[b]%(pair_mode ? 9:17);',
                      'x=issued[b]%(pair_mode ? ((int\'(pixels)+1)/2):int\'(pixels)); if(issued[b]%4==0)x=(pair_mode ? ((int\'(pixels)+1)/2):int\'(pixels))-1;')
    start=body.index('  // Inject an invalid odd-lane')
    end=body.index('  $display("PASS PAIRED_WRITER',start)
    body=body[:start]+'''  for(integer width_case=0;width_case<5;width_case=width_case+1)begin
   pixels=width_case==0 ? 1:width_case==1 ? 2:width_case==2 ? 3:width_case==3 ? 320:511;
   run_case(0,4,width_case%3,0);run_case(1,1,width_case%3,1);
  end
  pixels=17;
  // The public interface is now the packed source, not an impossible
  // independently corrupted post-expansion odd lane. Exercise faults here.
  for(integer bad_kind=0;bad_kind<8;bad_kind=bad_kind+1)begin
   flush_all(0);pair_mode=bad_kind<4;groups=1;expected_mask=pair_mode ? 8'h07:8'hff;
   fault_test=1;
   @(negedge clk);#2;iv=1;it=0;im=pair_mode ? 32'h3f:32'hff;id=64'h005544332211;
   case(bad_kind)
    0:it[15:0]=16'h2000;
    1:it[15:0]=16'd1;
    2:it[15:0]=16'd36;
    3:im[7:0]=8'hff;
    4:it[15:0]=16'h2000;
    5:it[15:0]=16'd68;
    6:im[7:0]=8'h7f;
    7:row_planes[1:0]=3;
   endcase
   repeat(3)@(negedge clk);#2;
   if(!error || v!=0)$fatal(1,"packed reject before access kind=%0d",bad_kind);
   rejects=rejects+1;iv=0;flush_all(0);fault_test=0;
  end
  run_case(1,1,2,0);
  if(holds<100 || edges<1000 || rejects!=8)$fatal(1,"writer coverage");
'''+body[end:]
    return body.replace('PASS PAIRED_WRITER','PASS COMPACT_WRITER')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--board',type=Path,required=True);a=p.parse_args()
    root=a.root.resolve();board=a.board.resolve();assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    (root/'frozen_runner.py').write_bytes(Path(__file__).read_bytes())
    try:
        indexed={Path(x['snapshot']).name:x for x in json.loads((board/'inputs.json').read_text())['inputs']}
        reference=Path(indexed['column_write8.sv']['snapshot']);expand=Path(indexed['result_pair_expand4.sv']['snapshot'])
        candidate=HERE/'variants/compact_writer/column_write8.sv';tb=HERE/'tb_paired_writer_equivalence.sv';records=[]
        for src in (reference,expand,candidate,tb):
            if src in (reference,expand):assert sha(src)==indexed[src.name]['sha256']
            dst=root/('column_write8_reference.sv' if src==reference else src.name)
            body=src.read_text()
            if src==reference:body=body.replace('module column_write8 #','module column_write8_reference #',1)
            if src==tb:body=bench(body)
            dst.write_text(body);records.append(dict(source=str(src),source_sha256=sha(src),snapshot=str(dst),sha256=sha(dst)))
        (root/'inputs.json').write_text(json.dumps(dict(inputs=records,board_inputs_sha256=sha(board/'inputs.json'),runner_sha256=sha(__file__)),indent=2))
        run([MS/'vlib.exe','work'],root,'vlib.log')
        body,_=run([MS/'vlog.exe','-sv','-timescale','1ns/1ps',*[x['snapshot'] for x in records]],root,'compile.log')
        assert 'Errors: 0, Warnings: 0' in body
        results=[]
        for seed in (71,391,1027):
            body,seconds=run([MS/'vsim.exe','-c','work.tb_compact_writer_equivalence',f'-gSEED={seed}','-do','onerror {quit -code 1}; run -all; quit -code 0'],root,f'seed{seed}.log',timeout=90)
            lines=re.findall(r'^# (PASS COMPACT_WRITER.*)$',body,re.M)
            assert len(lines)==1 and 'Errors: 0, Warnings: 0' in body
            results.append(dict(seed=seed,line=lines[0],seconds=seconds,log_sha256=sha(root/f'seed{seed}.log')))
        for row in records:assert sha(row['source'])==row['source_sha256'] and sha(row['snapshot'])==row['sha256']
        result=dict(status='COMPACT_WRITER_FOUR_STATE_CYCLE_EQUIVALENCE_PASS',results=results,inputs_sha256=sha(root/'inputs.json'),whole_board=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
