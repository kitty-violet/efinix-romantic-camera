"""Deterministic cluster interface replacement; stdout is installed with apply_patch."""
import argparse
from pathlib import Path


def transform(body):
    start=body.index(' wire [7:0] ev,er;wire [511:0] ed;')
    end=body.index(' always @(posedge clk)begin',start)
    old=body[start:end]
    assert 'result_pair_expand4 expand(' in old and '.PACKED_PAIR_INPUTS(1)' in old
    new=''' // Packed RQ results stay four-wide through the bank payload selector.
 assign expand_busy=1'b0;assign expand_error=1'b0;
 column_write8 writer(.clk(clk),.reset(reset),.cancel(flush),.pair_mode(rgb_pair),.pixels(pixels),.rows(rows),.groups(go),
  .expected_mask(mask),.row_planes(out_planes),.row_bases(out_bases),
  .in_valid(fused ? (feedback_valid & 4'b1100):ov),.in_ready(writer_ready),
  .in_data(fused ? feedback_data:od),.in_tag(fused ? feedback_tag:ot),
  .in_mask(om),.in_bias_overflow(ob),.in_range_error(oe),.read_locked(rdlocked),
  .mem_req_valid(wv),.mem_req_ready(wr),.mem_req_address(wa),.mem_req_data(wd),.mem_req_tag(wt),
  .mem_rsp_valid(sv & sw),.mem_rsp_ready(wsready),.mem_rsp_tag(st),.mem_rsp_was_write(sw),
  .busy(writer_busy),.error(writer_error),.writes(writes),.acks(acks));
'''
    return body[:start]+new+body[end:]


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    print(transform(a.base.read_text()),end='')
