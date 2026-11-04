"""Keep four lookahead slots; share two response payload muxes at acceptance."""
import argparse
from pathlib import Path


def transform(body):
    body='// Four outstanding row slots, two shared 64-bit RAM return selectors.\n// Returns to rows b and b+2 arbitrate fairly; blocked RAM responses retain\n// their original port lock until accepted. No read-ahead slot is deleted.\n'+body
    body=body.replace('assign mem_rsp_ready=port_locked & {24{active}};','// Response ready is selected by the shared return arbiters below.')
    start=body.index(' for(genvar b=0;b<4;b=b+1)begin:RETURN_WORD')
    end=body.index(' reg [1:0] choice[0:11];',start)
    body=body[:start]+''' reg [1:0] return_prefer;
 wire [1:0] return_valid;wire [1:0] return_owner[0:1];
 wire [63:0] shared_return_word[0:1];wire [23:0] return_mask[0:1];
 for(genvar s=0;s<2;s=s+1)begin:RETURN_ROUTE
  reg found;reg [3:0] selected_port;reg selected_high;
  always @*begin
   found=0;selected_port=0;selected_high=0;
   for(integer p=0;p<12;p=p+1)begin
    if(active && mem_rsp_valid[p*2+1] && port_locked[p*2+1] && port_owner[p][0]==1'(s) &&
       (!found || (port_owner[p][1]==return_prefer[s] && selected_high!=return_prefer[s])))begin
     found=1;selected_port=4'(p);selected_high=port_owner[p][1];
    end
   end
  end
  wire [23:0] selected_mask;
  reg [63:0] word_q;
  for(genvar p=0;p<12;p=p+1)begin:PORT
   assign selected_mask[p*2+:2]={found && selected_port==4'(p),1'b0};
  end
  always @*begin
   word_q=0;
   for(integer p=0;p<12;p=p+1)word_q=word_q | ({64{selected_mask[p*2+1]}} & mem_rsp_data[p*128+64+:64]);
  end
  assign return_mask[s]=selected_mask;assign return_valid[s]=found;
  assign return_owner[s]={selected_high,1'(s)};assign shared_return_word[s]=word_q;
  always @(posedge clk)begin
   if(reset || cancel)return_prefer[s]<=0;
   else if(found)return_prefer[s]<=!selected_high;
  end
 end
 assign mem_rsp_ready=return_mask[0]|return_mask[1];
 for(genvar b=0;b<4;b=b+1)begin:RETURN_WORD
  assign returned[b]=return_valid[b%2] && return_owner[b%2]==2'(b);
  assign return_word[b]=shared_return_word[b%2];
 end
'''+body[end:]
    return body


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    print(transform(a.base.read_text()),end='')
