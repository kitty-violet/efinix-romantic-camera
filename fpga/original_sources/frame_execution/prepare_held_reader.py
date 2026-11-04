"""Create a public RTL candidate that consumes the arena's held response.

The arena already keeps each physical RAM output stable until response ready.
Keep the reader's four metadata owners and its registered availability bit,
but remove the second 24x64 payload bank. This changes port retention, so it
requires concurrent compute/DMA tests; isolated reader equivalence is not enough.
"""
import argparse
from pathlib import Path
from run_units import ROOT, HERE
from frozen_sources import frozen_paths

def transform(text, even_only=False):
    replacements = [
        ('reg [15:0] expected[0:23];reg [63:0] payload[0:23];',
         'reg [15:0] expected[0:23];'),
        ('assign mem_rsp_ready=pending & {24{active}};',
         '// Release a physical response only when its complete packet is consumed.\n'
         ' // The arena/router retain its data and owner while ready is low.\n'
         ' assign mem_rsp_ready=pending & head_mask & {24{pop && active}};'),
        ('assign port_word[p]=payload[p];',
         'assign port_word[p]=mem_rsp_data[p*64+:64];'),
        ('if(mem_rsp_ready[p])begin pending[p]<=0;have[p]<=1;payload[p]<=mem_rsp_data[p*64+:64];end',
         'if(pending[p])have[p]<=1;\n'
         '     if(mem_rsp_ready[p])pending[p]<=0;'),
    ]
    for old,new in replacements:
        assert text.count(old)==1, old
        text=text.replace(old,new)
    if even_only:
        # Odd physical ports serve both writeback and residual reads. Holding
        # them until dispatcher consumption forms a dependency cycle through
        # the result FIFO. Retain their decoupling payloads; only even ports
        # can use the existing arena response without blocking writeback.
        text=text.replace('pending & head_mask & {24{pop && active}}',
                          'pending & (24\'haaaaaa | (head_mask & {24{pop}})) & {24{active}}')
        text=text.replace('assign port_word[p]=mem_rsp_data[p*64+:64];', '''if(p%2==0)begin:HELD_RAM
   assign port_word[p]=mem_rsp_data[p*64+:64];
  end else begin:DECOUPLED_ODD
   reg [63:0] payload;
   always @(posedge clk)if(active && mem_rsp_valid[p] && mem_rsp_ready[p])payload<=mem_rsp_data[p*64+:64];
   assign port_word[p]=payload;
  end''')
    return '// Candidate: '+('even ports use held RAM responses; odd ports retain writeback/skip decoupling.\n' if even_only else 'eliminate duplicate physical payload storage; retain RAM response ownership.\n')+text

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--even-only',action='store_true');a=p.parse_args()
    src=frozen_paths([HERE/'unified_column_reader.sv'],a.base)[0]
    dst=a.output.resolve();assert dst.is_relative_to(HERE/'variants') and not dst.exists()
    # Print the exact candidate; installation into the workspace uses apply_patch.
    print(transform(src.read_text(),a.even_only))
