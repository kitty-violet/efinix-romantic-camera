"""Transparent, spatially grouped control drivers. No added clock or cycle."""
import argparse,json
from pathlib import Path
from run_units import ROOT,sha

BASE=ROOT/'tmp/frame-execution/connected-board-control-state1/assembly/fixed_window_dot8x32.sv'

def localize(text):
    assert text==BASE.read_text(),'Transform only the frozen production arithmetic'
    anchor=' assign in_ready=!reset && !drain && room;'
    addition=''' // Local fanout branches follow physical row/channel consumers. EFX_LUT4
 // identity drivers plus syn_keep retain both the driver and its loads.
 // This is combinational distribution; advance/valid/drain retain their cycle.
 (* syn_keep="true" *) wire [5:0] history_advance;
 (* syn_keep="true" *) wire [3:0] early_depthwise,tail_advance;
 (* syn_keep="true" *) wire weight_advance;
 (* syn_keep="true" *) wire [15:0] pair_advance;
 (* syn_keep="true" *) wire [31:0] lane_depthwise;
 for(genvar local_row=0;local_row<6;local_row=local_row+1)begin:HISTORY_CONTROL
  EFX_LUT4 #(.LUTMASK(16'haaaa)) enable_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(history_advance[local_row]));
 end
 for(genvar local_row=0;local_row<4;local_row=local_row+1)begin:ROW_CONTROL
  EFX_LUT4 #(.LUTMASK(16'haaaa)) mode_driver(.I0(depthwise),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(early_depthwise[local_row]));
  EFX_LUT4 #(.LUTMASK(16'haaaa)) tail_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(tail_advance[local_row]));
  for(genvar local_pair=0;local_pair<4;local_pair=local_pair+1)begin:PAIR
   EFX_LUT4 #(.LUTMASK(16'haaaa)) enable_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(pair_advance[local_row*4+local_pair]));
  end
  for(genvar local_channel=0;local_channel<8;local_channel=local_channel+1)begin:CHANNEL
   EFX_LUT4 #(.LUTMASK(16'haaaa)) mode_driver(.I0(depthwise),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(lane_depthwise[local_row*8+local_channel]));
  end
 end
 EFX_LUT4 #(.LUTMASK(16'haaaa)) weight_enable_driver(.I0(advance),.I1(1'b0),.I2(1'b0),.I3(1'b0),.O(weight_advance));
'''
    assert text.count(anchor)==1;text=text.replace(anchor,addition+anchor)
    # Replace only payload users. Pipeline validity and handshakes keep the
    # original advance and depthwise, so no interface or warmup changes.
    changes={
      'SHARED_COLUMN_ACTIVATION!=0 && !depthwise ?':'SHARED_COLUMN_ACTIVATION!=0 && !early_depthwise[row] ?',
      '.CLK(clk),.CE(advance),.Q(q),.Q7());\n    assign delay':'.CLK(clk),.CE(history_advance[row]),.Q(q),.Q7());\n    assign delay',
      '.CLK(clk),.CE(advance),.Q(q),.Q7());\n   assign tail_dense':'.CLK(clk),.CE(tail_advance[r]),.Q(q),.Q7());\n   assign tail_dense',
      '.CLK(clk),.CE(advance),.Q(q),.Q7());\n   assign tail_weight':'.CLK(clk),.CE(weight_advance),.Q(q),.Q7());\n   assign tail_weight',
      '.CLK(clk),.CE(advance),.Q(q),.Q7());\n     assign a':'.CLK(clk),.CE(history_advance[r]),.Q(q),.Q7());\n     assign a',
      '.CLK(clk),.CE(advance),.Q(q),.Q7());\n     assign b':'.CLK(clk),.CE(weight_advance),.Q(q),.Q7());\n     assign b',
      '.A(depthwise ? dw[r]':'.A(lane_depthwise[r*8+c] ? dw[r]',
      '.CLK(clk),.CE(advance),.RST(1\'b0)':' .CLK(clk),.CE(pair_advance[r*4+c/2]),.RST(1\'b0)',
      'tail_a=depthwise ?':'tail_a=lane_depthwise[r*8+c] ?',
      'if(advance)product<=':'if(pair_advance[r*4+c/2])product<='
    }
    for before,after in changes.items():
        assert before in text,before;text=text.replace(before,after)
    return text

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert a.output.resolve().is_relative_to(ROOT/'tmp') and not a.output.exists()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(localize(BASE.read_text()))
    print(json.dumps(dict(source=str(BASE),source_sha256=sha(BASE),candidate_sha256=sha(a.output),added_cycles=0)))
