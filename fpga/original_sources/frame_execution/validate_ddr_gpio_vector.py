"""Four-state DDR pad-vector equivalence before any real-controller use."""
import argparse,json,os,re,shutil
from pathlib import Path
from probe_board_dpi_cost import SIM,run
from run_units import ROOT,sha


def vector_periphery(source):
    first=source.index(' for(genvar k=0;k<24;k=k+1)begin:g_cmd')
    end=source.index(' ddr3 #(',first)
    replacement='''
 ddr_gpio_vector_sim #(.WIDTH(24),.TRI_STATE(0),.DDR_OUT(0)) cmd_vec(
  .outclk(sdram_clk),.inclk(1'b0),.high_data(commands),.low_data(24'b0),.oe(24'hffffff),.pad(pins),.high_sample(),.low_sample());
 ddr_gpio_vector_sim #(.WIDTH(16)) dq_vec(
  .outclk(tx_cal_clk_90edge),.inclk(rx_cal_clk),.high_data(ddr_dq_out_hi),.low_data(ddr_dq_out_lo),.oe(ddr_dq_oe),
  .pad(dq),.high_sample(ddr_dq_in_hi),.low_sample(ddr_dq_in_lo));
 wire [3:0] dqs_hi,dqs_lo;
 ddr_gpio_vector_sim #(.WIDTH(4)) dqs_vec(
  .outclk(sdram_clk),.inclk(rx_cal_clk),.high_data({~ddr_dqs_out_hi,ddr_dqs_out_hi}),
  .low_data({~ddr_dqs_out_lo,ddr_dqs_out_lo}),.oe({ddr_dqs_oe,ddr_dqs_oe}),
  .pad({dqs_n,dqs}),.high_sample(dqs_hi),.low_sample(dqs_lo));
 assign ddr_dqs_in_hi=dqs_hi[1:0];assign ddr_dqs_in_lo=dqs_lo[1:0];
 ddr_gpio_vector_sim #(.WIDTH(2),.TRI_STATE(0)) dm_vec(
  .outclk(tx_cal_clk_90edge),.inclk(1'b0),.high_data(ddr_dm_hi),.low_data(ddr_dm_lo),.oe(2'b11),
  .pad(dm),.high_sample(),.low_sample());
'''
    return source[:first]+replacement+source[end:]


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--board',type=Path,required=True);a=p.parse_args();root=a.root.resolve();base=a.board.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists();root.mkdir()
    gpio=base/'EFX_GPIO_V3.v';vector=Path(__file__).parent/'board_integration/ddr_gpio_vector_sim.sv'
    for f in (gpio,vector):shutil.copy2(f,root/f.name)
    (root/'ddr516_periphery_vector.sv').write_text(vector_periphery((base/'ddr516_periphery_model.sv').read_text()))
    tb=r'''
module tb;timeunit 1ns;timeprecision 1ps;
reg outclk=0,inclk=0;always #1.25 outclk=~outclk;always #1.251 inclk=~inclk;
reg [15:0] hi,lo,oe,external_value,external_oe;
wire[15:0] pad_a,pad_b,high_a,high_b,low_a,low_b;
wire[15:0] reg_a,reg_b,ddr_a,ddr_b;
for(genvar k=0;k<16;k++)begin:g
 assign pad_a[k]=external_oe[k] ? external_value[k]:1'bz;
 assign pad_b[k]=external_oe[k] ? external_value[k]:1'bz;
 wire [3:0] sampled;
 EFX_GPIO_V3 #(.MODE("INOUT"),.OUT_REG("DDIO_RESYNC"),.IN_REG("DDIO_RESYNC"),.OE_REG("REG"),.IO_STANDARD("1.5_V_SSTL"))
  dut(.O({2'bzz,lo[k],hi[k]}),.OE(oe[k]),.OUTCLK(outclk),.INCLK(inclk),.IO(pad_a[k]),.I(sampled));
 assign high_a[k]=sampled[0];assign low_a[k]=sampled[1];
 EFX_GPIO_V3 #(.MODE("OUTPUT"),.OUT_REG("REG"),.IO_STANDARD("1.5_V_SSTL"))
  cmd(.O({3'bzzz,hi[k]}),.OUTCLK(outclk),.IO(reg_a[k]));
 EFX_GPIO_V3 #(.MODE("OUTPUT"),.OUT_REG("DDIO_RESYNC"),.IO_STANDARD("1.5_V_SSTL"))
  mask(.O({2'bzz,lo[k],hi[k]}),.OUTCLK(outclk),.IO(ddr_a[k]));
end
ddr_gpio_vector_sim vector(.outclk(outclk),.inclk(inclk),.high_data(hi),.low_data(lo),.oe(oe),.pad(pad_b),.high_sample(high_b),.low_sample(low_b));
ddr_gpio_vector_sim #(.TRI_STATE(0),.DDR_OUT(0)) cmd_vec(.outclk(outclk),.inclk(1'b0),.high_data(hi),.low_data(16'b0),.oe(16'hffff),.pad(reg_b),.high_sample(),.low_sample());
ddr_gpio_vector_sim #(.TRI_STATE(0)) mask_vec(.outclk(outclk),.inclk(1'b0),.high_data(hi),.low_data(lo),.oe(16'hffff),.pad(ddr_b),.high_sample(),.low_sample());
integer comparisons=0;integer x_cases=0,z_cases=0;reg [31:0] lfsr=32'h2485dc71;
always @(outclk or inclk or hi or lo or oe or external_oe or external_value)begin
 #0.001;
 if({pad_a,high_a,low_a,reg_a,ddr_a} !== {pad_b,high_b,low_b,reg_b,ddr_b})
  $fatal(1,"GPIO_VECTOR_MISMATCH t=%t pads=%h/%h hi=%h/%h low=%h/%h cmd=%h/%h ddr=%h/%h",$time,pad_a,pad_b,high_a,high_b,low_a,low_b,reg_a,reg_b,ddr_a,ddr_b);
 comparisons++;
end
initial begin
 hi=0;lo=0;oe=0;external_value=0;external_oe=0;
 for(integer i=0;i<5000;i++)begin
  #0.731;lfsr={lfsr[30:0],lfsr[31]^lfsr[21]^lfsr[1]^lfsr[0]};
  hi=lfsr[15:0];lo=lfsr[31:16];oe=16'(lfsr>>7);external_oe=16'(lfsr>>5);external_value=~lfsr[15:0];
  if(i%11==0)begin hi[i%16]=1'bx;oe[(i+3)%16]=1'bx;x_cases++;end
  if(i%13==0)begin lo[i%16]=1'bz;external_value[(i+2)%16]=1'bz;z_cases++;end
 end
 #10;if(comparisons<9000 || x_cases<100 || z_cases<100)$fatal(1,"coverage missing");
 $display("PASS GPIO_VECTOR_FOUR_STATE comparisons=%0d X=%0d Z=%0d",comparisons,x_cases,z_cases);$finish;
end
endmodule
'''
    (root/'tb.sv').write_text(tb)
    (root/'run.do').write_text('onerror {quit -code 1}\nonbreak {quit -code 1}\nrun -all\nquit -code 0\n')
    env=os.environ.copy();env['SALT_LICENSE_SERVER']='E:/quatra/LR-195771_License.dat'
    run([SIM/'vlib.exe','work'],root,'vlib',env,20)
    run([SIM/'vlog.exe','-sv',root/gpio.name,root/vector.name,root/'tb.sv'],root,'compile',env,30)
    wall=run([SIM/'vsim.exe','-batch','work.tb','-do','run.do'],root,'simulation',env,90)
    log=(root/'simulation.log').read_text(errors='replace');m=re.search(r'PASS GPIO_VECTOR_FOUR_STATE.*',log)
    assert m and not re.search(r'\*\* (Fatal|Error)',log)
    result=dict(status='DDR_GPIO_VECTOR_FOUR_STATE_EQUIVALENCE_PASS',wall_seconds=wall,coverage=m[0],
      source_sha256=sha(gpio),vector_sha256=sha(vector),log_sha256=sha(root/'simulation.log'),hardware_changed=False,
      real_ddr_pass=False,whole_board_speed=False)
    (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
