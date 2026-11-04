"""Exercise the board's actual encrypted HDMI encoder in isolation.

Ideal 1080p horizontal timing, 16 diagnostic rows, no audio. This is a local
codec/model test, not a full 1080p frame, serializer/PHY or board measurement.
The sink decodes all three TMDS symbols, searching one fixed pipeline latency.
No protected source is replaced; includes the board's output polarity inversion.
"""
import argparse
import json
import re
import shutil
import traceback
from pathlib import Path
from run_units import ROOT, EF, MS, sha, run

TB = r'''`timescale 1ns/1ps
module tb_board_dvi;
reg clk=0;always #6.734007 clk=~clk;
reg reset=1,de=0,hs=0,vs=0;reg[23:0] rgb=0;
wire[9:0] d0,d1,d2,dc;
wire[9:0] tx0=~d0,tx1=~d1,tx2=~d2,txc=~dc;
dvi_encoder dut(.pixelclk(clk),.rst_p(reset),.i_bdata(rgb[7:0]),
 .i_gdata(rgb[15:8]),.i_rdata(rgb[23:16]),.i_hs(hs),.i_vs(vs),.i_de(de),
 .video_format(2'b00),.video_VIC(7'd34),.audio_L(24'd0),.audio_R(24'd0),
 .audio_valid(1'b0),.audio_N(20'd6144),.audio_CTS(20'd74250),
 .audio_sample_frequency(4'd0),.audio_word_length(4'b1011),
 .tmds_data0(d0),.tmds_data1(d1),.tmds_data2(d2),.tmds_clk(dc));
integer fd,n,x,y;initial begin
 fd=$fopen("symbols.txt","w");if(fd==0)$fatal(1,"output");
 for(n=0;n<40000;n=n+1)begin
  @(negedge clk);reset=(n<64);x=(n-64)%2200;y=(n-64)/2200;
  hs=(x<44);vs=(y<2);de=(n>=64&&y>=2&&y<18&&x>=280);
  rgb=24'h005aa5+(y-2)*1920+x-280;
  @(posedge clk);#1;
  $fwrite(fd,"%0d %0d %06h %03h %03h %03h %03h\n",n,de,rgb,tx0,tx1,tx2,txc);
 end
 $fclose(fd);$display("PASS DVI_STIMULUS_COMPLETE");$finish;
end
endmodule
'''


def decode(symbol):
    # Restore board polarity, undo optional inversion, then XOR/XNOR chain.
    s = (~int(symbol, 16)) & 1023
    q = (s & 255) ^ (255 if s & 512 else 0)
    d = q & 1
    for bit in range(1, 8):
        b = ((q >> bit) ^ (q >> (bit - 1))) & 1
        if not s & 256:
            b ^= 1
        d |= b << bit
    return d


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True, type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    assert root.is_relative_to(ROOT/'tmp') and not root.exists()
    root.mkdir(parents=True)
    try:
        base = ROOT/'tmp/frame-execution/connected-board-sharing1/rtl/dvi_tx'
        sources = sorted(f for f in base.iterdir() if f.is_file() and f.suffix in ('.v', '.sv', '.vh'))
        sources += [EF/'sim_models/verilog/efx_srl8.v']
        records = []
        for f in sources:
            dest = root/f.name
            shutil.copy2(f, dest)
            records.append(dict(source=str(f), snapshot=str(dest), sha256=sha(f)))
        (root/'tb_board_dvi.sv').write_text(TB)
        (root/'inputs.json').write_text(json.dumps(dict(inputs=records, tb_sha256=sha(root/'tb_board_dvi.sv'),
                                                       runner_sha256=sha(__file__)), indent=2))
        run([MS/'vlib.exe','work'],root,'vlib.log')
        run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','+incdir+.',
             *[r['snapshot'] for r in records if not r['snapshot'].endswith('.vh')],root/'tb_board_dvi.sv'],root,'compile.log')
        body, seconds = run([MS/'vsim.exe','-c','work.tb_board_dvi','-do',
                             'onerror {quit -code 1}; onbreak {quit -code 1}; run -all; quit -code 0'],root,'simulation.log')
        assert 'PASS DVI_STIMULUS_COMPLETE' in body
        rows = [line.split() for line in (root/'symbols.txt').read_text().splitlines()]
        assert len(rows)==40000
        valid = [i for i,r in enumerate(rows) if r[1]=='1']
        assert len(valid)==1920*16
        candidates=[]
        for delay in range(65):
            matches=True
            for i in valid:
                symbols=rows[i+delay][3:6]
                if any(re.search('[xz]',s,re.I) for s in symbols):matches=False;break
                got=decode(symbols[0])|(decode(symbols[1])<<8)|(decode(symbols[2])<<16)
                if got!=int(rows[i][2],16):matches=False;break
            if matches:candidates.append(delay)
        assert len(candidates)==1, f'No unique full-pixel latency: {candidates}'
        for i in valid:
            assert rows[i+candidates[0]][6].lower() not in ('xxx','zzz')
        for r in records:assert sha(r['snapshot'])==r['sha256']
        result=dict(status='DVI_16_ROWS_RGB_ROUNDTRIP_PASS',pixels=len(valid),latency_pixels=candidates[0],
                    actual_encrypted_encoder=True,board_polarity=True,seconds=seconds,
                    inputs_sha256=sha(root/'inputs.json'),symbols_sha256=sha(root/'symbols.txt'),
                    full_frame=False,full_board=False,serial_phy=False,board_release=False)
        (root/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    except Exception as e:
        (root/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise


if __name__=='__main__':main()
