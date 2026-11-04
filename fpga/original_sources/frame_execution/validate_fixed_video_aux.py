"""Four-state fixed AVI protocol gate, with the original aux as timing reference."""
import argparse,json,re,shutil,traceback
from pathlib import Path
from run_units import ROOT,HERE,EF,MS,sha,run
from validate_board_dvi import decode

TERC=(0x29c,0x263,0x2e4,0x2e2,0x171,0x11e,0x18e,0x13c,0x2cc,0x139,0x19c,0x2c6,0x28e,0x271,0x163,0x2c3)
TB=r'''`timescale 1ns/1ps
module tb_fixed_aux;
 reg clk=0;always #5 clk=~clk;
 reg rst_p=1,i_hs=0,i_vs=0;
 reg island_preamble=0,island_lead_gb=0,data_island_period=0,island_trail_gb=0,video_preamble=0,video_lead_gb=0;
 wire [1:0] video_format=0;wire [6:0] video_VIC=34;wire [23:0] audio_L=0,audio_R=0;wire audio_valid=0;
 wire [19:0] audio_N=6144,audio_CTS=74250;wire [3:0] audio_sample_frequency=0,audio_word_length=11;
 wire [9:0] a,b,c,ra,rb,rc;
 tx_aux_video_only dut(.ch0_dout(a),.ch1_dout(b),.ch2_dout(c),.*);
 tx_aux reference(.ch0_dout(ra),.ch1_dout(rb),.ch2_dout(rc),.*);
 integer fd,epoch,j,mode;
 initial begin
  fd=$fopen("symbols.txt","w");
  for(epoch=0;epoch<8;epoch=epoch+1)begin
   for(j=0;j<160;j=j+1)begin
    @(negedge clk);rst_p=j<3;i_hs=epoch%2;i_vs=(epoch/2)%2;
    mode=j<8?0:j<16?1:j<18?2:j<82?3:j<84?4:j<100?0:j<108?5:j<110?6:0;
    island_preamble=mode==1;island_lead_gb=mode==2;data_island_period=mode==3;
    island_trail_gb=mode==4;video_preamble=mode==5;video_lead_gb=mode==6;
    @(posedge clk);#1;
    $fwrite(fd,"%0d %0d %0d %0d %03h %03h %03h %03h %03h %03h\n",epoch,j,mode,{i_vs,i_hs},a,b,c,ra,rb,rc);
   end
  end
  $fclose(fd);$display("PASS FIXED_AUX_STIMULUS");$finish;
 end
endmodule
'''

def ecc(data):
    value=0
    for byte in data:
        for bit in range(8):
            value=(value>>1)^(0x83 if (value^(byte>>bit))&1 else 0)
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);a=p.parse_args()
    r=a.root.resolve();assert r.is_relative_to(ROOT/'tmp') and not r.exists();r.mkdir()
    base=ROOT/'tmp/frame-execution/connected-board-raw10-only2/rtl/dvi_tx'
    files=[f for f in base.iterdir() if f.suffix in ('.sv','.v','.vh')]
    files += [EF/'sim_models/verilog/efx_srl8.v',HERE/'variants/fixed_video_aux/tx_aux_video_only.sv']
    records=[]
    for f in files:
        dst=r/f.name;shutil.copy2(f,dst);records.append(dict(source=str(f),snapshot=str(dst),sha256=sha(f)))
    (r/'tb_fixed_aux.sv').write_text(TB)
    (r/'inputs.json').write_text(json.dumps(dict(inputs=records,tb_sha256=sha(r/'tb_fixed_aux.sv'),runner_sha256=sha(__file__)),indent=2))
    try:
        run([MS/'vlib.exe','work'],r,'vlib.log')
        run([MS/'vlog.exe','-sv','-timescale','1ns/1ps','+incdir+.',*[x['snapshot'] for x in records if not x['snapshot'].endswith('.vh')],r/'tb_fixed_aux.sv'],r,'compile.log')
        body,sec=run([MS/'vsim.exe','-c','work.tb_fixed_aux','-do','onerror {quit -code 1}; run -all; quit -code 0'],r,'simulation.log',timeout=120)
        assert 'PASS FIXED_AUX_STIMULUS' in body and 'Errors: 0, Warnings: 0' in body
        rows=[s.split() for s in (r/'symbols.txt').read_text().splitlines()];assert len(rows)==1280
        # Non-packet channels must exactly match the unchanged vendor timing.
        equal=0
        for index,row in enumerate(rows):
            epoch,j,mode,control=map(int,row[:4])
            if 4<=j<158 and mode!=3:
                actual=rows[index+2]
                assert actual[4:7]==actual[7:10],('vendor-control-timing',row,actual)
                equal+=3
        packets=[]
        for epoch in range(8):
            active=[(row,rows[i+2]) for i,row in enumerate(rows) if int(row[0])==epoch and int(row[2])==3]
            for start in (0,32):
                hb=0;sb=[0]*4
                for k,(row,actual) in enumerate(active[start:start+32]):
                    a,b,c=(TERC.index(int(s,16)) for s in actual[4:7])
                    assert a&3==int(row[3]) and a>>3==int(k!=0),(epoch,k,row)
                    hb|=((a>>2)&1)<<k
                    for lane in range(4):sb[lane]|=((b>>lane)&1)<<(2*k);sb[lane]|=((c>>lane)&1)<<(2*k+1)
                h=hb.to_bytes(4,'little');sub=[v.to_bytes(8,'little') for v in sb]
                assert h[:3]==bytes([0x82,2,13]) and h[3]==ecc(h[:3])
                assert all(v[7]==ecc(v[:7]) for v in sub)
                payload=b''.join(v[:7] for v in sub)
                assert (sum(h[:3])+sum(payload[:14]))%256==0
                assert payload[1:5]==bytes([0,8,0,34]) and not any(payload[5:])
                packets.append(dict(header=h.hex(),payload=payload.hex()))
        for x in records:assert sha(x['source'])==sha(x['snapshot'])==x['sha256']
        result=dict(status='FIXED_VIDEO_AUX_PROTOCOL_PASS',vendor_control_symbols=equal,avi_packets=len(packets),packets=packets[:1],seconds=sec,inputs_sha256=sha(r/'inputs.json'),log_sha256=sha(r/'simulation.log'),symbols_sha256=sha(r/'symbols.txt'),whole_video=False,board_release=False)
        (r/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
    except Exception as e:
        (r/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2));raise
if __name__=='__main__':main()
