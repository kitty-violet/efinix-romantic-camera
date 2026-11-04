"""Actual legacy native observer vs compact transport and explicit page oracle."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_serializer.py');code=source.read_text()
code=code.replace('from video_service_serializer_native import extension,P,B,net','from video_service_compact_native import extension,P,B,net')
code=code.replace("source=HERE/'variants/video_service/video_snapshot_serializer.sv'","source=HERE/'variants/video_service_compact/video_snapshot_serializer.sv'")
code=code.replace(" for k in {g[n][0] for n in selected}:"," helper=HERE/'variants/video_service/video_dualclock_sample.sv';shutil.copy2(helper,root/helper.name)\n for k in {g[n][0] for n in selected}|{'EFX_RAM10'}:")
code=code.replace("prefix='wire service_serial_tdo,service_snapshot_publish,service_snapshot_publish_allowed;\\n'","prefix='wire service_serial_tdo;\\n'")
code=code.replace("probes,output wire tdo,busy,valid", "probes,input wire[2:0]service_page_phase,input wire[1:0]service_page_part,input wire[182:0]service_observation_packet,output wire tdo,busy,valid")
anchor=" (root/'tb.sv').write_text(tb)"
extra=r'''
 tb=tb.replace('wire old_tdo,new_tdo', 'reg[2:0]service_page_phase=0;reg[1:0]service_page_part=0;reg[182:0]service_observation_packet=0;\n wire old_tdo,new_tdo')
 tb=tb.replace('probes,old_tdo','probes,service_page_phase,service_page_part,service_observation_packet,old_tdo').replace('probes,new_tdo','probes,service_page_phase,service_page_part,service_observation_packet,new_tdo')
 oracle=r"""
 reg[191:0]expected_source=0,expected_published=0,expected_scan=0;
 reg[2:0]source_index=0,published_index=0,scan_index=0;
 reg source_legacy=1,published_legacy=1,scan_legacy=1;
 always @(posedge system_clk)if(new_dut.\cpu/system_hardJtag_debug_logic_jtagBridge/observation/n5 )begin
  source_legacy<=service_page_phase<2;
  source_index<={service_page_part,service_page_phase[0]};
  if(service_page_phase<2)expected_source<={16'd0,8'd4,8'd0,(32'h4c4f0000|{service_page_part,service_page_phase[0]}),128'd0};
  else expected_source<={9'd0,service_observation_packet[182:176],8'd4,service_observation_packet[167:0]};
 end
 always @(posedge tck)if(new_dut.\cpu/system_hardJtag_debug_logic_jtagBridge/observation/n2000 )begin
  expected_published<=expected_source;published_index<=source_index;published_legacy<=source_legacy;
 end
 """
 # The integrated page sequencer is outside this transport fixture. A fixed
 # page_sequence=0 is an explicit cut; actual current-map packet checks cover it.
 tb=tb.replace('integer checked=0,frames=0,pauses=0;',oracle+'\n integer checked=0,frames=0,pauses=0;')
 tb=tb.replace('previous=0,record=0;','previous=0,record=0,old_record=0;')
 tb=tb.replace('capture=1;tick();capture=0;record=0;', 'capture=1;expected_scan=expected_published;scan_index=published_index;scan_legacy=published_legacy;tick();capture=0;record=0;old_record=0;')
 tb=tb.replace('if(compare_old && new_tdo!==old_tdo)','if(compare_old && i<64 && new_tdo!==old_tdo)')
 tb=tb.replace('record[i]=new_tdo;', 'record[i]=new_tdo;old_record[i]=old_tdo;')
 tb=tb.replace('shift=0;frames=frames+1;',r"""
   if(n>=1024 && record[32])begin
    if(record[1023:256]!==0)$fatal(1,"compact tail nonzero");
    if(record[255:192]!==expected_scan[191:128])$fatal(1,"compact header/meta mismatch %h %h",record[255:192],expected_scan[191:128]);
    if(scan_legacy)begin
     if(scan_index==7)begin
      if(record[191:64]!=={64'd0,old_record[1023:960]})$fatal(1,"legacy last word/padding");
     end else if(record[191:64]!==old_record[(64+scan_index*128)+:128])$fatal(1,"legacy word mismatch %d",scan_index);
    end else if(record[191:64]!==expected_scan[127:0])$fatal(1,"service payload changed during scan");
   end
   shift=0;frames=frames+1;""")
 seed=''
 for dut in ('old_dut','new_dut'):
  for i in range(1,8):seed+=f"  {dut}.\\cpu/system_hardJtag_debug_logic_jtagBridge/observation/sequence_number[{i}]~FF .Q=1'b{250>>i&1};\n"
 seed+='  #1;if(old_sequence!==250 || new_sequence!==250 || old_busy || new_busy)$fatal(1,"bad directed wrap state");\n'
 tb=tb.replace('  for(integer j=0;j<270;j=j+1)begin',seed+'  for(integer j=0;j<24;j=j+1)begin\n   service_page_phase=j%6;service_page_part=j/6;service_observation_packet={4\'d0,3\'b101,8\'d2,8\'d17,(32\'h56530000|((j/6)<<3)|(j%6)),32\'hdeadbeef,32\'h12345678,32\'h87654321,32\'hcafefeed};')
 tb=tb.replace('if(!new_busy || old_busy)$fatal(1,"partial scan did not defer publication");','if(new_busy || old_busy)$fatal(1,"compact snapshot not published");')
 tb=tb.replace('start(1);scan(1024,0,1);','start(1);scan(1024,1,0);')
 tb=tb.replace("8'd16","8'd20").replace('272 completions','26 completions').replace('completed=272 sequence_wrap=1','completed=26 directed_sequence_seed=250 sequence_wrap=1')
'''
assert code.count(anchor)==1;code=code.replace(anchor,extra+'\n'+anchor)
code=code.replace("addition=extra.replace(prefix.strip(),'') if mode=='new' else ''", "addition=extra.replace(prefix.strip(),'') if mode=='new' else ''\n  if mode=='new':addition='wire[7:0]service_page_sequence=0;\\n'+addition")
stop=code.index(" result=dict(status='RUNNING',hardware_access=False)")
code=code[:stop]+" print('COMPACT_NATIVE_FIXTURE_PREPARED',root)\nif __name__=='__main__':main()\n"
exec(compile(code,str(source)+':compact-fixture','exec'),globals())
