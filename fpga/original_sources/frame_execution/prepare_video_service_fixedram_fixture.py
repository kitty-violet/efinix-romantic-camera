"""Same actual primitive reference, banked transport DUT, bounded fixture."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_serializer.py')
code=source.read_text()
code=code.replace('from video_service_serializer_native import extension,P,B,net',
 'from video_service_fixedram_native import extension,P,B,net')
code=code.replace("source=HERE/'variants/video_service/video_snapshot_serializer.sv'",
 "source=HERE/'variants/video_service_ram_fixed/video_snapshot_serializer.sv'")
code=code.replace("prefix='wire service_serial_tdo,service_snapshot_publish,service_snapshot_publish_allowed;\\n'",
 "prefix='wire service_serial_tdo,service_snapshot_capture,service_snapshot_store_done;\\n'")
code=code.replace(" for k in {g[n][0] for n in selected}:"," for k in {g[n][0] for n in selected}|{'EFX_RAM10'}:")
anchor=" (root/'tb.sv').write_text(tb)"
changes=r'''
 # Source capture -> fixed words -> RAM is intentionally 60 cycles slower.
 # Header/payload are still from the same published epoch, never mixed banks.
 tb=tb.replace('if(compare_old && new_tdo!==old_tdo)',
  'if(compare_old && (i<64 || record[32]) && new_tdo!==old_tdo)')
 tb=tb.replace('if(!new_busy || old_busy)$fatal(1,"partial scan did not defer publication");',
  'if(new_busy || old_busy)$fatal(1,"RAM bank not published after partial scan");')
 tb=tb.replace('start(1);scan(1024,0,1);settled();start(1);scan(1024,1,0);',
  'start(1);scan(1024,1,0);settled();start(1);scan(1024,1,0);')
 tb=tb.replace('completed=272 sequence_wrap=1','completed=22 directed_sequence_seed=250 sequence_wrap=1')
 seed=''
 for dut in ('old_dut','new_dut'):
  for i in range(1,8):seed+=f"  {dut}.\\cpu/system_hardJtag_debug_logic_jtagBridge/observation/sequence_number[{i}]~FF .Q=1'b{250>>i&1};\n"
 seed+='  #1;if(old_sequence!==250 || new_sequence!==250 || old_busy || new_busy)$fatal(1,"bad directed wrap state");\n'
 tb=tb.replace('  for(integer j=0;j<270;j=j+1)begin',seed+'  for(integer j=0;j<20;j=j+1)begin')
'''
assert code.count(anchor)==1;code=code.replace(anchor,changes+'\n'+anchor)
stop=code.index(" result=dict(status='RUNNING',hardware_access=False)")
code=code[:stop]+" print('FIXEDRAM_NATIVE_FIXTURE_PREPARED',root)\nif __name__=='__main__':main()\n"
exec(compile(code,str(source)+':fixedram-fixture','exec'),globals())
