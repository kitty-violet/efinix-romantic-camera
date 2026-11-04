"""Actual original observer and same-edge enable-fold observer comparison."""
from pathlib import Path
source=Path(__file__).with_name('validate_video_service_serializer.py');code=source.read_text()
code=code.replace('from video_service_serializer_native import extension,P,B,net',
 'from video_service_serializer_native import P,B,net\nfrom video_service_enable_native import extension as enable_extension\ndef extension(D,removed,edits,supplied):return enable_extension(D,removed,edits)')
code=code.replace("prefix='wire service_serial_tdo,service_snapshot_publish,service_snapshot_publish_allowed;\\n' if mode=='new' else ''",
 "prefix='\\n'.join(line for line in extra.splitlines() if line.startswith('wire '))+'\\n' if mode=='new' else ''")
code=code.replace("addition=extra.replace(prefix.strip(),'') if mode=='new' else ''",
 "addition='\\n'.join(line for line in extra.splitlines() if not line.startswith('wire ')) if mode=='new' else ''")
code=code.replace("  if mode=='new':addition+='\\nassign '+render_token(net(B+'diagnostic_tdo'))+'=service_serial_tdo;'",'')
code=code.replace(" source=HERE/'variants/video_service/video_snapshot_serializer.sv';shutil.copy2(source,root/source.name)"," source=HERE/'video_service_enable_native.py'")
anchor=" (root/'tb.sv').write_text(tb)"
change='''
 tb=tb.replace('if(!new_busy || old_busy)$fatal(1,"partial scan did not defer publication");','if(new_busy || old_busy)$fatal(1,"original handshake did not publish");')
 tb=tb.replace('start(1);scan(1024,0,1);settled();start(1);scan(1024,1,0);','start(1);scan(1024,1,0);settled();start(1);scan(1024,1,0);')
 tb=tb.replace('completed=272 sequence_wrap=1','completed=22 directed_sequence_seed=250 sequence_wrap=1')
 seed=''
 for dut in ('old_dut','new_dut'):
  for i in range(1,8):seed+=f"  {dut}.\\\\cpu/system_hardJtag_debug_logic_jtagBridge/observation/sequence_number[{i}]~FF .Q=1'b{250>>i&1};\\n"
 seed+='  #1;if(old_sequence!==250 || new_sequence!==250 || old_busy || new_busy)$fatal(1,"bad directed wrap state");\\n'
 tb=tb.replace('  for(integer j=0;j<270;j=j+1)begin',seed+'  for(integer j=0;j<20;j=j+1)begin')
'''
assert code.count(anchor)==1;code=code.replace(anchor,change+'\n'+anchor)
stop=code.index(" result=dict(status='RUNNING',hardware_access=False)")
code=code[:stop]+" print('ENABLE_FOLD_NATIVE_FIXTURE_PREPARED',root)\nif __name__=='__main__':main()\n"
exec(compile(code,str(source)+':enable-fold-fixture','exec'),globals())
