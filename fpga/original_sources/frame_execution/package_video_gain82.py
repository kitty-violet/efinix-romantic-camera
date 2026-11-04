"""Reuse released geometry and full ROM audit; change gain and its expectation."""
from pathlib import Path
import re
source=Path(__file__).with_name('package_video_exposure76.py')
code=source.read_text(encoding='utf8')
edits={
 'video-exposure76-candidate1':'video-gain82-candidate1',
 'video-exposure76-bram1':'video-gain82-bram1',
 'video-exposure76-pin1':'video-gain82-pin2',
 'board-full-150-video-exposure76-20261018':'board-full-150-video-gain82-20261018',
 'EXPOSURE76_ROM_EXACT_DELTA_PASS_NOT_RELEASED':'GAIN82_ROM_EXACT_DELTA_PASS_NOT_RELEASED',
 "assert cr['changed_addresses']==[116] and cr['changes']=={'116':dict(register=0x3e01,old=0x46,new=0x54)}":
 "assert cr['changed_addresses']==[116,162,266]\n assert cr['changes']=={'116':dict(register=0x3e01,old=0x46,new=0x54),'162':dict(register=0x3e09,old=0x34,new=0x2a),'266':dict(register=0,old=0x34,new=0x2a,kind='readback_expected_byte')}",
 "assert cr['gain_before']==cr['gain_after']==2.503 and cr['verification_bank_unchanged']":
 "assert cr['gain_before']==2.503 and cr['gain_after']==2.021 and not cr['verification_bank_unchanged']\n assert cr['verification_changed_address']==266 and cr['live76_exposure_unchanged']",
 'OFFICIAL_BRAM_EXPOSURE76_OVERLAY_PASS_NOT_RELEASED':'OFFICIAL_BRAM_GAIN82_OVERLAY_PASS_NOT_RELEASED',
 "br['changed_addresses']==[116]":"br['changed_addresses']==[116,162,266]",
 'update_video_exposure76_bram.py':'update_video_gain82_bram.py',
 "target=br['target'];assert before[target][256*20:]==after[target][256*20:]":
 "target=br['target']\n assert {i//20 for i,(a,b) in enumerate(zip(before[target],after[target])) if a!=b}=={116,162,266}\n assert int(after[target][266*20:267*20][::-1],2)==0x2a<<1",
 'prepare_video_exposure76.py':'prepare_video_gain82.py',
 "data=table(3.0,[1.1431,.73435,1.4135],8)":"import video_input_tone80 as tone80\n data=tone80.table()\n bindfile(Path(tone80.__file__))\n bindfile(source)",
 "3ed37a9cc745220b7ea90dc39dcc7de55e22761423cc2b1ae8f8108f010f805b":"b8970f10fde770c9a859d457df68446e329e6d1e7222a85bbe8c430f82d4b96d",
 "plan=dict(gamma=3.0,black=8,gains=[1.1431,.73435,1.4135],swap_rb=True,\n  sha256=hashlib.sha256(data).hexdigest(),crc32=f'{zlib.crc32(data):08x}',hardware_applied=False)":
 "plan=tone80.describe();plan['hardware_applied']=False;plan['sensor_unchanged']=dict(exposure_half_lines=1344,analog_gain=2.021);plan['bitstream_version']=82\n plan['table_formula_unchanged_from_80']=True;plan['new_gain_visual_calibration_pending']=True",
 "sp=dict(analog_gain=2.503,exposure_half_lines=1344,gain_registers=[128,52],exposure_registers=[0,84,0])":
 "sp=dict(analog_gain=2.021,exposure_half_lines=1344,gain_registers=[128,42],exposure_registers=[0,84,0])",
 'frozen73 physical implementation plus official exposure-only ROM overlay76':'frozen73 physical implementation plus official gain82 write-and-readback ROM overlay; exposure matches live76',
 'supervised76 startup and actual brightness comparison':'supervised82 lower-gain startup and actual headroom/color comparison',
 'exposure76_program':'gain82_program',
}
for old,new in edits.items():
 assert old in code,old
 code=code.replace(old,new)
start=code.index(" (OUT/'README.md').write_text(f'''# 76:") if " (OUT/'README.md').write_text(f'''# 76:" in code else code.index(" (OUT/'README.md').write_text(f'''# 76：")
end=code.index("''',encoding='utf8')",start)+len("''',encoding='utf8')")
readme=""" (OUT/'README.md').write_text(f'''# 82：相对76只降低相机模拟增益

SC431HAI模拟增益从2.503倍降至2.021倍：3e08维持80，3e09由34改2a。曝光保持1344半行（00/54/00）。配置写值和ROM校验区地址266的读回期望同时更新，旧34读回必须拒绝。参考图电脑屏幕亮度不作补偿。

有效设计为73冻结实现加source_overlay.json；相对73的ROM地址116/162/266变化，相对当前76只改变增益命令及期望两项。官方工具核对147逻辑存储，其余146不变；原图恢复控制和新位流重复生成一致。实际I2C正常、旧增益拒绝、错误读回、NACK和复位恢复五场景通过。

原map/place/route/VDB/SDC保持，WNS {pm['wns_ns']:+.3f}ns、WHS {pm['whs_ns']:+.3f}ns来自同一冻结几何，没有重跑或虚改时序。150MHz，XLR60462/60800、RAM245/256、DSP160/160。只更新参数存储，无新逻辑。

比较时上传原曲线80（CRC971d9d30），曝光、RGB曲线和几何都不变，单独测试降增益。该表不因此获得新增益下的视觉验收。默认位流的ISP表仍需显式上传并完整768字节读回。不要上传本轮三个被否决的参考图拟合表。

需要用户在旁监督，仅临时JTAG、不写Flash。下载会清DDR；启动/DDR/摄像头核对后重新上传A20KB并完整读回，prepare INPUT_PREVIEW，再传80表并完整读回，最后恢复实时预览。未复采前不能宣称消除饱和/校色完成。CNN、实际15fps、丢帧及其余系统职责仍待验证。
''',encoding='utf8')"""
code=code[:start]+readme+code[end:]
exec(compile(code,str(source)+':gain82','exec'),globals())
