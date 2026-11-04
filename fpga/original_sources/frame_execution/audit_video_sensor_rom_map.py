"""Adapt the complete retained-primitive audit to eight muxes and a ROM bank."""
from pathlib import Path
source=Path(__file__).with_name('audit_video_sensor72.py')
code=source.read_text(encoding='utf8')
def once(old,new):
 global code
 assert code.count(old)==1,old
 code=code.replace(old,new)
once("proof=json.loads((native/'result.json').read_text())","proof=json.loads((native/'inputs.json').read_text())")
once("proof['candidate_native_sha256']","proof['source_sha256']")
begin=code.index(' for patch in proof[\'native_init_patches\']:')
end=code.index(' assert expected.keys()==g.keys()',begin)
code=code[:begin]+''' from prepare_video_service_color_exposure import read_rom
 _,blocks=read_rom((native/'native_source.v').read_text(encoding='utf8'))
 blob=next(x[4] for x in blocks if x[3]==20)
 target='camera_checked/config_rom/ROM__D$2'
 for i in range(40):ep[target]['INIT_'+format(i,'X')]=(256,(blob>>(256*i))&((1<<256)-1))
 assert expected[target][1]['RADDR'][0]==('literal',"1'b0",None)
 expected[target][1]['RADDR'][0]=('net','n80869_2',None)
 for bit in range(8):
  n=f'camera_config_diagnostic[{bit+16}]~FF';old_d=('net','camera_checked/n229',bit)
  assert expected[n][1]['D']==[old_d]
  net=('net',f'sensor_rom_expected_{bit}',None);expected[n][1]['D']=[net]
  name=f'sensor_rom_expected_mux{bit}'
  expected[name]=('EFX_LUT4',{'I0':[old_d],'I1':[('net','camera_checked/rom_data',bit+1)],'I2':[('net','n80869_2',None)],'O':[net]})
  ep[name]={'LUTMASK':(16,0xcaca)}
''' +code[end:]
code=code.replace('sensor72-map-parse.pkl','sensor73-map-parse.pkl').replace('sensor72-map-audit.json','sensor73-map-audit.json')
code=code.replace('SENSOR72_ALL_RETAINED_MAP_PASS','SENSOR73_ROM_VERIFY_ALL_RETAINED_MAP_PASS')
once('retained=len(old),new_luts=5,expected_ff_d_changes=4,','retained=len(old),new_luts=8,expected_ff_d_changes=8,ram_address_pin_changes=1,')
once("native/'result.json',board/'inputs.json'","source,native/'inputs.json',board/'inputs.json'")
begin=code.index(" rom=(native/'candidate_i2c_master_reg_rom.v').read_text()")
end=code.index(" tb=(HERE/'tb_camera_checked_config.sv').read_text()",begin)
code=code[:begin]+" shutil.copy2(native/'expected.hex',prepared/'expected.hex')\n"+code[end:]
code=code.replace('ACTUAL_SENSOR72_CONE_PREPARED','ACTUAL_SENSOR73_CONE_PREPARED')
exec(compile(code,str(source)+':rom-backed-eight-muxes','exec'),globals())
