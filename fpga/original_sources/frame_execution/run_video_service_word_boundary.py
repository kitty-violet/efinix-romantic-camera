"""Exact emitted schema3 wiring: all16 words, status signature and controls."""
from pathlib import Path
original=Path(__file__).with_name('run_video_service_display_boundary.py')
code=original.read_text()
changes=[
 ('from video_service_display_native import extension,P,O,split_net_declarations','from video_service_word_native import extension,P,O,split_net_declarations'),
 ('reg[182:0] packet;reg[2:0] phase;reg[1:0] part;reg[127:0] expected;','reg[86:0] packet;reg[2:0] phase;reg[3:0] part;reg[31:0] expected;'),
 ('packet[159:128]','packet[63:32]'),('packet[175:168]!==2','packet[79:72]!==3'),('packet[182:179]','packet[86:83]'),
 ('part*128+:128','part*32+:32'),('packet[127:0]','packet[31:0]'),
 ('if(phase==3 && part==1 && packet[127:96]!==55)','if(phase==3 && part==7 && packet[31:0]!==55)'),
 ('if(phase==2 && packet[176] && part==3 && (packet[127:96]!==32\'h56494431 || packet[31:0]!==1))',
  'if(phase==2 && packet[80] && ((part==15 && packet[31:0]!==32\'h56494431) || (part==12 && packet[31:0]!==1)))'),
 ('repeat(48)snapshot();','repeat(192)snapshot();'),
 ('$fatal(1,"process committed counter layout");', '$fatal(1,"process committed counter layout");\n   if(phase==3 && part==12 && (packet[31:16]!==16\'h5352 || !packet[0]))$fatal(1,"atomic prepared signature");'),
 ('#100000;','#200000;')]
for old,new in changes:
 assert old in code,old
 code=code.replace(old,new)
exec(compile(code,str(original)+':schema3-word','exec'),globals())
