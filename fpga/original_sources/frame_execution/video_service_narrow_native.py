"""All observation fields in 32-bit words; no changes to functional paths.

Version5 has32 legacy words and16 words per service page. Every32-bit
counter remains atomic. A complete legacy sweep keeps one C/E/D selection.
"""
from video_service_display_native import extension as display_original
from video_service_compact_epoch_native import extension as compact_original
from video_service_serializer_native import net

def display(D,removed,edits):
 code,r=display_original(D,removed,edits)
 changes=[
  ('wire [127:0] service_pixel_cache;wire [1:0] service_pixel_part;', 'wire [31:0] service_pixel_cache;wire [3:0] service_pixel_part;'),
  ('reg [1:0] service_page_part=0;', 'reg [3:0] service_page_part=0; reg [1:0] service_legacy_epoch=0;'),
  ('#(.WIDTH(128),.SOURCE_WIDTH(512))','#(.WIDTH(32),.SOURCE_WIDTH(512))'),
  ("if(service_page_phase==5)begin service_page_phase<=0;service_page_part<=service_page_part+1'b1;end", "if(service_page_phase==5)begin service_page_phase<=0;service_page_part<=service_page_part+1'b1; if(service_page_part==15)service_legacy_epoch<=service_legacy_epoch+1'b1;end"),
  ('wire [127:0] service_selected_page;','wire [31:0] service_selected_page;'),
  ('service_page_part*128+:128','service_page_part*32+:32'),
  ('wire [1:0] service_selected_part;','wire [3:0] service_selected_part;'),
  ('wire [182:0] service_observation_packet;','wire [86:0] service_observation_packet;'),
  ("8'd2,service_page_sequence,27'h2b29800","8'd5,service_page_sequence,25'haca600")]
 for old,new in changes:
  assert old in code,old;code=code.replace(old,new)
 # These obsolete held-payload assignments are unused after the compact
 # serializer cut. Keep their full183-bit declarations without OOB access.
 import re
 code=re.sub(r'service_observation_packet\[(\d+)\]',lambda m:m[0] if int(m[1])<87 else "1'b0",code)
 r.update(schema_version=5,chunk_bits=32,chunks_per_page=16,legacy_chunks=32,
          whole_page_atomic=False,each_word_atomic=True,legacy_epoch_whole_sweep=True)
 return code,r

def serializer(D,removed,edits,supplied):
 code,r=compact_original(D,removed,edits,supplied)
 changes=[
  ('wire [2:0] service_legacy_part;','wire [4:0] service_legacy_part;'),
  ('wire [191:0] service_compact_packet;','wire [95:0] service_compact_packet;'),
  ("16'd0,8'd4,service_page_sequence,29'h989e000", "16'd0,8'd5,service_page_sequence,27'h2627800"),
  ('service_legacy_part*128+:128','service_legacy_part*32+:32'),
  ("{9'd0,service_observation_packet[182:176],8'd4,service_observation_packet[167:0]}","{9'd0,service_observation_packet[86:80],8'd5,service_observation_packet[71:0]}")]
 for old,new in changes:
  assert code.count(old)==1,(old,code.count(old));code=code.replace(old,new)
 # Reuse the previously proved observation-only cut; change the epoch
 # source to one increment per96 requests, stable over all32 legacy words.
 counts={3:0,4:0}
 for pins in edits.values():
  for replacements in pins.values():
   for old,new in list(replacements.items()):
    if new in (net('service_page_sequence',3),net('service_page_sequence',4)):
     bit=new[2];replacements[old]=net('service_legacy_epoch',bit-3);counts[bit]+=1
 assert counts=={3:118,4:1},counts
 r.update(source_word_bits=96,published_word_bits=96,scan_register_bits=160,
  tail_bits_zero=864,legacy_chunks=32,legacy_chunk_bits=32,service_chunks=16,
  service_chunk_bits=32,schema_version=5,legacy_camera_selector='service_legacy_epoch[0]',
  legacy_progress_qualifier='service_legacy_epoch[1]',legacy_camera_chunks=list(range(24,30)),
  legacy_camera_cycle='C/E/C/D per96-request sweep; stable for all32 legacy words')
 return code,r
