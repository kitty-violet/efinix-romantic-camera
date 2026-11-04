"""All diagnostic fields retained, transferred as32-bit atomic words.

Version3: 16 words per512-bit page. Legacy phases0/1 are still full OFP5
snapshots. A word contains its own phase/part/version tag; no full-page atomic
claim. Prepared status gets a signature within its single atomic word.
"""
from pathlib import Path
original=Path(__file__).with_name('video_service_display_native.py');code=original.read_text()
changes=[
 ('positions=positions[:183]','positions=positions[:87]'),
 ('wire [127:0] service_pixel_cache;wire [1:0] service_pixel_part;','wire [31:0] service_pixel_cache;wire [3:0] service_pixel_part;'),
 ('reg [1:0] service_page_part=0;','reg [3:0] service_page_part=0;'),
 ('#(.WIDTH(128),.SOURCE_WIDTH(512))','#(.WIDTH(32),.SOURCE_WIDTH(512))'),
 ("{19'd0,service_video_fault_code","{16'h5352,3'd0,service_video_fault_code"),
 ('wire [127:0] service_selected_page;','wire [31:0] service_selected_page;'),
 ('service_page_part*128+:128','service_page_part*32+:32'),
 ('wire [1:0] service_selected_part;','wire [3:0] service_selected_part;'),
 ('wire [182:0] service_observation_packet;','wire [86:0] service_observation_packet;'),
 ("8'd2,service_page_sequence,27'h2b29800","8'd3,service_page_sequence,25'haca600"),
 ('wire [182:0] service_observation_mux;','wire [86:0] service_observation_mux;'),
 ('schema_version=2,chunk_bits=128,chunks_per_page=4','schema_version=3,chunk_bits=32,chunks_per_page=16'),
 ('unchanged_legacy_payload_bits=384','unchanged_legacy_payload_bits=480')]
for old,new in changes:
 assert old in code,old;code=code.replace(old,new)
exec(compile(code,str(original)+':word-service','exec'),globals())
