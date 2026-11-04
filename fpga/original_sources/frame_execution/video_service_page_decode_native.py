"""Flatten diagnostic selection into mutually exclusive data masks.

The original selection separately multiplexes each 512-bit page down to128
then multiplexes the pages. A single shared phase/part decode selects the same
word. Keep all32 binary selector combinations including illegal phases; no
register, field, cycle, or external handshake changes.
"""
from video_service_display_native import extension as original
def extension(D,removed,edits):
 code,receipt=original(D,removed,edits)
 start=code.index('    assign service_selected_page=')
 end=code.index(';',start)+1
 terms=[];decl=['    wire [12:0] service_word_select;']
 for j,(phase,page) in enumerate(((3,'service_process_page'),(4,'service_health_page'),(5,'service_path_page'))):
  for part in range(4):
   idx=4*j+part
   decl.append(f"    assign service_word_select[{idx}]=(service_page_phase==3'd{phase}) && (service_page_part==2'd{part});")
   terms.append(f'({{128{{service_word_select[{idx}]}}}} & {page}[{part*128}+:128])')
 decl.append("    assign service_word_select[12]=(service_page_phase!=3'd3) && (service_page_phase!=3'd4) && (service_page_phase!=3'd5);")
 terms.append('({128{service_word_select[12]}} & service_pixel_cache)')
 replacement='\n'.join(decl)+'\n    assign service_selected_page=\n      '+' |\n      '.join(terms)+';'
 code=code[:start]+replacement+code[end:]
 # Exhaustive selector proof independent of any physical resource claim.
 cases=0
 for phase in range(8):
  for part in range(4):
   active=[4*j+k for j,p in enumerate((3,4,5)) for k in range(4) if p==phase and k==part]
   if phase not in (3,4,5):active.append(12)
   expected=4*(phase-3)+part if phase in (3,4,5) else 12
   assert active==[expected];cases+=1
 receipt.update(flat_word_decode=True,selector_truth_cases=cases,latency_added=0,fields_preserved=True)
 return code,receipt
