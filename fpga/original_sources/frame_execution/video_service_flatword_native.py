"""Share one phase/part decoder across the diagnostic data word.

This is an isolated area candidate. It changes no state, packet position,
sampling edge or functional path. Actual mapping and physical savings remain
separate evidence; the selector proof is for all reachable binary states.
"""
from video_service_narrow_native import display as original_display, serializer as original_serializer


def display(D, removed, edits):
    return original_display(D, removed, edits)


def serializer(D, removed, edits, supplied):
    code, receipt = original_serializer(D, removed, edits, supplied)
    rows = []
    for phase in (0, 1):
        for part in range(16):
            rows.append((phase, part, 'service_legacy_padded', part*2+phase))
    for phase, page in ((3, 'service_process_page'), (4, 'service_health_page'), (5, 'service_path_page')):
        for part in range(16):
            rows.append((phase, part, page, part))
    assert len(rows) == 80
    lines = ['wire [80:0] service_flat_select;', 'wire [31:0] service_flat_data;',
             'wire [95:0] service_flat_packet;']
    terms = []
    for i, (phase, part, page, word) in enumerate(rows):
        lines.append(f"assign service_flat_select[{i}]=(service_page_phase==3'd{phase}) && (service_page_part==4'd{part});")
        terms.append(f'({{32{{service_flat_select[{i}]}}}} & {page}[{word*32}+:32])')
    lines.append("assign service_flat_select[80]=(service_page_phase==3'd2) || (service_page_phase==3'd6) || (service_page_phase==3'd7);")
    terms.append('({32{service_flat_select[80]}} & service_pixel_cache)')
    lines += ['assign service_flat_data=\n '+' |\n '.join(terms)+';',
              'assign service_flat_packet={service_compact_packet[95:32],service_flat_data};']
    marker='video_snapshot_serializer service_serializer('
    assert code.count(marker)==1
    code=code.replace(marker,'\n'.join(lines)+'\n'+marker)
    assert code.count('.source_data(service_compact_packet)')==1
    code=code.replace('.source_data(service_compact_packet)', '.source_data(service_flat_packet)')
    cases=0
    for phase in range(8):
        for part in range(16):
            chosen=[(page,word) for p,k,page,word in rows if p==phase and k==part]
            if phase in (2,6,7):chosen.append(('service_pixel_cache',0))
            expected=('service_legacy_padded',part*2+phase) if phase<2 else (
                ({3:'service_process_page',4:'service_health_page',5:'service_path_page'}[phase],part)
                if phase in (3,4,5) else ('service_pixel_cache',0))
            assert chosen==[expected],(phase,part,chosen,expected)
            cases+=1
    receipt.update(flat_combined_word_decode=True, flat_binary_selector_cases=cases,
                   added_registers=0, latency_added=0, format_changed=False,
                   physical_improvement_unproved=True)
    return code,receipt
