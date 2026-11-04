"""Post-map audit: prove deletion of the five now-unused old valid LUTs.

The map input/runner receipt and its first conservative audit failure are
preserved. This auditor neither changes nor reruns synthesis.
"""
from pathlib import Path
from control_command_repair import once
WRAPPER=Path(__file__).resolve()
SOURCE=WRAPPER.with_name('continuous_control_board.py')
code=SOURCE.read_text(encoding='utf8')
code=once(code,'    assert not missing,sorted(missing)', '''    expected={f'LUT__{i}' for i in range(159749,159754)}
    assert missing==expected,sorted(missing)
    from audit_continuous_cut import expand_graph
    expanded=expand_graph(source)
    dead={t for n in missing for t in expanded[n][1]['O']}
    assert all(expanded[n][0]=='EFX_LUT4' for n in missing)
    assert not [(n,p,t) for n,(k,ps) in expanded.items() if n not in missing
                for p,ts in ps.items() for t in ts if t in dead]
    # The outputs are internal n-nets and have no assignments, top ports,
    # module boundaries or protected-cell users outside their own LUTs.
    clean=CELL.sub(lambda z:'' if z[2].lstrip('\\\\') in missing else z[0],source)
    for t in dead:
        assert t[0]=='net' and t[2] is None and re.fullmatch(r'n\\d+_2',t[1])
        assert len(re.findall(r'\\b'+re.escape(t[1])+r'\\b',clean))==1,t
''')
code=once(code,'verify(flattened_public_ports(source),flattened_public_ports(filtered))',
          'verify(flattened_public_ports(clean),flattened_public_ports(filtered))')
code=once(code,'lexical_roundtrip(source,filtered)','lexical_roundtrip(clean,filtered)')
code=once(code,"Path(__file__),TRANSFORM/'result.json'","Path(__file__),POSTMAP_AUDITOR,TRANSFORM/'result.json'")
code=once(code,"lexical_audit=lex,protected_boundary=boundary,", "removed_dead_old_valid_luts=sorted(missing),postmap_auditor_sha256=sha(POSTMAP_AUDITOR),lexical_audit=lex,protected_boundary=boundary,")
code=once(code,"(BOARD/'failure.json')","(BOARD/'postmap_audit_failure2.json')")
scope={'__file__':str(SOURCE),'__name__':'_continuous_postmap','POSTMAP_AUDITOR':WRAPPER}
exec(compile(code,str(SOURCE)+':postmap-dead-valid-cone','exec'),scope)
BOARD=scope['BOARD'];transform_check=scope['transform_check']

def audit_delta(board):
    result=scope['audit_delta'](board)
    import json
    from run_units import sha
    r=json.loads((Path(board)/'result.json').read_text())
    assert r['postmap_auditor_sha256']==sha(WRAPPER)
    return result

if __name__=='__main__':
    import sys
    assert sys.argv[1:]==['--audit-only'],'This version only audits the existing exact map'
    scope['main']()
