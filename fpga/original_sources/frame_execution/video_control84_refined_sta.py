"""Original full2000-path STA; accept the bound refinement comment spelling."""
from pathlib import Path

source_text=Path(__file__).with_name('canonical_snapshot_sta_isolated.py').read_text()
needle="exec(compile(code,str(source)+':isolated-reporter-processes','exec'),globals())"
old="    assert (base/'ti60f225_oob.sdc').read_text() in allowed"
new="""    if reserve==.3:
        assert sha(margin['proof'])==margin['proof_sha256']
        allowed.append(canonical.read_text()+'\\n# Optimization reserve only; release uses original SDC.\\nset_clock_uncertainty -setup -to [get_clocks {*}] 0.300\\n')
    assert (base/'ti60f225_oob.sdc').read_text() in allowed"""
assert source_text.count(needle)==1
source_text=source_text.replace(needle,
 "assert code.count('-npaths 5 -nworst 1')==2\n"
 "code=code.replace('-npaths 5 -nworst 1','-npaths 2000 -nworst 1')\n"
 "code=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')\n"
 f"assert code.count({old!r})==1\ncode=code.replace({old!r},{new!r})\n"+needle)
exec(compile(source_text,__file__+':refined-bound-original2000','exec'),globals())
