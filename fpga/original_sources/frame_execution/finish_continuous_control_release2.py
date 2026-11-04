"""Bind the importable snapshot entry without changing hardware or old proofs."""
from pathlib import Path
source=Path(__file__).with_name('finish_continuous_control_release.py')
code=source.read_text(encoding='utf8')
old="    decoder=bind(ROOT/'tmp/board-test/continuous-progress-decoder1.json');verify(decoder['evidence'])"
new="""    entry=bind(ROOT/'tmp/board-test/continuous-package-entry1.json');verify(entry['evidence'])
    assert entry['status']=='CONTINUOUS_PACKAGE_READER_IMPORT_HELP_AND_GATE_PASS'
"""+old
assert code.count(old)==1;code=code.replace(old,new)
old=".replace('read_cnn_observer_trim_snapshot','read_continuous_snapshot')"
assert code.count(old)==1;code=code.replace(old,".replace('read_cnn_observer_trim_snapshot','read_continuous_snapshot_v2')")
exec(compile(code,str(source)+':entrypoint2','exec'),globals())
