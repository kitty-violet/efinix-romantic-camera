"""Original acceptance checks with 2000 distinct setup/hold endpoints per run.

Only report breadth and bounded host runtime differ from video_service_sta.
No constraint, geometry, acceptance rule or mandatory timing check changes.
"""
from pathlib import Path
from run_units import sha

source = Path(__file__).with_name('canonical_snapshot_sta_isolated.py')
code = source.read_text(encoding='utf8')
needle = "code=source.read_text(encoding='utf8')"
assert code.count(needle) == 1
code = code.replace(needle, needle + """
assert code.count('-npaths 5 -nworst 1') == 2
code=code.replace('from windows_process_tree import stop_launched_tree','from windows_process_tree_safe import stop_launched_tree')
code=code.replace('-npaths 5 -nworst 1','-npaths 2000 -nworst 1')
assert code.count('--timeout 180') == 1
code=code.replace('--timeout 180','--timeout 360')
""")
assert code.count('timeout=210') == 1
code = code.replace('timeout=210', 'timeout=390')
adapter_sources = {
    str(source): sha(source),
    str(Path(__file__)): sha(__file__),
    str(source.with_name('canonical_snapshot_sta.py')): sha(source.with_name('canonical_snapshot_sta.py')),
}
exec(compile(code, str(source) + ':full-endpoint-acceptance', 'exec'), globals())
output.update(adapter_sources=adapter_sources, requested_paths=2000,
              paths_per_endpoint=1, full_original_acceptance_checks=True,
              board_release=False)
(root/'result.json').write_text(json.dumps(output, indent=2)+'\n')
