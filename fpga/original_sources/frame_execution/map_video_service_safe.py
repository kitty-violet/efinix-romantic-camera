"""Existing synthesis probe with bounded, handle-checked child cleanup."""
import json,re,subprocess,time
from pathlib import Path
from windows_process_tree_safe import stop_launched_tree
from run_units import sha
source=Path(__file__).with_name('map_video_service_candidate.py')
code=source.read_text();old='from run_units import ROOT,HERE,sha,run'
assert code.count(old)==1
serial="'video_service_modes.sv','video_service_health.sv']"
assert code.count(serial)==1
code=code.replace(serial,"'video_service_modes.sv','video_service_health.sv','video_snapshot_serializer.sv']")
def run(args,build,name,timeout=600):
    t=time.monotonic()
    with (build/name).open('x',encoding='utf8') as log:
        proc=subprocess.Popen(list(map(str,args)),cwd=build,stdout=log,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            (build/(name+'-timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2));raise
    text=(build/name).read_text()
    if rc or re.search(r'(?m)^#?\s*(FAIL|\*\* (Error|Fatal))',text):raise RuntimeError(name+text[-2000:])
    return text,time.monotonic()-t
frozen=sha(source)
exec(compile(code.replace(old,'from run_units import ROOT,HERE,sha'),str(source)+':safe-cleanup','exec'),globals())
assert sha(source)==frozen
