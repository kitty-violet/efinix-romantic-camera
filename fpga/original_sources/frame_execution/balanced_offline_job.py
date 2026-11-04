"""User-approved larger offline Job; original stop/memory protections retained.

Only this task's child tree is affected. Default callers still request explicit
limits; maximum eight separate physical cores, 50% CPU, four GiB memory.
"""
import hashlib,inspect,json,sys
from pathlib import Path
import windows_bounded_job as old
from windows_physical_cores_job import topology

def main():
    _,record=topology()
    record.update(user_authorized_cpu_increase=True,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_guard_sha256=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest(),
        topology_wrapper_sha256=hashlib.sha256(Path(__file__).with_name('windows_physical_cores_job.py').read_bytes()).hexdigest())
    def choose_mask(available,count):
        choices=[m&available for m in record['core_masks'] if m&available]
        assert len(choices)>=count
        selected=sum(m&-m for m in choices[:count])
        assert selected.bit_count()==count and selected&available==selected
        record.update(selected_mask=selected,selected_physical_cores=count)
        return selected
    code=inspect.getsource(old.main)
    before='1<=a.cores<=4 and 1<=a.cpu_percent<=25'
    assert code.count(before)==1
    code=code.replace(before,'1<=a.cores<=8 and 1<=a.cpu_percent<=50')
    before='mask=sum(1<<i for i in [i for i in range(ctypes.sizeof(ctypes.c_size_t)*8) if available.value>>i&1][:a.cores])'
    assert code.count(before)==1
    code=code.replace(before,'mask=choose_mask(available.value,a.cores)')
    def save(path,data):
        if path.name in ('progress.json','result.json'):
            data=dict(data,physical_core_selection=record)
        old.save(path,data)
    scope=dict(old.__dict__,choose_mask=choose_mask,save=save)
    exec(compile(code,__file__+':bounded-policy','exec'),scope)
    return scope['main']()

if __name__=='__main__':raise SystemExit(main())
