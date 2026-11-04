"""User-approved offline limits: eight physical cores, 50% CPU, six GiB.

Preserve the original Job timeout, child-tree cleanup and six-GiB free-RAM
reserve. No changes to Windows global settings or frozen earlier launchers.
"""
import hashlib,inspect
from pathlib import Path
import windows_bounded_job as old
from windows_physical_cores_job import topology

def main():
    _,record=topology()
    record.update(user_authorized_cpu_increase=True,user_authorized_memory_increase=True,
        limits_unchanged=False,system_reserve_and_timeout_unchanged=True,
        wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_guard_sha256=hashlib.sha256(Path(old.__file__).read_bytes()).hexdigest())
    def choose_mask(available,count):
        choices=[m&available for m in record['core_masks'] if m&available]
        assert len(choices)>=count
        selected=sum(m&-m for m in choices[:count])
        assert selected.bit_count()==count and selected&available==selected
        record.update(selected_mask=selected,selected_physical_cores=count)
        return selected
    code=inspect.getsource(old.main)
    before='64<=a.memory_mb<=4096 and 1<=a.cores<=4 and 1<=a.cpu_percent<=25'
    assert code.count(before)==1
    code=code.replace(before,'64<=a.memory_mb<=6144 and 1<=a.cores<=8 and 1<=a.cpu_percent<=50')
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
