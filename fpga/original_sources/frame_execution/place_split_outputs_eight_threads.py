"""Reuse the audited native placement with eight official worker threads."""
import json,runpy,sys
from pathlib import Path
from run_units import HERE,sha

def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    assert not root.exists()
    source=HERE/'place_split_outputs_four_threads.py'
    code=source.read_text()
    for before,after in [('value="4"','value="8"'),('official_max_threads=4','official_max_threads=8'),
                         ('max_threads=4','max_threads=8'),("root.name+'-four-thread-wrapper.py'","root.name+'-eight-thread-wrapper.py'")]:
        assert code.count(before)==1,before
        code=code.replace(before,after)
    driver=root.parent/(root.name+'-eight-thread-source.py')
    assert not driver.exists()
    compile(code,str(driver),'exec');driver.write_text(code,encoding='utf8')
    driver.with_suffix('.json').write_text(json.dumps(dict(source=str(source),source_sha256=sha(source),
        driver_sha256=sha(driver),wrapper_sha256=sha(__file__),max_threads=8,hardware_access=False),indent=2))
    sys.argv=[str(driver),*sys.argv[1:]]
    runpy.run_path(str(driver),run_name='__main__')

if __name__=='__main__':main()
