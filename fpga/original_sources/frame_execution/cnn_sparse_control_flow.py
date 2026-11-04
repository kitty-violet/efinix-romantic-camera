"""All existing current-map short gates, without reusing another map's prefix."""
from pathlib import Path
SOURCE=Path(__file__).with_name('camera_payload_flow.py')
code=SOURCE.read_text(encoding='utf8').replace('camera_payload','cnn_sparse_control')
code=code.replace("HERE/'camera_checked_gray_v2.py'","HERE/'camera_joint_gray.py'")
needle="        generated(source,code,'-payload-short-wrapper')"
assert code.count(needle)==1
injection='''        old="[sys.executable,'-B',str(HERE/script),*map(str,args)]"
        new="[sys.executable,'-B',str(ROOT/'tmp/board-test/run-with-locked-evidence.py'),'--cache-log',str(root/(name+'-cache.json')),str(HERE/script),*map(str,args)]"
        assert code.count(old)==1;code=code.replace(old,new)
        code=once(code,"root/'map-audit'],100)","root/'map-audit'],240)")
'''
code=code.replace(needle,injection+needle)
exec(compile(code,str(SOURCE)+':sparse-control','exec'),globals())
