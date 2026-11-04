"""Full retained DDR/Gray/camera/service audits plus current RAM JTAG payload."""
from pathlib import Path
source=Path(__file__).with_name('video_service_gray.py')
code=source.read_text()
needle="    driver=root.parent/(root.name+'-service-driver.py')"
extra="""    old='from audit_snapshot_jtag_paths import audit as audit_jtag'
    assert code.count(old)==1
    code=code.replace(old,'from video_service_fixedram_cdc import audit as audit_jtag')
"""
assert code.count(needle)==1;code=code.replace(needle,extra+needle)
exec(compile(code,str(source)+':fixedram','exec'),globals())
