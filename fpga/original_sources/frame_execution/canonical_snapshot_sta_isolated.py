"""Run all original STA checks in two processes to avoid reporter-state crashes.

No timing constraint or acceptance rule is removed. Both processes must finish
successfully, and the original geometry/source checks still execute verbatim.
"""
from pathlib import Path
from control_command_repair import once
from run_units import sha
source=Path(__file__).with_name('canonical_snapshot_sta.py')
code=source.read_text(encoding='utf8')
code=once(code,"    'puts {BEGIN_CHECKS}; flush stdout\\n'\n",'')
code=once(code,"    'check_timing -file canonical_checks.rpt -verbose\\n'\n",'')
code=once(code,"'puts {COMPLETE_ALL}; flush stdout\\n'","'puts {COMPLETE_TIMING}; flush stdout\\n'")
code=once(code,"runner_sha256=sha(__file__),timing_policy_sha256=", "runner_sha256=sha(__file__),canonical_source_sha256=sha(source),process_isolation='timing_then_checks_both_required',timing_policy_sha256=")
begin=code.index('    with (root/\'sta.log\').open(\'w\') as log:')
end=code.index("    assert code==0,(code,body[-3000:])",begin)
replacement='''    stages=[]
    checks_root=root/'ck'
    checks_root.mkdir()
    for folder in ('outflow','work_pnr'): shutil.copytree(root/folder,checks_root/folder)
    for f in root.iterdir():
        if f.is_file() and f.suffix in ('.xml','.sdc','.ini','.placeloc'): shutil.copy2(f,checks_root/f.name)
    def run_stage(name,command,cwd):
        start=time.monotonic()
        with (root/(name+'.log')).open('x') as log:
            proc=subprocess.Popen(['cmd.exe','/d','/c',command],cwd=cwd,stdout=log,stderr=subprocess.STDOUT)
            try: rc=proc.wait(timeout=210)
            except subprocess.TimeoutExpired:
                (root/(name+'-timeout.json')).write_text(json.dumps(stop_launched_tree(proc),indent=2))
                raise
        body=(root/(name+'.log')).read_text(errors='replace')
        body+='\\n'+'\\n'.join(f.read_text(errors='replace') for f in (cwd/'outflow').glob('*.sta.out'))
        stages.append(dict(name=name,returncode=rc,seconds=time.monotonic()-start,log_sha256=sha(root/(name+'.log'))))
        (root/'isolated_stages.json').write_text(json.dumps(stages,indent=2))
        assert rc==0,(name,rc,body[-3000:])
        return body
    timing_body=run_stage('sta','run.cmd',root)
    assert 'COMPLETE_TIMING' in timing_body
    (checks_root/'isolated.tcl').write_text('report_timing_summary -file isolated_summary.rpt\\ncheck_timing -file isolated_checks.rpt -verbose\\nputs {COMPLETE_CHECKS}; flush stdout\\n')
    (checks_root/'run.cmd').write_text((root/'run.cmd').read_text().replace('canonical.tcl','isolated.tcl'))
    checks_body=run_stage('checks','run.cmd',checks_root)
    assert 'COMPLETE_CHECKS' in checks_body
    for f in geometry: assert sha(checks_root/'outflow'/f.name)==frozen[str(f)]
    assert sha(checks_root/'ti60f225_oob.sdc')==sha(canonical)
    shutil.copy2(checks_root/'outflow/isolated_checks.rpt',root/'outflow/canonical_checks.rpt')
    body=timing_body+'\\n'+checks_body
    body+='\\n'+'\\n'.join(f.read_text(errors='replace') for f in (root/'outflow').glob('*.sta.out'))
    code=0
'''
code=code[:begin]+replacement+code[end:]
code=once(code,"reports={n:sha(root/'outflow'/n) for n in names},", "reports={n:sha(root/'outflow'/n) for n in names},isolated_stages_sha256=sha(root/'isolated_stages.json'),")
exec(compile(code,str(source)+':isolated-reporter-processes','exec'),globals())
