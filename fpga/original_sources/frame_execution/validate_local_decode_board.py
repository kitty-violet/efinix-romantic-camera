"""Run unchanged OFP3/serial/mapped regressions against the local decode copy."""
import json,sys,subprocess
from pathlib import Path
from run_units import HERE,sha
from local_control_decode import audit_delta
import control_ddr_combined
import validate_control_ddr

if __name__=='__main__':
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    # Same documented Starter performance-warning handling already used by
    # run_board_startup.py. Keep the warning visible; no functional error,
    # assertion, license failure, or encrypted-model check is suppressed.
    original_popen=subprocess.Popen;actual_commands=[]
    def launch(args,*pos,**kwargs):
        if Path(str(args[0])).name.lower()=='vsim.exe':
            args=[args[0],'-warning','14408,16154','-voptargs=-warning 14408',*args[1:]]
            actual_commands.append(list(map(str,args)))
        return original_popen(args,*pos,**kwargs)
    subprocess.Popen=launch
    control_ddr_combined.audit_delta=audit_delta
    validate_control_ddr.audit_delta=audit_delta
    if sys.argv[1]=='observe':
        del sys.argv[1]
        from validate_control_ddr_observation import main
        main()
    else:validate_control_ddr.main()
    (root/'local_decode_adapter.json').write_text(json.dumps(dict(wrapper_sha256=sha(__file__),
        regression_sha256=sha(HERE/'validate_control_ddr.py'),
        transform_sha256=sha(HERE/'local_control_decode.py'),
        actual_simulator_commands=actual_commands,
        change='Accepted local source delta and existing Starter capacity performance warnings; functional test expectations unchanged.',hardware_access=False),indent=2))
