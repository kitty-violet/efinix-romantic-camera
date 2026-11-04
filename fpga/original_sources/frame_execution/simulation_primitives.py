"""Add exact installed public primitive models only when selected RTL needs them."""
import re
from run_units import EF

def with_public_lut4(paths):
    paths=list(paths)
    texts=[f.read_text() for f in paths]
    if any(re.search(r'\bEFX_LUT4\s*#',s) for s in texts) and not any(re.search(r'\bmodule\s+EFX_LUT4\b',s) for s in texts):
        paths.append(EF/'sim_models/verilog/efx_lut4.v')
    return paths
