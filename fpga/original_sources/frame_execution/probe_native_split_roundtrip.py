"""Exercise the linear decomposition through two real small map runs."""
import json
import probe_native_map_roundtrip as probe
from native_logic_split import split_logic_comb
from run_units import HERE, sha

if __name__ == '__main__':
    probe.split_logic_comb = split_logic_comb
    probe.main()
    (probe.RUN/'transform_binding.json').write_text(json.dumps(dict(
        runner_sha256=sha(__file__),base_runner_sha256=sha(probe.__file__),
        transform_sha256=sha(HERE/'native_logic_split.py'),
        result_sha256=sha(probe.RUN/'result.json')),indent=2))
