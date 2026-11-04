"""Compare exact unchanged feed semantics with explicit fixed SRL tap segments."""
import concurrent.futures,json
from run_units import ROOT,HERE
from map_feed_layout_trials import one

if __name__=='__main__':
    root=ROOT/'tmp/frame-execution/segmented-history-screen1'
    assert not root.exists()
    root.mkdir()
    choices={n:HERE/'variants'/v/'unified_column_dispatch32.sv' for n,v in
        [('reference','shared_existing_taps'),('weight','segmented_weight'),('history','segmented_history')]}
    results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs={n:pool.submit(one,root/n,s) for n,s in choices.items()}
        for n,f in jobs.items():
            results[n]=f.result();print(n,json.dumps(results[n]),flush=True)
    (root/'result.json').write_text(json.dumps(dict(status='SEGMENTED_HISTORY_SCREEN_COMPLETE_NOT_BOARD',trials=results),indent=2))
