"""Wall-clock ceiling independent of simulator stdout/progress activity."""
import json
import math
import threading
import time
from owned_simulator_process import stop_owned_tree

MAX_HOST_SECONDS = 4 * 60 * 60


class HostRuntimeBudget:
    def __init__(self, proc, seconds, root, started=None):
        if not math.isfinite(seconds) or not 0 < seconds <= MAX_HOST_SECONDS:
            raise ValueError('Host budget must be positive and at most four hours')
        self.proc = proc
        self.seconds = seconds
        self.root = root
        self.started = time.monotonic() if started is None else started
        self.result = None
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self.thread = threading.Thread(target=self._wait, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _save(self):
        path = self.root / 'host_timeout.json'
        staging = self.root / 'host_timeout.json.tmp'
        staging.write_text(json.dumps(self.result, indent=2), encoding='utf8')
        staging.replace(path)

    def _wait(self):
        remaining = max(0, self.seconds - (time.monotonic() - self.started))
        if self.cancelled.wait(remaining):
            return
        with self.lock:
            if self.cancelled.is_set() or self.proc.poll() is not None:
                return
            self.result = dict(kind='HOST_TIME_BUDGET',
                status='STOPPING_OWNED_SIMULATION_FOR_HOST_BUDGET',
                limit_seconds=self.seconds,
                elapsed_seconds=time.monotonic() - self.started,
                complete_frame=False,
                reason='User maximum wall-clock simulation budget exceeded; preserve prefix, do not retry.')
            try:
                self._save()
            except Exception as exc:
                self.result['record_error'] = repr(exc)
            # Even a failed evidence write must not disable the runtime ceiling.
            try:
                self.result['termination'] = stop_owned_tree(self.proc)
                self.result['status'] = 'HOST_BUDGET_STOP_REQUESTED'
            except Exception as exc:
                self.result['status'] = 'HOST_BUDGET_TERMINATION_ERROR'
                self.result['termination_error'] = repr(exc)
            try:
                self._save()
            except Exception as exc:
                self.result['record_error'] = repr(exc)

    def finish(self):
        self.cancelled.set()
        self.thread.join()
        return self.result
