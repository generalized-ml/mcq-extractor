"""Recorded PDF wall time across bounded/resumed pipeline invocations."""

from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
import uuid

from .hybrid_mcq.run import save_json


class RuntimeRecorder:
    def __init__(self, root: Path, method: str, *, started=None, history_complete=True):
        self.path = root / 'runtime.json'
        self.started = time.perf_counter() if started is None else started
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {
            'method': method, 'scope': 'pipeline_wall_time_excluding_evaluation',
            'history_complete': history_complete, 'sessions': []}
        if self.data['method'] != method:
            raise ValueError(f'Runtime method differs from saved run: {self.path}')
        self.session = {'id': uuid.uuid4().hex, 'started_at_utc': datetime.now(timezone.utc).isoformat(),
                        'seconds': 0.0, 'status': 'running'}
        self.data['sessions'].append(self.session)

    def save(self, *, finished=False):
        self.session.update(seconds=time.perf_counter() - self.started,
                            status='finished' if finished else 'running')
        save_json(self.path, self.data)


def recorded_runtime(prediction_path: Path, method: str | None) -> dict:
    """Read a PDF's sidecar; question CPU times cannot substitute for wall time."""
    for root in (prediction_path.parent, prediction_path.parent.parent):
        path = root / 'runtime.json'
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        if method is not None and data.get('method') != method:
            continue
        sessions = data.get('sessions')
        if not isinstance(sessions, list) or not sessions:
            raise ValueError(f'Invalid runtime sessions: {path}')
        values = [session.get('seconds') for session in sessions]
        if any(isinstance(value, bool) or not isinstance(value, (int, float))
               or not math.isfinite(value) or value < 0 for value in values):
            raise ValueError(f'Invalid runtime duration: {path}')
        return {'method_wall_seconds': math.fsum(values), 'method_timed_pdfs': 1,
                'method_missing_time_pdfs': 0,
                'method_unfinished_time_pdfs': int(any(s.get('status') != 'finished' for s in sessions)),
                'method_incomplete_history_pdfs': int(not data.get('history_complete', False)),
                'method_runtime_scope': data.get('scope', 'unknown')}
    return {'method_wall_seconds': None, 'method_timed_pdfs': 0,
            'method_missing_time_pdfs': 1, 'method_unfinished_time_pdfs': 0, 'method_incomplete_history_pdfs': 0,
            'method_runtime_scope': 'unknown'}


def combine_runtimes(reports):
    values = [r['method_wall_seconds'] for r in reports if r['method_wall_seconds'] is not None]
    scopes = {r['method_runtime_scope'] for r in reports if r['method_wall_seconds'] is not None}
    return {'method_wall_seconds': math.fsum(values) if values else None,
            **{key: sum(r[key] for r in reports) for key in
               ('method_timed_pdfs', 'method_missing_time_pdfs', 'method_unfinished_time_pdfs',
                'method_incomplete_history_pdfs')},
            'method_runtime_scope': next(iter(scopes)) if len(scopes) == 1 else 'unknown' if not scopes else 'mixed'}
