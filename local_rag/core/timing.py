"""Per-request timings; nested retrieval does not double-count the same stage."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter

_current = ContextVar("rag_timing", default=None)
_phase = ContextVar("rag_model_phase", default="answer")


class Trace:
    def __init__(self):
        self.started = perf_counter()
        self.durations = {}
        self.active = set()
        self.calls = []

    def result(self):
        result = {name + "_ms": round(value * 1000, 3) for name, value in self.durations.items()}
        result["total_ms"] = round((perf_counter() - self.started) * 1000, 3)
        result["embedding_chroma_ms"] = round(sum(result.get(k, 0) for k in ("embedding_ms", "chroma_ms")), 3)
        chats = [c for c in self.calls if c["kind"] == "chat"]
        result["llm_ms"] = round(sum(c["wall_ms"] for c in chats), 3)
        result["generation_ms"] = round(sum(c["wall_ms"] for c in chats
                                           if c["phase"] in {"answer", "answer_translation"}), 3)
        result["llm_load_ms"] = round(sum(c["load_ms"] for c in chats), 3)
        result["llm_eval_ms"] = round(sum(c["eval_ms"] for c in chats), 3)
        result["model_calls"] = list(self.calls)
        return result


@contextmanager
def stage(name):
    trace = _current.get()
    if trace is None or name in trace.active:
        yield
        return
    trace.active.add(name)
    started = perf_counter()
    try:
        yield
    finally:
        trace.active.remove(name)
        trace.durations[name] = trace.durations.get(name, 0) + perf_counter() - started


def timed_operation(name):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            owner = _current.get() is None
            trace = Trace() if owner else _current.get()
            token = _current.set(trace) if owner else None
            try:
                with stage(name):
                    result = fn(*args, **kwargs)
                if isinstance(result, dict):
                    result["timings"] = trace.result()
                return result
            finally:
                if owner:
                    _current.reset(token)
        return wrapper
    return decorate


def model_phase(name):
    def decorate(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            token = _phase.set(name)
            try:
                return fn(*args, **kwargs)
            finally:
                _phase.reset(token)
        return wrapper
    return decorate


def record_model_call(kind, model, response, seconds):
    trace = _current.get()
    if trace is None:
        return
    def duration(name):
        value = response.get(name, 0)
        return round(value / 1_000_000, 3) if type(value) in (int, float) and value >= 0 else 0.0
    trace.calls.append({"kind": kind, "phase": _phase.get() if kind == "chat" else "embedding",
                        "model": model, "wall_ms": round(seconds * 1000, 3),
                        "load_ms": duration("load_duration"), "prompt_eval_ms": duration("prompt_eval_duration"),
                        "eval_ms": duration("eval_duration"), "total_ms": duration("total_duration"),
                        "eval_tokens": response.get("eval_count", 0)})
