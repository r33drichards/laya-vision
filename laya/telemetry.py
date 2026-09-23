"""OpenTelemetry for training, eval and Modal jobs: traces, metrics and the job's stdout as logs, over OTLP/HTTP.

On by default whenever the ``opentelemetry-sdk`` and ``opentelemetry-exporter-otlp-proto-http`` packages are
installed (``pip install laya[otel]``; the Modal images have them) and a token is available: data goes to
``DEFAULT_ENDPOINT``, the project's otel-lgtm collector behind a bearer-token proxy, with the token from
``LAYA_OTLP_TOKEN`` (or the standard ``OTEL_EXPORTER_OTLP_HEADERS``). The Modal jobs get it from the ``laya-otel``
Modal secret. ``OTEL_EXPORTER_OTLP_ENDPOINT`` sends to another collector (``/v1/traces``, ``/v1/metrics`` and
``/v1/logs`` are appended; no token needed). ``OTEL_SDK_DISABLED=true`` or an empty ``OTEL_EXPORTER_OTLP_ENDPOINT``
turns it off; ``tests/conftest.py`` does that for pytest.

    @telemetry.traced_job(attrs=("run_name",))       # a span per call, stdout -> logs, numeric results -> metrics
    def evaluate(run_name, ...): ...

    with telemetry.job("vlm_train", run="smoke"):    # the same as a context manager
        telemetry.record("laya.train.loss", 0.42)    # a gauge; the job's attrs are added to every point
        telemetry.record_metrics("laya.eval", metrics_from(records), split="val")

Metric names become Prometheus names with dots turned into underscores (``laya_train_loss``). Attributes are
kept to low-cardinality values (job, run, dataset, split, ...): never a step number or an example id.
"""
import contextlib
import functools
import inspect
import logging
import math
import os
import sys
import threading
import traceback
from typing import Dict, Iterable, Optional

SERVICE_NAME = "laya-vision"
# otlp-auth on Railway (irc project; infra/telemetry/otlp-auth-proxy.ts): checks the token, forwards to otel-lgtm
DEFAULT_ENDPOINT = "https://otlp-auth-production.up.railway.app"
METRIC_EXPORT_MS = 15_000
FLUSH_MS = 10_000

_state = {"on": None, "pid": None, "tracer": None, "meter": None, "providers": [], "gauges": {}}
_attrs = threading.local()  # the enclosing jobs' attributes, merged into every metric point
_lock = threading.Lock()


def endpoint() -> Optional[str]:
    """The collector's base URL, or None when telemetry is turned off (including the default collector without a
    token, which would only answer 401)."""
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        return None
    if "OTEL_EXPORTER_OTLP_ENDPOINT" in os.environ:
        return os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"].strip().rstrip("/") or None
    if not (os.environ.get("LAYA_OTLP_TOKEN") or os.environ.get("OTEL_EXPORTER_OTLP_HEADERS")):
        return None
    return DEFAULT_ENDPOINT


def _headers() -> Optional[Dict[str, str]]:
    """``LAYA_OTLP_TOKEN`` as a bearer header; None lets the exporters read ``OTEL_EXPORTER_OTLP_HEADERS``."""
    token = os.environ.get("LAYA_OTLP_TOKEN", "").strip()
    return {"authorization": "Bearer " + token} if token else None


def enabled() -> bool:
    return bool(setup())


def setup(service_name: str = SERVICE_NAME) -> bool:
    """Configure the global tracer, meter and log providers once per process. Returns whether telemetry is on."""
    with _lock:
        if _state["on"] is not None:
            # a forked child (a DataLoader worker) inherits the providers but not their export threads: stay quiet
            return _state["on"] and _state["pid"] == os.getpid()
        _state.update(on=False, pid=os.getpid())
        url = endpoint()
        if not url:
            return False
        try:
            from opentelemetry import metrics, trace
            from opentelemetry._logs import set_logger_provider
            from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
            from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
        except ImportError:
            if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):  # asked for explicitly: say why nothing is sent
                print("telemetry: OTEL_EXPORTER_OTLP_ENDPOINT is set but opentelemetry-sdk / "
                      "opentelemetry-exporter-otlp-proto-http are not installed; telemetry is off", file=sys.stderr)
            return False

        from . import __version__

        res = {"service.name": os.environ.get("OTEL_SERVICE_NAME", service_name), "service.version": __version__}
        for env, key in (("MODAL_TASK_ID", "modal.task_id"), ("MODAL_REGION", "cloud.region"),
                         ("MODAL_CLOUD_PROVIDER", "cloud.provider"), ("MODAL_ENVIRONMENT", "modal.environment"),
                         ("LAYA_GIT_SHA", "vcs.revision")):
            if os.environ.get(env):
                res[key] = os.environ[env]
        resource = Resource.create(res)  # also merges OTEL_RESOURCE_ATTRIBUTES

        tp = TracerProvider(resource=resource)
        headers = _headers()
        tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=url + "/v1/traces", headers=headers)))
        trace.set_tracer_provider(tp)
        reader = PeriodicExportingMetricReader(OTLPMetricExporter(endpoint=url + "/v1/metrics", headers=headers),
                                               export_interval_millis=METRIC_EXPORT_MS)
        mp = MeterProvider(resource=resource, metric_readers=[reader])
        metrics.set_meter_provider(mp)
        lp = LoggerProvider(resource=resource)
        lp.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=url + "/v1/logs", headers=headers)))
        set_logger_provider(lp)

        stdout_log = logging.getLogger("laya.stdout")
        stdout_log.setLevel(logging.INFO)
        stdout_log.propagate = False  # the line is already on the real stdout
        stdout_log.addHandler(LoggingHandler(level=logging.INFO, logger_provider=lp))

        _state.update(on=True, tracer=tp.get_tracer("laya"), meter=mp.get_meter("laya"), providers=[tp, mp, lp])
        return True


def flush() -> None:
    for p in _state["providers"]:
        try:
            p.force_flush(FLUSH_MS)
        except Exception as e:  # telemetry never fails a job
            print("telemetry: flush failed: %r" % e, file=sys.stderr)


def _current_attrs() -> Dict:
    return getattr(_attrs, "value", {})


def _clean(attrs: Dict) -> Dict:
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, bool, int, float)) else str(v)
    return out


def record(name: str, value, **attrs) -> None:
    """Set gauge ``name`` to ``value``; the enclosing jobs' attributes are added. Non-finite values are skipped."""
    if not setup() or value is None:
        return
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(value):
        return
    g = _state["gauges"].get(name)
    if g is None:
        g = _state["gauges"][name] = _state["meter"].create_gauge(name)
    g.set(value, _clean({**_current_attrs(), **attrs}))


def record_metrics(prefix: str, metrics: Dict[str, Dict], **attrs) -> None:
    """Record a ``laya.vlm_train.metrics_from`` result (``{dataset: {"acc", "ece", "nll", ...}}``) as
    ``<prefix>.<metric>`` gauges with a ``dataset`` attribute (``all`` is the pooled row)."""
    for dataset, row in (metrics or {}).items():
        if not isinstance(row, dict):
            continue
        for key, v in row.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                record("%s.%s" % (prefix, key), v, dataset=dataset, **attrs)


def record_results(prefix: str, result, keys: Optional[Iterable[str]] = None, **attrs) -> None:
    """Record the numeric top-level values of a result dict (``median_ms``, ``kill_rate``, ...) as
    ``<prefix>.<key>`` gauges, and set them on the current span."""
    if not isinstance(result, dict) or not setup():
        return
    from opentelemetry import trace

    span = trace.get_current_span()
    for k, v in result.items():
        if keys is not None and k not in keys:
            continue
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            record("%s.%s" % (prefix, k), v, **attrs)
            span.set_attribute("laya.result.%s" % k, v)
        elif isinstance(v, str) and len(v) <= 200:
            span.set_attribute("laya.result.%s" % k, v)


@contextlib.contextmanager
def span(name: str, **attrs):
    """A child span of the current one; nothing when telemetry is off."""
    if not setup():
        yield None
        return
    with _state["tracer"].start_as_current_span(name, attributes=_clean(attrs)) as s:
        yield s


class _Tee:
    """Wraps sys.stdout: writes through, and sends every complete line to the ``laya.stdout`` logger (so it lands
    in Loki with the current span's trace id). Only the process that set telemetry up emits: forked DataLoader
    workers inherit the wrapper but just write through."""

    def __init__(self, stream, pid):
        self._stream, self._pid, self._buf = stream, pid, ""
        self._log = logging.getLogger("laya.stdout")
        self._busy = threading.local()

    def write(self, s):
        n = self._stream.write(s)
        if os.getpid() == self._pid and not getattr(self._busy, "on", False):
            self._buf += s
            if "\n" in self._buf:
                *lines, self._buf = self._buf.split("\n")
                self._busy.on = True
                try:
                    for line in lines:
                        if line.strip():
                            self._log.info(line)
                finally:
                    self._busy.on = False
        return n

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


@contextlib.contextmanager
def job(name: str, **attrs):
    """Run a job under a root span ``<name>``, send its stdout as logs, add ``job=<name>`` and ``attrs`` to every
    metric point inside, record failures on the span, and flush everything at the end (Modal containers can be
    torn down right after a call returns). A job inside a job is just a child span."""
    if not setup():
        yield None
        return
    attrs = _clean(dict(job=name, **attrs))
    prev = _current_attrs()
    _attrs.value = {**prev, **attrs}
    tee = None
    if not isinstance(sys.stdout, _Tee):
        tee = sys.stdout = _Tee(sys.stdout, os.getpid())
    try:
        with _state["tracer"].start_as_current_span(name, attributes={"laya." + k: v for k, v in attrs.items()},
                                                    record_exception=True, set_status_on_exception=True) as s:
            try:
                yield s
            except BaseException:
                logging.getLogger("laya.stdout").error("%s failed:\n%s" % (name, traceback.format_exc()))
                raise
    finally:
        _attrs.value = prev
        if tee is not None:
            sys.stdout = tee._stream
        flush()


def traced_job(name: Optional[str] = None, attrs: Iterable[str] = ()):
    """Decorator form of ``job``: ``attrs`` names the function's arguments to put on the span and on every metric
    point (e.g. ``("run_name", "backbone")``), and a dict return value's numeric entries are recorded as
    ``laya.result.<key>`` gauges. Keeps the signature, so Modal still parses CLI arguments from it."""
    attrs = tuple(attrs)

    def deco(fn):
        sig = inspect.signature(fn)
        job_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            values = {}
            if attrs:
                bound = sig.bind_partial(*args, **kwargs)
                bound.apply_defaults()
                values = {k: bound.arguments[k] for k in attrs if k in bound.arguments}
            with job(job_name, **values):
                result = fn(*args, **kwargs)
                record_results("laya.result", result)
                return result

        return wrapper

    return deco
