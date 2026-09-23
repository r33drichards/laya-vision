"""``laya.telemetry``: on by default, a no-op when turned off, and OTLP/HTTP traces, metrics and stdout logs.

The exporting test runs in a subprocess (the OpenTelemetry providers are process-global) against a local HTTP
server that stands in for the collector and records the paths it is posted to.
"""
import inspect
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from laya import telemetry

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_endpoint_defaults_and_opt_outs(monkeypatch):
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    assert telemetry.endpoint() == telemetry.DEFAULT_ENDPOINT
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4318/")
    assert telemetry.endpoint() == "http://collector:4318"
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    assert telemetry.endpoint() is None
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert telemetry.endpoint() is None


def test_off_when_disabled(capsys):
    # tests/conftest.py sets OTEL_SDK_DISABLED=true, so nothing here is sent anywhere
    if telemetry.enabled():
        pytest.skip("telemetry is configured in this process")

    @telemetry.traced_job(attrs=("run_name",))
    def job(run_name: str, n: int = 2):
        """doc"""
        print("hello")
        telemetry.record("laya.test.value", 1.0)
        with telemetry.span("inner"):
            return {"n": n}

    assert job("r") == {"n": 2}
    assert capsys.readouterr().out == "hello\n"
    assert list(inspect.signature(job).parameters) == ["run_name", "n"] and job.__doc__ == "doc"
    assert sys.stdout.__class__.__name__ != "_Tee"


class _Collector(BaseHTTPRequestHandler):
    paths = []

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Collector.paths.append(self.path)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *args):
        pass


CHILD = r"""
from laya import telemetry

@telemetry.traced_job(attrs=("run_name",))
def job(run_name):
    print("a line for the logs")
    telemetry.record("laya.test.value", 3.0, dataset="x")
    telemetry.record_metrics("laya.test.eval", {"all": {"n": 5, "acc": 0.5}})
    return {"median_ms": 12.5}

assert telemetry.enabled()
assert job("smoke") == {"median_ms": 12.5}
try:
    with telemetry.job("failing"):
        raise ValueError("boom")
except ValueError:
    pass
"""


def test_exports_traces_metrics_and_logs():
    pytest.importorskip("opentelemetry.sdk")
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http")
    _Collector.paths = []
    server = HTTPServer(("127.0.0.1", 0), _Collector)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        env = dict(os.environ, OTEL_EXPORTER_OTLP_ENDPOINT="http://127.0.0.1:%d" % server.server_port,
                   NO_PROXY="127.0.0.1", no_proxy="127.0.0.1", PYTHONPATH=ROOT)
        env.pop("OTEL_SDK_DISABLED", None)
        out = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert "a line for the logs" in out.stdout  # still on the real stdout
    finally:
        server.shutdown()
    assert {"/v1/traces", "/v1/metrics", "/v1/logs"} <= set(_Collector.paths)
