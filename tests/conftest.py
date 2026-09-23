import os

# laya.telemetry is on by default and would send test runs to the project's collector
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
