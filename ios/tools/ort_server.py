"""onnxruntime behind a pipe, for ``laya-core-check predict`` (tests/test_ios_core.py).

Lets the Swift ``Predictor`` run the exported graphs on Linux, where the iOS app's Objective-C ONNX Runtime API is not
available: the Swift side writes one request per graph call, this runs it with onnxruntime on CPU and writes the
output back. Framing, all little-endian: request = graph name (1 byte: v, t, h) + int32 count of inputs, then per
input: int32 name length, name, 1 byte dtype (f = float32, i = int64), int32 rank, int64 dims, raw data. Response =
int32 count of outputs, then per output: int32 rank, int64 dims, float32 data (or int32 -1, int32 length and a
UTF-8 error message).

    python ios/tools/ort_server.py <export dir> [suffix]      # suffix: "", "_fp16" or "_q8"
"""
import struct
import sys

import numpy as np
import onnxruntime as ort

OUTPUTS = {"v": ["image_features"], "t": ["last_hidden_state"], "h": ["logits", "act_logits"]}
NAMES = {"v": "vision", "t": "text", "h": "head"}


def read_exact(f, n):
    b = f.read(n)
    if len(b) != n:
        raise EOFError
    return b


def main():
    out_dir, suffix = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "")
    sessions = {k: ort.InferenceSession("%s/%s%s.onnx" % (out_dir, n, suffix), providers=["CPUExecutionProvider"])
                for k, n in NAMES.items()}
    fin, fout = sys.stdin.buffer, sys.stdout.buffer
    while True:
        try:
            g = read_exact(fin, 1).decode()
        except EOFError:
            return
        (n,) = struct.unpack("<i", read_exact(fin, 4))
        feeds = {}
        for _ in range(n):
            (ln,) = struct.unpack("<i", read_exact(fin, 4))
            name = read_exact(fin, ln).decode()
            dt = read_exact(fin, 1).decode()
            (rank,) = struct.unpack("<i", read_exact(fin, 4))
            dims = struct.unpack("<%dq" % rank, read_exact(fin, 8 * rank))
            dtype = np.float32 if dt == "f" else np.int64
            count = int(np.prod(dims)) if rank else 1
            feeds[name] = np.frombuffer(read_exact(fin, count * 4 if dt == "f" else count * 8), dtype=dtype).reshape(dims)
        try:
            outs = sessions[g].run(OUTPUTS[g], feeds)
        except Exception as e:  # report to the Swift side instead of dying
            msg = str(e).encode()
            fout.write(struct.pack("<ii", -1, len(msg)) + msg)
            fout.flush()
            continue
        fout.write(struct.pack("<i", len(outs)))
        for o in outs:
            o = np.ascontiguousarray(o, dtype=np.float32)
            fout.write(struct.pack("<i", o.ndim) + struct.pack("<%dq" % o.ndim, *o.shape) + o.tobytes())
        fout.flush()


if __name__ == "__main__":
    main()
