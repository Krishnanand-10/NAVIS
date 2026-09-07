#!/usr/bin/env python3
"""Export the trained speed head for the Android side.

    python scripts/export_model.py

Produces `out/speednet.onnx` and `out/speednet.torchscript.pt`, and verifies
that the exported graph reproduces the PyTorch output. That verification is the
point of this script: an export that silently changes behaviour is worse than
no export, because the discrepancy only surfaces as mysteriously bad navigation
on the phone.

**Getting to TFLite.** Android's NNAPI wants TFLite, and converting needs
TensorFlow, which is not installed here and is not worth adding to this repo
just to run a converter. The friend building the app should run, on a machine
with TensorFlow:

    pip install onnx onnx2tf tensorflow
    onnx2tf -i speednet.onnx -o speednet_tf
    # then tf.lite.TFLiteConverter.from_saved_model("speednet_tf")

Alternatively skip TFLite entirely and use **ONNX Runtime Mobile** or
**PyTorch ExecuTorch**, both of which consume what this script already emits and
avoid the conversion step. For a 394k-parameter model at 10 Hz, any of the
three is fast enough; pick whichever the app developer finds least painful.

**The input contract, which must be honoured exactly on the phone:**

  shape   (1, 200, 6)  float32
  layout  200 samples at 100 Hz = a 2.0 s window, ending at "now"
  channels [ax, ay, az, gx, gy, gz] in the GRAVITY-LEVELLED frame,
           produced the same way as `drnav.features.level_window`:
           low-pass the accelerometer at 0.4 Hz to get the gravity
           direction, build a frame with z along it, rotate both vectors.
  units   m/s^2 and rad/s. Gravity is NOT removed from az.
  output  (speed_m_s, log_variance); sigma = exp(0.5 * log_variance)

Port `level_window` to Kotlin carefully -- it is the single most likely place
for the phone and the pipeline to silently disagree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from drnav.dataset import WINDOW_N                      # noqa: E402
from drnav.features import N_CHANNELS                   # noqa: E402
from drnav.model import SpeedNet                        # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "out"


def main() -> None:
    ckpt_path = OUT / "speednet.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no checkpoint at {ckpt_path} -- run scripts/train_speed.py first")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SpeedNet()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    example = torch.randn(1, WINDOW_N, N_CHANNELS)
    with torch.no_grad():
        ref_speed, ref_logvar = model(example)

    # -- TorchScript -------------------------------------------------------
    ts_path = OUT / "speednet.torchscript.pt"
    traced = torch.jit.trace(model, example)
    traced.save(str(ts_path))
    with torch.no_grad():
        s, lv = torch.jit.load(str(ts_path))(example)
    assert torch.allclose(s, ref_speed, atol=1e-5), "TorchScript output diverged"
    assert torch.allclose(lv, ref_logvar, atol=1e-5), "TorchScript output diverged"
    print(f"torchscript -> {ts_path}  ({ts_path.stat().st_size / 1e6:.2f} MB, verified)")

    # -- ONNX --------------------------------------------------------------
    onnx_path = OUT / "speednet.onnx"
    torch.onnx.export(
        model, (example,), str(onnx_path),
        input_names=["imu_window"],
        output_names=["speed", "log_variance"],
        dynamic_axes={"imu_window": {0: "batch"},
                      "speed": {0: "batch"}, "log_variance": {0: "batch"}},
        # opset 18: the exporter emits 18 natively and downgrading to 17 makes
        # onnx's version converter throw before falling back, which produces an
        # alarming traceback for an export that actually succeeded.
        opset_version=18,
    )
    print(f"onnx        -> {onnx_path}  ({onnx_path.stat().st_size / 1e6:.2f} MB)")

    # PyTorch 2.14's exporter writes weights to a sidecar `.onnx.data` file by
    # default, leaving an 89 KB `.onnx` that looks complete and silently has no
    # weights in it. That is a trap for a handoff over email or git, so the
    # graph is re-saved with everything embedded and the sidecar removed.
    import onnx
    onnx.save_model(onnx.load(str(onnx_path)), str(onnx_path),
                    save_as_external_data=False)
    sidecar = onnx_path.with_suffix(".onnx.data")
    if sidecar.exists():
        sidecar.unlink()
    print(f"             re-saved self-contained: {onnx_path.stat().st_size / 1e6:.2f} MB")

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"imu_window": example.numpy()})
        assert np.allclose(out[0], ref_speed.numpy(), atol=1e-4), "ONNX output diverged"
        print("             ONNX output verified against PyTorch")
    except ImportError:
        print("             (install onnxruntime to verify the ONNX graph here)")

    print(f"\ninput contract: (1, {WINDOW_N}, {N_CHANNELS}) float32, "
          f"gravity-levelled, 100 Hz, 2.0 s window")
    print(f"training metrics: {ckpt.get('metrics', {})}")


if __name__ == "__main__":
    main()
