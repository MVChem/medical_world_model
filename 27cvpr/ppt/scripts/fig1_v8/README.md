# Figure 1: native forecasting with predicted state context

As of 2026-09-15, the user-selected manuscript Figure 1 is the existing
`../../ppt/fig1_v7.pdf`. This v8 variant is retained for reference. Running the
builder below also replaces the manuscript figure, so it is not the refresh
command for the currently selected Figure 1.

`build.py` derives `../../ppt/fig1_v8.pptx` from the archived, editable v7 deck,
preserving the medical thumbnails and the existing two-stage layout. It updates
the state readout to four fusion plus four native-vision slots and adds a native
Qwen forecast decoder with two inputs: true current image/report/EHR/horizon,
and projected predicted future slots.

The predictor retains exactly its current-state and horizon inputs. The future
encoder is a fixed Stage 1 target; its state reaches the latent loss through the
existing stop-gradient connector. Future images never connect to the decoder.
The figure describes the revised protocol, whose formal results remain pending.

Run from the repository root:

```bash
/home/data2/chk/workspace/2026/.venv/bin/python 27cvpr/ppt/scripts/fig1_v8/build.py
```

The script exports editable PPTX, vector PDF/SVG and a 4,200-pixel PNG in
`../../ppt/`, then updates the manuscript's `../../../imgs/fig1.pdf` and PNG.
The historical v7 source and exports remain unchanged. Running an old v7 refresh
would restore its older figure, so use this v8 script for the revised paper.
