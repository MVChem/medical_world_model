# GPT Image 2 Prompt

```text
Use case: scientific-educational
Asset type: 16:9 publication methods figure

Create a publication-ready vector schematic titled “Latent Forecasting of Future Chest X-rays.” Use a white background, two aligned panels, generous whitespace, thin arrows, rounded boxes, and Times New Roman labels. Palette: frozen modules blue-gray, trainable modules orange, target/evaluation green.

(a) TRAINING. Show two current-only paths:
“Current frontal CXR” → “Frozen V-JEPA2” → “current latent S_t”.
“Current frontal CXR” + “Current report” + “Coarse horizon bin” → “Qwen3.5-4B: frozen vision/base; trainable text LoRA + 24 learned queries” → “transition queries q_t”.
Merge S_t and q_t into “Trainable VLA-JEPA predictor” → “predicted future latent Ŝ_{t+h}”. Below, show a dashed green target-only branch: “Observed future CXR” → “same frozen V-JEPA2, separate target call” → “future latent S_{t+h}”. Connect prediction and target with “L1 latent-alignment loss.”

(b) HELD-OUT PATIENT VALIDATION. Compare four routes against the same S_{t+h}: “Correct Query,” “Copy Current State,” “Zero Query,” and “Shuffled Query (different patient).” Copy bypasses the predictor; zero and shuffled queries pass through it. Label the readouts “mean L1 ↓” and “mean cosine distance ↓”; do not invent values. State the desired criterion: Correct Query is lower than Copy, Zero, and Shuffled.

Footer: “Inputs only: current CXR, current report, coarse horizon. Future data are target/evaluation only.” Add “Latent prediction only—no image decoder.” No generated future CXR, action head, future text entering Qwen, decorative anatomy, extra equations, logos, or watermark.
```
