# MedWorld-JEPA Figure 1 — GPT Image 2 master prompt (v3)

Recommended generation settings: `gpt-image-2`, `quality="high"`, `size="2400x1200"`, opaque PNG.

```text
Create one publication-ready scientific methods figure for the main body of a machine-learning paper. The figure presents MedWorld-JEPA, a predictive medical representation learner in which authentic patient time shapes a reusable representation of the present.

COMMUNICATION GOAL
Make one idea immediately clear: information available at or before time t is encoded once into a structured, horizon-independent current patient state; read-only task queries selectively use that state; observed future studies supervise fine-grained latent and clinical-event prediction only through a separate EMA stop-gradient target branch.

CANVAS AND ART DIRECTION
Use a strict 2:1 landscape canvas, designed to remain legible when reduced to full paper text width. Use a warm off-white background (#FAFAF7), generous negative space, crisp thin keylines, and a refined flat 2D vector aesthetic. The visual quality should resemble a carefully art-directed Nature-family methods figure: editorial, restrained, technically precise, and visually memorable. Do not create a title banner or an internal caption. Avoid the look of a software dashboard, slide template, or collection of large UI cards.

Use clean Helvetica/Arial-style sans-serif typography, dark charcoal #303942, with short labels only. Use consistent optical alignment, even spacing, and compact modules. Use mostly open groupings and subtle outlines rather than heavy containers. Use no gradients, shadows, glow, glass, bevels, glossy 3D objects, photographic people, decorative anatomy, or ornamental backgrounds.

COLOR SYSTEM — USE CONSISTENTLY
- Deep blue #3C5488: encoder, global context, and primary online structure.
- Cyan #4DBBD5: multi-scale dense features and anatomy.
- Vermilion #E64B35: pathology and report/finding semantics.
- Salmon #F39B7F: lesion slots and localized supervision.
- Muted violet #8491B4: trajectory and permitted patient history.
- Teal #00A087: future target branch only.
- Pale neutral keylines #CBD3D7; use very light 8–12% tints of the semantic colors for fills.

Encode meaning redundantly through both color and shape. Pathology uses circular clusters; lesion uses outlined square localization slots; trajectory uses a single timeline-shaped token. Do not rely on color alone.

OVERALL COMPOSITION — ONE CONTINUOUS LEFT-TO-RIGHT PIPELINE
Build a single flowing composition with five zones: current evidence at far left, the online encoder, the structured current state as the largest central focal point, selective task queries, and predicted-versus-target future transitions at far right. Keep all connectors short, one-way, non-crossing, and easy to trace.

1. CURRENT EVIDENCE, FAR LEFT, ABOUT 16% OF THE WIDTH
Under the short heading “CURRENT EVIDENCE ≤ t”, show three small, consistent grayscale medical-study pictograms: one chest radiograph, one axial CT slice, and one sagittal MRI slice. These are schematic study thumbnails, not real patient images. Beside or just below them, show a compact report sheet and a short past-to-present timeline ending at t, representing current report, prior studies, and permitted clinical context. Label this small group “HISTORY ≤ t”. Nothing later than t appears in this zone.

2. ONLINE VISUAL AND TEXT ENCODING, ABOUT 18% OF THE WIDTH
Send the current imaging through one compact trainable module labeled “2D / 3D JEPA”. Inside the module, use minimal iconography to show modality-aware 2D patch tokenization and 3D cube tokenization merging into one shared transformer-like visual spine. Add a tiny physical-coordinate axis glyph to communicate spacing, orientation, and physical geometry without explanatory text. The encoder outputs a clearly layered cyan feature pyramid labeled “DENSE FEATURES”.

Process the report through a small parallel pill labeled “TEXT ENCODER”. Let prior-study/history evidence follow a thin violet timeline route. Do not concatenate everything into one generic multimodal token sequence. Do not show an image decoder.

3. STRUCTURED CURRENT STATE, CENTRAL 38%, THE PRIMARY VISUAL FOCUS
Place a compact module labeled “GROUPED STATE TOKENIZER” between the encoder outputs and the state. Then create one open, lightly outlined field labeled “CURRENT PATIENT STATE”, with a small badge reading “NO HORIZON”.

Inside this field, clearly separate one multi-scale dense feature pyramid from five typed state-token groups. Do not call Dense a token group.

- “Dense”: a cyan stack of feature maps, visually separate from the token interface.
- “Global”: exactly one deep-blue circular token, representing shared context.
- “Anatomy”: a cyan set of position-aware tokens arranged around a simple wireframe volume.
- “Pathology”: a vermilion set of circular finding tokens, suggesting identity, location, and severity.
- “Lesion”: a salmon set of exchangeable outlined square slots with small localization marks.
- “Trajectory”: exactly one muted-violet token with a short past-to-present curve ending at t.

The repeated glyphs for Anatomy, Pathology, and Lesion are symbolic sets only; do not imply any specific token count. Global and Trajectory must look like single tokens. The token interface is fixed-size, while the dense pyramid remains spatial and multi-scale.

Show the visibility and role-shaping logic with very short, local routes rather than a table or prose:
- Shared visual features reach all state groups, with the clearest cyan/blue emphasis on Dense, Global, and Anatomy.
- The current report/text route supplements Pathology and also contributes to Trajectory.
- A small region/mask glyph labeled “REGIONS” routes only to Lesion.
- The permitted history timeline routes to Trajectory; it must end at t and must not depict an observed future.
- Add four tiny objective cues near their destinations: “VOLUME JEPA” over Dense + Anatomy, “REPORT ALIGNMENT” over Pathology, “LOCALIZATION” over Lesion, and “PATIENT TIME” over Trajectory plus the forecast path.

These routes bias token roles; do not visually claim perfect disentanglement. Global is shared context and has no invented dedicated supervision objective.

4. READ-ONLY TASK QUERIES, RIGHT OF THE STATE, ABOUT 16% OF THE WIDTH
Use one-way charcoal arrows leaving the current state and fan them into four compact horizontal query rows under the heading “READ-ONLY QUERIES”. No query arrow may return to or modify the state.

Represent selective readout using small colored letter chips after each query, not long text lists:
- “CLS” followed by G, P, τ.
- “SEG” followed by D, A, L.
- “RETRIEVAL” followed by G, P.
- “FORECAST(h)” followed by P, L, τ.

Match every letter chip to its state-group color and shape. Only “FORECAST(h)” receives a small clock/horizon glyph, visibly attached after the state has already been formed. No horizon symbol may appear inside Current Patient State or Trajectory.

5. FINE-GRAINED FORECAST AND TARGET-SIDE TRAINING, FAR RIGHT, ABOUT 28% OF THE WIDTH
Give FORECAST(h) a slightly stronger visual path into a compact three-part latent transition network. Use one central muted-violet transition core marked Fτ, branching to a vermilion pathology head Fp and a salmon lesion head Fℓ. Its output is labeled “PREDICTED TRANSITIONS” and consists of future Pathology, Lesion, and Trajectory token glyphs plus a small event glyph.

Use elegant before-to-after pictograms instead of text lists:
- a pathology cluster appearing, persisting, becoming milder or stronger, and disappearing;
- a localized lesion slot appearing, disappearing, growing, shrinking, or changing shape;
- a violet next-trajectory token with one small curved feedback arrow labeled “ROLLOUT”.

This is latent/token and clinical-event prediction. Never draw a predicted or synthesized future scan.

Above this prediction module, place a narrow, pale-teal dashed enclosure that is visually separate from the online branch. Label it “FUTURE TARGET t+h” and “STOP-GRADIENT”. Inside, show an observed follow-up scan plus an optional report entering one teal module labeled “EMA ENCODER + TOKENIZER”, followed by target Pathology, Lesion, Trajectory, and event glyphs. Label the target token set “EMA TARGET”.

Connect predicted and target token/event sets only at one small comparison node labeled “ALIGN”, using a dashed teal supervision line. The dashed line represents a training loss, not forward data flow. It must terminate at the comparison node and must never point from a future scan, future report, future token, or future label into the online JEPA encoder, grouped tokenizer, current state, or history.

CONNECTOR GRAMMAR
- Solid charcoal or deep-blue arrows: online forward data flow and query readout.
- Short semantic-colored arrows: local visibility or objective-routing cues.
- Dashed teal enclosure and connector: future-only EMA target supervision.
- No bidirectional arrows, no tangled wiring, no long connector passing through another module.

VISIBLE TEXT POLICY
Render only concise labels from the following set, spelled exactly and each shown at most once unless a single-letter token chip must repeat:
“CURRENT EVIDENCE ≤ t”, “HISTORY ≤ t”, “2D / 3D JEPA”, “TEXT ENCODER”, “DENSE FEATURES”, “GROUPED STATE TOKENIZER”, “CURRENT PATIENT STATE”, “NO HORIZON”, “Dense”, “Global”, “Anatomy”, “Pathology”, “Lesion”, “Trajectory”, “REGIONS”, “VOLUME JEPA”, “REPORT ALIGNMENT”, “LOCALIZATION”, “PATIENT TIME”, “READ-ONLY QUERIES”, “CLS”, “SEG”, “RETRIEVAL”, “FORECAST(h)”, “PREDICTED TRANSITIONS”, “ROLLOUT”, “FUTURE TARGET t+h”, “STOP-GRADIENT”, “EMA ENCODER + TOKENIZER”, “EMA TARGET”, “ALIGN”, plus the token/head symbols D, G, A, P, L, τ, Fτ, Fp, and Fℓ.

Do not render the section headings or instructions from this prompt. Add no explanatory sentences, equations, legend, title, subtitle, caption, numerical results, citations, logos, watermarks, or invented labels. Avoid tiny type; all visible text must remain clean and readable at paper scale.

SCIENTIFIC INVARIANTS AND EXCLUSIONS
- All online evidence is available at or before t.
- The current patient state contains no requested horizon and no future observation.
- Trajectory summarizes current and permitted past factors associated with subsequent change; it is not a future token.
- Dense is a multi-scale feature pyramid, followed by five typed state-token groups: Global, Anatomy, Pathology, Lesion, and Trajectory.
- Queries only read the state. Only FORECAST(h) receives h.
- Future observations are target-side only; learned visual and state target encoders are EMA copies and receive no gradient.
- Forecasting predicts future pathology/lesion/trajectory latents and clinical transition events, with optional token-space rollout, never future pixels or voxels.
- Do not invent token counts, feature dimensions, mask ratios, encoder depth, attention heads, loss weights, or any quantitative result.
- Do not include action tokens, observation-token labels, intervention or treatment conditioning, treatment effects, counterfactuals, causal claims, pixel reconstruction, voxel reconstruction, a VAE, diffusion or flow generator, GMM output, or a future-image decoder.
- Do not include legacy or unrelated architecture names or details such as V-JEPA2, Qwen, LoRA, VLA-JEPA, 24 learned queries, D/A/G/M state notation, or an L1-only objective.

FINAL QUALITY CHECK
The result should read in seconds: current evidence → 2D/3D JEPA → dense pyramid + five typed token groups → selective read-only queries; FORECAST(h) predicts token-level clinical transitions and aligns them with a completely isolated EMA future target. Prioritize scientific clarity, graceful spacing, and a distinctive Nature-style editorial composition over decorative complexity.
```
