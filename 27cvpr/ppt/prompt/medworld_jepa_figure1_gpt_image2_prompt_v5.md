# MedWorld-JEPA Figure 1 — GPT Image 2 master prompt (v5, manuscript-aligned)

Suggested generation setup: `gpt-image-2`, `quality="high"`, `size="2560x1152"`, opaque PNG.

```text
Create one publication-ready scientific methods figure for the main body of a CVPR paper. The figure explains MedWorld-JEPA: observed patient futures are used during pretraining to shape a reusable multimodal representation of the patient at the present time.

COMMUNICATION GOAL
The complete scientific story must be understandable in a few seconds:
evidence available by time t → a medical 2D/3D encoder → a retained multi-scale dense spatial pyramid plus compact visual tokens → a VLM sequence ending in exactly 8 generic learned patient-state slots → a reusable task- and horizon-independent current representation → a read-only task query or a horizon-conditioned transition query. For forecasting, a latent predictor outputs global and regional future embeddings and aligns them with a data-isolated EMA stop-gradient target computed from an observed follow-up scan Xₜ⁺. A single feedback arrow from this loss to the current-representation brace makes the thesis explicit: future loss shapes the present representation.

CANVAS AND ART DIRECTION
Use a 2560 × 1152 landscape canvas, approximately 2.22:1. Design for reduction to the full text width of a two-column paper. Use a warm ivory background (#FAFAF7), generous negative space, a strict alignment grid, thin charcoal keylines, crisp arrowheads, and a refined flat 2D vector aesthetic. The result should feel like a carefully art-directed Nature-family methods figure: restrained, editorial, elegant, technically precise, and visually memorable.

Do not make a title banner, internal caption, software dashboard, slide template, or collection of large cards. Prefer open groupings, small pictograms, light braces, and subtle hairline dividers. Use clean Helvetica/Arial-style sans-serif typography. All labels must remain legible after 50% reduction; no tiny prose.

COLOR SYSTEM — NPG-INSPIRED AND COLOR-BLIND-CONSCIOUS
- Deep blue #3C5488: online medical encoder and primary present-data flow.
- Cyan #4DBBD5: dense spatial features and compact visual tokens.
- Muted violet #8491B4: VLM, generic state slots, and cached patient state.
- Soft orange #F39B7F: task-query path and downstream task heads.
- Vermilion #E64B35: horizon-conditioned transition path, latent predictor, and the single future-loss feedback arrow.
- Teal #00A087: observed future, EMA target branch, stop-gradient targets, and target-to-loss supervision only.
- Charcoal #303942: typography and neutral online connectors.
- Pale gray-blue #D7DEE2: subtle dividers and grouping outlines.
Use very pale 8–12% tints for module fills. Encode roles through both color and shape, never color alone. Use no gradients, shadows, glow, glass, bevels, glossy 3D objects, decorative molecular patterns, or ornamental anatomy.

OVERALL COMPOSITION — ONE EDITORIAL RIVER WITH A FORK AND REJOIN
Use four open visual zones without panel boxes, panel letters, or zone headings: evidence and encoder at left, tokenization and VLM at center-left, reusable state and read-only queries at center-right, and prediction versus target at far right. Run one dominant online pipeline through the visual center. The encoder forks into a slim cyan dense-feature rail above and a compact token rail below; these paths are reunited conceptually by the current-representation brace. The task-query path rises quietly to small output icons, while the transition path continues to the predictor. At far right, predicted and target latent hierarchies mirror each other and meet at one loss node. Keep every connector short, one-way, non-crossing, and easy to trace.

EVIDENCE AND MEDICAL ENCODER — LEFT ZONE
At far left, create a compact open group labeled “EVIDENCE ≤ t”. Show small, de-identified, schematic grayscale study thumbnails: a chest radiograph, an axial CT slice, and a sagittal MRI slice. Use slight stacking to indicate a current study plus permitted prior studies. Add one small report-sheet icon and a short past-to-present timeline ending at t, representing a permitted report, history, and clinical context. No item may appear after t.

Send all permitted studies through one deep-blue module labeled “Medical 2D / 3D encoder”. Inside it, use minimal flat glyphs for modality-specific 2D patching and 3D voxel patching merging into a shared visual spine. Include a tiny symbol-only coordinate-axis glyph for spacing, orientation, and acquisition plane; never depict the slice axis as video time. Do not attempt to show the auxiliary masked-JEPA target path in this figure unless it can be rendered faithfully; omit it in this composition.

Split the encoder output into two clearly different paths:

1. DENSE BYPASS: draw a small cyan stack of progressively coarser feature sheets labeled “Dense spatial memory  Dₜ¹:ᴸ”. This is the retained multi-scale spatial pyramid for the current study. Carry a thin cyan bypass above the rest of the pipeline toward the dense-task icon and the latent predictor. It remains outside the VLM bottleneck.

2. COMPACT TOKEN PATH: feed encoder features from every permitted study into a compact cyan module labeled “Resampler + projector”. Output a short row of cyan rounded-square tokens labeled “visual tokens  Vᵢ”, with an ellipsis showing a fixed compact set rather than every image or voxel token. Attach tiny, symbol-only modality and relative-time badges directly to the visual tokens; these are embeddings attached to evidence, not separate tokens. Turn the permitted report/history/context into a short row of pale neutral gray-blue text-pill tokens with charcoal outlines and the same style of attached time badge, labeled “text evidence”. Do not use vermilion for text evidence. Do not draw a separate native VLM vision encoder and do not send the dense pyramid into the VLM.

TOKENIZATION AND REUSABLE PRESENT STATE — CENTRAL HERO ZONE
Create one elegant muted-violet module labeled “VLM + LoRA · Qwen3.5-9B”. Inside, make the token choreography the visual focus.

Show one horizontal native autoregressive sequence moving left to right:
- first, mixed evidence tokens: cyan square visual tokens, neutral gray-blue text pills, and their small symbol-only time/modality tabs;
- last, exactly EIGHT uniformly styled violet learned slot positions, visibly after all evidence. The eight slot embeddings are separate learned inputs; uniform styling must not imply equal or tied values.

Place one subtle left-to-right causal-attention sweep above the sequence so the slots clearly read preceding evidence. Do not label tokens individually and do not assign any semantic name, icon, color, or clinical role to an individual slot.

Depict the eight positions as exactly eight aligned, two-part vertical capsule columns: a small outlined upper capsule for each learned input slot and a filled lower capsule for the hidden state returned at that same position. There must be eight columns total, not sixteen independent slots. Use one brace over the complete paired group labeled “8 generic slots → Sₜ”. Do not use an ellipsis within the eight-slot group.

Use one thin open brace to unite the cyan dense pyramid Dₜ¹:ᴸ with the violet slot state Sₜ. Label the brace exactly “Rₜ = (Dₜ¹:ᴸ, Sₜ)” and add one small badge “TASK- & HORIZON-INDEPENDENT”. This combined pair is the reusable current representation. Keep it visually central and calm.

READ-ONLY REUSE AND FORECASTING — CENTER-RIGHT ZONE
Place a slim orange module labeled “Read-only queries  Q”. Draw memory arrows into this decoder only from the cached violet Sₜ slots. Do not connect raw evidence, compact visual tokens, or the dense pyramid directly to Q. Use one small symbol-only eye glyph to reinforce read-only access. Draw no arrow back to Sₜ or the VLM.

Inside Q, show two compact token rows rather than prose blocks:

TASK ROW: a neutral instruction pill followed by one orange task-query token. Label the row only “task q → e_q”. Fan the resulting orange embedding into two small, unlabeled output groups: two tiny icons for classification/risk and retrieval; and one segmentation/grounding mask icon that visibly also receives the cyan dense-memory rail. Do not show report generation in this fan; autoregressive answer tokens are outside this compact figure. Convey the global-versus-dense distinction through icons and wiring, not additional text.

FORECAST ROW: after Sₜ has already been constructed, introduce one small clock-shaped token representing the horizon-bin embedding e(h) and one vermilion transition-query token. Label the row only “transition q + e(h) → Aₜ,ₕ”. Fork the same clock-shaped e(h) token: one branch enters the forecast row in Q and a second branch goes directly to the latent predictor. No horizon conditioning may appear in the evidence, VLM sequence, generic slots, Sₜ, or Rₜ. The model never receives the realized follow-up interval.

Route four conceptually distinct inputs into one compact vermilion module labeled “Latent predictor”: the cyan dense memory Dₜ¹:ᴸ, a direct violet route from Sₜ, the vermilion Aₜ,ₕ token, and the same clock-shaped e(h) token used by the forecast row. Use short separated connectors; do not collapse these into one unlabeled arrow.

The predictor outputs a latent hierarchy labeled “Predicted future latents”: one larger global latent glyph plus a fanned set of small regional feature grids at multiple selected scales, with an ellipsis so no exact scale count is implied. Use abstract tokens and feature grids, never a reconstructed scan.

ISOLATED OBSERVED-FUTURE TARGET — FAR-RIGHT ZONE
Next to or just above the predicted hierarchy, create the figure's only dashed container: a compact pale-teal target area. Inside it, show one observed follow-up scan labeled “FUTURE STUDY  Xₜ⁺”. It is a real later observation assigned to horizon bin h; do not label it t+h and do not imply that the realized interval is an online input.

Send Xₜ⁺ into one compact teal encoder and then deterministic pooling, presented together under the single label “EMA TARGET · STOP-GRADIENT”. Its output hierarchy must visually mirror the prediction: one global latent glyph plus a fanned set of regional physical-grid feature maps. Keep the target branch smaller and quieter than the online path.

Show that the target image encoder follows the online image encoder by EMA with one extremely subtle pale-teal dashed parameter-update connector routed along the empty outer margin. Use a small symbol-only circular-update glyph near its endpoint and no text label. This is a parameter update, not a future-data route; it must not pass through or point into any evidence, token, state, query, or predictor node.

The target branch contains no VLM, no resampler, no projector, no patient-state slots, no text encoder, and no future report. The observed future scan has no arrow into any online input or representation module.

Bring the predicted and target hierarchies into one small unlabeled comparison/distance glyph. Use a solid vermilion line from the prediction and a dashed teal supervision line from the target. The two hierarchies must have matching global-plus-regional visual structure.

To express the paper’s central learning mechanism without implying data leakage, draw one clean medium-weight curved vermilion feedback arrow from the comparison node to the Rₜ brace, routed through open lower whitespace. Label it once “FUTURE LOSS SHAPES Rₜ”. This is gradient feedback from the loss, not future-data flow. It must start at the comparison node, end at the reusable current-representation brace, and touch no evidence input, task token, or task head. Do not add a long gradient rail or multiple upward ticks.

TOKEN AND CONNECTOR GRAMMAR
- Cyan rounded squares with tiny tabs: compact per-study visual evidence tokens with modality/time metadata.
- Neutral gray-blue short pills with charcoal outlines: native text/context evidence; small attached badges indicate study time.
- Exactly 8 uniformly styled violet two-part capsule columns: eight generic learned slot positions and their corresponding output hidden states, not sixteen slots.
- One orange instruction pill plus query token: the task-query row.
- One clock token plus one vermilion query token: e(h) and the transition-query row; the same e(h) token also branches directly to the predictor.
- One vermilion capsule: Aₜ,ₕ after read-only transition querying.
- One global glyph plus schematic multi-scale regional grids: both predicted and target future latent hierarchies.
- Solid charcoal/deep-blue arrows: online encoding and forward flow.
- Thin cyan line: retained dense spatial-memory bypass.
- Dashed teal outline/line: future-only EMA target and stop-gradient supervision.
- One curved vermilion feedback arrow from the loss comparison to Rₜ: future-loss gradient shaping the reusable present representation.
- No bidirectional arrows, crossing connectors, tangled wiring, or long line passing through a module.

VISIBLE TEXT POLICY — STRICT 18-LABEL BUDGET
Render only the following 18 concise labels, spelled exactly, horizontally, and each exactly once:
“EVIDENCE ≤ t”, “Medical 2D / 3D encoder”, “Dense spatial memory  Dₜ¹:ᴸ”, “Resampler + projector”, “visual tokens  Vᵢ”, “text evidence”, “VLM + LoRA · Qwen3.5-9B”, “8 generic slots → Sₜ”, “Rₜ = (Dₜ¹:ᴸ, Sₜ)”, “TASK- & HORIZON-INDEPENDENT”, “Read-only queries  Q”, “task q → e_q”, “transition q + e(h) → Aₜ,ₕ”, “Latent predictor”, “Predicted future latents”, “FUTURE STUDY  Xₜ⁺”, “EMA TARGET · STOP-GRADIENT”, and “FUTURE LOSS SHAPES Rₜ”.

Do not render any section heading or explanatory sentence from this prompt. Add no figure title, subtitle, caption, legend, paragraphs, equations beyond the listed compact symbols, token dimensions, token counts other than 8 generic slots, citations, dataset names, numerical results, logos, watermarks, patient identifiers, or invented labels. Every pictogram, thumbnail, attached badge, axis, timeline, query token, clock, task icon, and update glyph must be symbol-only: print no letters, numerals, modality abbreviations, dates, t, Δt, h, x/y/z, report text, or scan annotations inside them. The only visible characters anywhere are those contained in the exact 18 whitelisted labels above.

SCIENTIFIC INVARIANTS — MUST ALL BE PRESERVED
- Online evidence contains only items available at or before cutoff t.
- The online medical encoder supports both 2D radiographs and physically grounded 3D CT/MRI volumes.
- Each permitted study contributes a compact set of resampled visual tokens; the VLM never receives every image or voxel token.
- Only the current study’s Dₜ¹:ᴸ pyramid is retained as high-resolution dense spatial memory.
- The VLM’s native visual front end is bypassed. Medical visual tokens enter the VLM directly after projection.
- The VLM sequence contains evidence first and exactly 8 generic learned slots last. The slots are not clinically named or guaranteed to be interpretable.
- The reusable representation is the pair Rₜ = (Dₜ¹:ᴸ, Sₜ), not a single pooled multimodal vector.
- State construction happens before querying. Sₜ is cacheable, task-independent, and horizon-independent.
- Q cross-attends only to Sₜ and never writes back. Dense heads combine e_q with Dₜ¹:ᴸ outside Q.
- Horizon conditioning is introduced only on the forecast path after Sₜ is built. The same e(h) feeds both the forecast row of Q and P; the model does not receive the realized future interval.
- The latent predictor uses Dₜ¹:ᴸ, Sₜ, Aₜ,ₕ, and h, and predicts matching global and regional embeddings.
- The future observation is Xₜ⁺, target-side only. It is encoded by the EMA image encoder and pooled into stop-gradient global and regional latent targets.
- L_future updates the online encoder, resampler/projector, VLM LoRA and slots, the shared query decoder through its forecast row, the transition token and horizon embedding, and the latent predictor; stop-gradient applies only to the target hierarchy.
- Forecasting predicts latent embeddings, never future pixels or voxels. Clinical transitions are evaluated downstream and are not direct typed transition-token outputs in this core architecture.

HARD EXCLUSIONS
Do not show semantic state groups named Global, Anatomy, Pathology, Lesion, or Trajectory. Do not show a typed state tokenizer, a target-side VLM, target-side patient slots, a future report entering the latent target, direct clinical-event tokens, rollout arrows, action tokens, interventions, treatments, counterfactuals, causal effects, a generated future scan, pixel/voxel reconstruction, an image decoder, VAE, diffusion model, flow model, GMM head, Qwen’s native vision encoder, frozen online modules, 24 query tokens, or an L1-only objective. Do not invent N_v, scale counts, hidden dimensions, mask ratios, LoRA rank, horizon ranges, or quantitative results.

FINAL VISUAL CHECK
At first glance, the eye should follow: evidence ≤ t → medical encoder → dense bypass plus compact tokens → VLM sequence ending in 8 generic slots → reusable Rₜ → read-only task or transition query. At second glance, the viewer should see the forecast row and e(h) enter the latent predictor, while prediction and the isolated EMA stop-gradient target from observed Xₜ⁺ enter one comparison node; a single feedback arrow shows that this future loss shapes Rₜ. Favor scientific clarity, graceful spacing, and token-level storytelling over decorative complexity. If any element feels crowded, remove a decorative icon or increase whitespace; never shrink the type, add another label, or turn the design into a dashboard.
```
