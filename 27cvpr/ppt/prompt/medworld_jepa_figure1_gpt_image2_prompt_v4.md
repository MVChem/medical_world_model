# MedWorld-JEPA Figure 1 — GPT Image 2 master prompt (v4)

Suggested generation setup: `gpt-image-2`, high quality, opaque PNG, landscape canvas close to 2.2:1 (for example 2400 × 1100 px).

```text
Create one publication-ready Figure 1 for the main body of a machine-learning paper. The figure explains the complete MedWorld-JEPA pipeline: how longitudinal medical evidence available at or before time t becomes a reusable current-patient representation, how compact visual and text tokens are used by a VLM, how read-only task queries access that representation, and how an observed future scan supplies a stop-gradient latent target during training.

The scientific message must be readable in a few seconds:
evidence available by t → medical 2D/3D encoder → dense spatial memory plus compact visual tokens → VLM with 8 generic patient-state slots → a task- and horizon-independent current representation → read-only task or transition queries; the transition query predicts a multi-scale future latent hierarchy and aligns it with a completely isolated EMA target encoded from the observed follow-up scan.

ART DIRECTION AND CANVAS
Use a wide landscape canvas close to 2.2:1, composed as one continuous left-to-right scientific story rather than a grid of panels. It must remain legible when reduced to the full width of a two-column paper. Use a warm ivory background (#FAFAF7), generous negative space, thin charcoal keylines, crisp arrowheads, careful optical alignment, and a refined flat 2D vector style. Aim for the editorial restraint and color discipline of a Nature-family methods figure: elegant, compact, technically precise, and visually memorable.

Use an NPG-inspired, color-blind-conscious palette consistently:
- deep blue #3C5488 for the online medical encoder and primary current-data flow;
- cyan #4DBBD5 for dense multi-scale features and compact visual tokens;
- muted violet #8491B4 for the VLM and generic patient-state slots;
- coral #E64B35 for the horizon-conditioned transition path and latent predictor;
- soft orange #F39B7F for task queries and task heads;
- teal #00A087 only for the EMA future-target branch and dashed supervision;
- charcoal #303942 for typography and neutral arrows;
- pale gray-blue #D7DEE2 for subtle dividers and grouping outlines.
Use very pale 8–12% tints for module fills. Encode roles with both shape and color. Do not use gradients, shadows, glow, glass, bevels, glossy 3D objects, oversized cards, stock-photo aesthetics, decorative anatomy, or a title banner.

COMPOSITION
Build one dominant horizontal online pipeline through the middle. A slim cyan bypass above it carries dense spatial features. A compact orange task branch sits below the right half. A separate dashed teal future-target branch floats above the far-right prediction module. Keep every connector short, directional, non-crossing, and easy to trace.

1. EVIDENCE AVAILABLE BY t — FAR LEFT, ABOUT 14% OF THE WIDTH
Create a compact open grouping headed “EVIDENCE ≤ t”. Show small schematic grayscale thumbnails for a chest radiograph, an axial CT slice, and a sagittal MRI slice, indicating support for both 2D and 3D studies. Add one smaller prior-study thumbnail behind the current study, a tiny report sheet, and a short past-to-present timeline ending at t. Use the concise labels “current + prior studies” and “report + context”. All items must visibly lie on or before t. Do not use real patient identifiers or long report text.

2. MEDICAL IMAGE ENCODING AND THE TWO VISUAL PATHS — ABOUT 18% OF THE WIDTH
Send the study thumbnails into one blue module labeled “Medical 2D / 3D Encoder”, with a small secondary label “JEPA”. Inside, use minimal flat glyphs for modality-specific 2D patching and 3D voxel patching, plus one tiny physical-coordinate axis to suggest spacing, orientation, and acquisition plane. Do not depict the slice axis as video time.

Split the encoder output clearly into two simultaneous paths:

UPPER BYPASS: draw a layered cyan pyramid with three decreasing-resolution feature sheets labeled “Dense pyramid  Dₜ¹:ᴸ”. This is high-resolution spatial memory retained outside the VLM bottleneck. Continue this thin cyan bypass above the token/VLM path toward dense task heads and the future latent predictor.

MAIN TOKEN PATH: feed the encoder features into a compact cyan module labeled “Resampler + projector”. Its output is a short row of compact cyan rounded-square tokens labeled “visual tokens  Vᵢ”, with an ellipsis to indicate a compact set rather than all image or voxel tokens. Add tiny modality and Δt tabs to this token row. The report and permitted history become a short row of coral text-token glyphs labeled “text + time tokens”. Merge the visual-token and text-token rows only as multimodal evidence entering the VLM. Do not send the full dense pyramid into the VLM.

3. VLM PATIENT-STATE CONSTRUCTION — CENTRAL FOCAL MODULE, ABOUT 24% OF THE WIDTH
Use one elegant violet module labeled “VLM + LoRA”, with a small secondary label “Qwen3.5-9B”. Make the token mechanism visually explicit without using a text block.

Inside the VLM, show a single left-to-right autoregressive sequence. The first segment contains the mixed multimodal evidence tokens: cyan square visual tokens, coral short-pill text tokens, and tiny time/modality tabs. The sequence ends with exactly eight identical violet learned slot tokens. Label this final segment “8 generic slots”. A subtle left-to-right attention sweep should make clear that the slots occur after, and can read, all preceding evidence. Do not show a separate Qwen visual encoder; the native VLM visual front end is bypassed.

Below the sequence, show the eight output hidden states at those same slot positions as eight filled violet capsules labeled “Patient state  Sₜ”. These slots are generic and exchange no manually assigned semantic names. Their specialization emerges from joint training.

Directly above or below this central area, use one thin open brace to unite the cyan dense pyramid and the violet patient state. Label the brace “Reusable current representation  Rₜ” and add a small restrained badge “NO TASK · NO HORIZON”. This combined representation, not a pooled vector, is the reusable interface.

4. READ-ONLY QUERYING — RIGHT OF THE VLM, ABOUT 20% OF THE WIDTH
Place one slim orange module labeled “Read-only query decoder  Q”. Draw cross-attention from the decoder only to the cached violet patient-state slots Sₜ. Do not connect raw evidence or visual tokens directly to this decoder, and draw no return arrow from the decoder to the VLM or patient state.

Inside the decoder, show two compact query rows:
- “task query” with a small instruction-token glyph followed by one orange q-task token;
- “transition query + h” with a small instruction-token glyph, one clock-shaped horizon token h, and one coral q-transition token.

The task-query row produces one compact query-conditioned embedding. Fan it into two minimal output glyphs below:
- “global tasks”: small icons for classification, retrieval, and report generation, reading the patient state only;
- “dense tasks”: a small segmentation-mask icon that visibly combines the query embedding with the cyan dense-pyramid bypass.

The transition-query row produces one distinct coral transition token labeled “Aₜ,ₕ”. The horizon token h appears here, after the patient state has already been formed. No horizon symbol may appear in the evidence, VLM sequence, generic state slots, or Rₜ brace.

5. HORIZON-CONDITIONED LATENT PREDICTION — FAR RIGHT, ABOUT 16% OF THE WIDTH
Route three compact inputs into one coral module labeled “Latent predictor  P”: the cyan dense-pyramid bypass Dₜ¹:ᴸ, the violet patient-state slots Sₜ, and the coral transition token Aₜ,ₕ with h. Use three short, visually separated incoming connectors.

The predictor outputs a matching hierarchy labeled “Predicted future latents”: one larger global latent token above two small regional feature grids at different scales. Use abstract feature maps and tokens, not a reconstructed medical image. The global and regional outputs must visibly preserve both overall patient state and spatial detail.

6. ISOLATED FUTURE-TARGET BRANCH — ABOVE THE PREDICTOR
Create a narrow pale-teal area enclosed by a dashed teal outline. Its heading is “FUTURE TARGET · TRAINING ONLY”. Inside, show one observed follow-up scan labeled “future study  Xₜ⁺”, feeding one teal module labeled “EMA image encoder”. Then show deterministic global and regional pooling as one global token plus two regional feature grids labeled “Target future latents”. Add a small “STOP-GRADIENT” badge directly on this target hierarchy.

The future target branch contains no VLM, no patient-state slots, no tokenizer, and no generated report. A future report may be omitted entirely; it does not enter the online representation or latent target path.

Connect the online medical encoder to the EMA image encoder with one subtle dashed teal arrow labeled “EMA update”. Connect the predicted and target latent hierarchies only at one small comparison node labeled “ALIGN”, using a dashed teal supervision connector. The target connector must end at ALIGN; it must never point from any future item into the current evidence, online encoder inputs, resampler, VLM, patient state, or query decoder.

7. JOINT TRAINING CUE — VERY SMALL FOOTER, OPTIONAL IF SPACE ALLOWS
Along the bottom edge, use one quiet single-line strip, not four large boxes: “Joint training” followed by four compact colored loss chips “L_JEPA”, “L_VLM”, “L_ground”, and “L_future”. Visually associate L_JEPA with the medical encoder, L_VLM with the VLM, L_ground with the dense-task interface, and L_future with ALIGN. This cue must remain secondary to the main pipeline and may be omitted if it compromises legibility.

TOKEN AND CONNECTOR GRAMMAR
- cyan rounded squares = compact visual tokens;
- coral short pills = text/context tokens;
- tiny tabs = modality or time metadata;
- exactly eight identical violet capsules = generic patient-state slots;
- one orange query token = q-task;
- one coral query token plus a clock h = q-transition and requested horizon;
- one global token plus multi-scale regional grids = both predicted and target future latent hierarchies;
- solid charcoal/deep-blue arrows = online forward flow;
- thin cyan bypass = retained dense spatial memory;
- dashed teal arrows and outlines = EMA update or target-only supervision, never online input flow.

VISIBLE TEXT POLICY
Use only the following concise labels, spelled cleanly and horizontally. Do not render the numbered section headings or any prose from this prompt:
“EVIDENCE ≤ t”, “current + prior studies”, “report + context”, “Medical 2D / 3D Encoder”, “JEPA”, “Dense pyramid  Dₜ¹:ᴸ”, “Resampler + projector”, “visual tokens  Vᵢ”, “text + time tokens”, “VLM + LoRA”, “Qwen3.5-9B”, “8 generic slots”, “Patient state  Sₜ”, “Reusable current representation  Rₜ”, “NO TASK · NO HORIZON”, “Read-only query decoder  Q”, “task query”, “transition query + h”, “global tasks”, “dense tasks”, “Aₜ,ₕ”, “Latent predictor  P”, “Predicted future latents”, “FUTURE TARGET · TRAINING ONLY”, “future study  Xₜ⁺”, “EMA image encoder”, “Target future latents”, “STOP-GRADIENT”, “EMA update”, “ALIGN”, and, only if the footer is included, “Joint training”, “L_JEPA”, “L_VLM”, “L_ground”, “L_future”.

Do not add a figure title, subtitle, caption, explanatory sentences, equations beyond the listed compact symbols, legends, token dimensions, citations, dataset names, numerical results, logos, watermarks, or invented labels. Keep all text large enough for paper-scale reading.

SCIENTIFIC INVARIANTS AND HARD EXCLUSIONS
- Online inputs contain only evidence available at or before t.
- The full current representation is the pair of multi-scale dense features and generic VLM patient-state slots: Rₜ = {Dₜ¹:ᴸ, Sₜ}.
- The VLM receives compact resampled visual tokens, not every image or voxel token; the dense feature pyramid remains outside the VLM bottleneck.
- Show exactly 8 generic patient-state slots. Do not name or color individual slots as anatomy, pathology, lesion, trajectory, global, or any other semantic group.
- State construction occurs before querying. The state is reusable, cacheable, task-independent, and horizon-independent.
- The read-only decoder cross-attends only to Sₜ. Its outputs never write back into Sₜ.
- Dense tasks combine a task-conditioned embedding with Dₜ¹:ᴸ. The future predictor uses Dₜ¹:ᴸ, Sₜ, Aₜ,ₕ, and h.
- Only the transition query receives the requested horizon h.
- The observed future scan is target-side only. Its EMA features are pooled into stop-gradient global and regional targets. There is no target-side VLM or target-side patient state.
- Forecasting predicts a latent hierarchy, never future pixels or voxels. Do not draw a synthesized future scan.
- Do not imply causal treatment effects, actions, interventions, counterfactuals, or clinical decision making.
- Do not add a VAE, diffusion model, flow model, GMM head, image decoder, future report generator, typed state tokenizer, grouped semantic state tokens, or bidirectional attention between queries and state.

FINAL QUALITY CHECK
At first glance, the eye should follow one clean story: evidence by t → medical encoder → dense spatial bypass plus resampled visual tokens → VLM evidence sequence ending in 8 generic slots → reusable Rₜ → read-only task or transition query. At second glance, the viewer should see that FORECAST uses h to predict global and regional future latents, while an isolated EMA branch supplies only a dashed stop-gradient target. Favor scientific accuracy, graceful spacing, and a distinctive Nature-style editorial composition over decorative complexity.
```
