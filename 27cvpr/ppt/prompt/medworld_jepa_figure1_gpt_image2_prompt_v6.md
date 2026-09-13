# MedWorld-JEPA Figure 1 — GPT Image 2 redesign prompt (v6)

Recommended use: for the strongest redesign, paste the prompt below into a fresh generation with no reference image. If you want the rejected draft available only as a scientific cross-check, attach `Codex Image Aug 29, 2026, 05_30_21 AM.png` as Image 1; the conditional instructions below forbid copying its layout. Generate with `gpt-image-2`, high quality, `2560x1152`, opaque PNG.

```text
Create a completely redesigned, publication-ready scientific methods figure for the main body of a CVPR paper.

REFERENCE HANDLING
If Image 1 is attached, it is a rejected visual draft. Use it only to cross-check the scientific components and their relationships. Create a clean new canvas: do not inpaint it, trace it, reuse its pixels, or preserve any silhouette, spatial partition, proportion, connector path, or typographic treatment. This is a structural redesign, not a recoloring and not a minor edit. Rebuild every visual element from scratch. If no Image 1 is attached, simply follow the complete specification below.

Do not preserve or imitate these defects from Image 1: the tall boxed evidence column, the dark navy encoder tower, the stacked beveled scan frames, the large boxed VLM and query dashboard, the rainbow legend, the isolated narrow target card in the corner, the empty upper-right quadrant, the dashed line wrapping around the top and right edges, duplicate or dangling metadata tokens, multiple clock icons, the pause-button-like loss node, or the giant bright-red arc spanning the bottom.

PRIMARY COMMUNICATION GOAL
The figure must communicate one clean story:
evidence available by time t is encoded into a reusable present representation; the representation contains multi-scale dense spatial memory plus exactly 8 generic VLM state slots; read-only queries use this cached state for tasks or horizon-conditioned forecasting; an observed future study supplies only an EMA stop-gradient latent target; future latent prediction shapes the representation of the present.

The visual hierarchy must be obvious in this order:
1. a compact evidence intake;
2. a light 2D/3D medical encoder that forks into dense memory and compact visual tokens;
3. a broad central VLM token sequence ending in exactly 8 generic slots;
4. a read-only task/forecast interface;
5. vertically mirrored predicted and target latent hierarchies at the right.

CANVAS AND SPATIAL BALANCE
Use a 2560 × 1152 landscape canvas, approximately 2.22:1, intended for reduction to full two-column paper width. Use an invisible alignment grid and three horizontal bands. Scientific content should occupy approximately 92% of the width and 80–84% of the height, with controlled and evenly distributed whitespace. Do not leave an entire quadrant empty.

Use this weight-balanced Z composition:
- x = 3–24%: compact two-lane evidence intake followed by the shallow encoder;
- x = 26–55%: resampler, broad VLM sequence, the single dense-memory stack and rail above, and one vertical current-representation brace immediately to the right of the aligned Dₜ¹:ᴸ and Sₜ endpoints;
- x = 57–71%: read-only query interface, with tiny task outputs above it and forecast continuation on the midline;
- x = 73–94%: latent predictor below, compact future-study/EMA intake above, and the two equal-length target and prediction hierarchies extending rightward;
- x = 96–98%: one small comparison node centered between the two latent lanes.

The upper-right must be actively occupied by the wide observed-future target lane. The target lane and prediction lane must have equal visual length and matching latent glyph sizes. No target element may float alone in a distant corner.

Keep the dense-memory rail on the lower boundary of the upper band, beneath both the target lane and the task icons, so it never crosses either. Branch upward only locally to the dense-task mask icon and downward only locally to the predictor.

ART DIRECTION — QUIET NATURE-FAMILY EDITORIAL DESIGN
Use a restrained, low-chroma mineral palette on warm paper. The figure should feel ink-drawn, materially flat, calm, contemporary, and carefully typeset—not like a PowerPoint infographic or software architecture dashboard.

Exact palette:
- warm paper background: #FBFAF6;
- primary ink and online arrows: #2E383B;
- secondary text: #69767A;
- hairlines and dividers: #CDD4D1;
- online medical imaging, dense memory, and visual tokens: storm blue #557487; pale fill #EDF2F4;
- VLM, state slots, current representation, and ordinary task-query accents: dusty plum #82798F; pale fill #F1EEF3;
- forecast, latent predictor, comparison, and future-loss feedback: muted clay #BF755E; pale fill #F7EFEC;
- EMA future target and stop-gradient supervision: sage #6E9182; pale fill #EDF3F0;
- reports and text evidence: neutral gray #8B969A with near-white fill #F3F4F2.

Let background, white space, and neutrals occupy at least 78% of the canvas. Large modules must remain white or use only a 6–10% pale tint. Full-strength colors may appear only on thin outlines, small token glyphs, one short connector, or a compact accent. Never fill a large module with a dark or saturated color. Render every one of the 16 whitelisted labels in primary ink #2E383B; never color-code text.

Use one consistent flat-vector language: medium-fine 2–3 source-pixel strokes that remain at least about 1 pixel after 50% reduction, modest corner radii, precise optical alignment, and no mixed icon styles. Absolutely no gradients, shadows, glow, translucency, glass, bevels, embossing, glossy highlights, metallic surfaces, pseudo-3D extrusion, or airbrushed shading. Do not use royal blue, fluorescent cyan, pure red, bright orange, or traffic-light green.

LEFT SIDE — REBUILD AS A COMPACT TWO-LANE INTAKE
Do not use an enclosing evidence card. Keep the entire evidence-plus-encoder area low and horizontally oriented, approximately 21–23% of the canvas width and no more than 42% of the canvas height.

Place the label “Evidence ≤ t” above a compact open evidence ribbon. In its upper lane, show three small, equal-size, de-identified grayscale medical-study thumbnails on one short horizontal rail: a schematic chest radiograph, an axial computed-tomography slice, and a sagittal magnetic-resonance slice. They are compact modality examples and permitted current/prior studies, not three large stacked photographs. Use thin neutral frames, no shadows, no thick mats, no patient identifiers, and no letters inside the thumbnails.

In the lower lane, show one small report/context pictogram and a short history rail. This lower lane must bypass the image encoder and continue directly toward the VLM as native text evidence. Make this bypass unmistakable. Reports and context do not pass through the medical image encoder or the resampler.

Immediately to the right of the study thumbnails, place a shallow horizontal white module labeled “Medical 2D / 3D encoder”. It must not be a vertical tower. Inside, show only two flat glyphs side by side: one 2D patch sheet and one 3D voxel grid, merging once into a short shared spine. Do not add a funnel, long ladder, serial stack of stages, coordinate axes, or decorative internal pipeline.

The encoder has exactly two output ports:
- an upper port launches a thin storm-blue rail above the main pipeline. On this rail, directly above the eight-slot group, place one and only one small unboxed multi-scale stack labeled “Dense memory  Dₜ¹:ᴸ”. This is the sole Dₜ¹:ᴸ stack and endpoint in the figure; do not draw another stack beside the encoder or at the end of the rail;
- a middle port enters a small white module labeled “Resampler + projector” and emits exactly three representative storm-blue square glyphs followed by one ellipsis. This output continues directly into and becomes the first segment of the single VLM token sequence; never redraw it as a second row.

The dense-memory rail remains visually continuous and outside the VLM bottleneck. It branches only twice later: once to the dense-task mask icon and once to the latent predictor. Keep both branches short.

CENTRAL HERO — ONE TOKEN SEQUENCE, NOT A LARGE CARD
The central VLM must be the broad visual anchor, approximately 24–28% of the canvas width, but it should look like an elegant horizontal sequence rail rather than a large rounded dashboard card. Use a uniform flat #F1EEF3 strip confined tightly to the token rail, or only two #82798F hairlines to define the region. Label it “VLM + LoRA”. Do not append a backbone or checkpoint name.

Show the entire token mechanism once, inside one left-to-right autoregressive sequence:
- exactly three representative storm-blue rounded-square visual glyphs plus one ellipsis—the same glyphs arriving from the resampler, not a duplicate row;
- exactly two representative neutral gray short-pill text glyphs plus one ellipsis, entering directly from the report/context bypass;
- exactly EIGHT dusty-plum state-slot capsules at the end.

Each visual or text evidence token may carry one narrow monochrome inset notch, using only a darker value of that token’s existing role color, to symbolize modality and relative-time embeddings. Introduce no extra metadata colors. The notch must be integrated into the token silhouette; do not hang dice, badges, labels, or miniature tokens below the row.

For the slot group, count outer silhouettes, not internal segments: there must be exactly eight capsule silhouettes, no more and no fewer, with no ellipsis. Make each slot one contiguous capsule silhouette with exactly one subtle internal hairline; use a pale upper region and a filled lower region to suggest learned input slot to returned hidden state without creating two objects. The slots are uniformly styled but are not tied or equal. Give them no anatomy, pathology, lesion, trajectory, global, or other semantic names. Place one clean brace over the group labeled “8 generic slots  Sₜ”.

Treat the single labeled dense-memory stack already placed on the upper rail as the Dₜ¹:ᴸ endpoint; position it directly above the slot-state endpoint Sₜ. Place one vertical open brace immediately to the right of these aligned endpoints so it physically spans Dₜ¹:ᴸ and Sₜ; label it “Rₜ = (Dₜ¹:ᴸ, Sₜ)” directly beside the brace. Place the small badge “task- & horizon-independent” immediately beside this label. Do not duplicate the stack or endpoint, and never place the brace beneath the sequence or in a detached empty lower area.

QUERY INTERFACE — SLIM, OPEN, AND READ-ONLY
Place one slim interface to the right of Sₜ, labeled “Read-only queries  Q”. It receives one short dusty-plum memory arrow only from Sₜ. Raw evidence, compact visual tokens, and Dₜ¹:ᴸ do not enter Q. Draw no return arrow to Sₜ or the VLM.

Use two unboxed rows separated by one hairline:
- upper row labeled “task q → e_q”: one small instruction pill and one plum query token, producing one embedding that fans upward into only two tiny symbol-only task icons—one global/risk icon and one segmentation/grounding mask icon. The mask icon also receives a short branch from the storm-blue dense-memory rail;
- lower row labeled “transition q + e(h) → Aₜ,ₕ”: one instruction pill and one muted-clay transition token. Place exactly ONE clock-shaped e(h) glyph in the entire figure at the shared lower-row input fork, immediately outside or touching the lower-left boundary of Q, never inside Q.

From that single clock junction, one short arrow enters Q beside the instruction/transition token in the forecast row, and one short external arrow goes directly to the latent predictor. Q contains no second clock. Do not redraw a clock inside either destination or anywhere else. Horizon conditioning appears only after Sₜ is formed.

FORECAST AND TARGET — MIRRORED TWO-LANE COMPARISON
Align the forecast row horizontally with one compact pale-clay module labeled “Latent predictor”. Give the predictor four distinct input ports, separated by spacing, port position, and the already assigned role colors only; introduce no additional hues:
1. Dₜ¹:ᴸ from the storm-blue upper rail;
2. Sₜ through a direct short dusty-plum route;
3. Aₜ,ₕ from the query decoder;
4. the same single e(h) clock glyph.

The lower right lane is the prediction lane. The predictor outputs a hierarchy labeled “Prediction  Ŷₜ,ₕ”: one flat global latent glyph followed by a compact group of regional feature grids at multiple schematic scales. Use simple planar tokens and grids, not cubes, jewels, or 3D objects.

Directly above it, spanning the same horizontal width, create the observed-future target lane. Use only two dashed sage hairlines above and below the lane, not a four-sided card. Show “Future study  Xₜ⁺” as one small grayscale scan, followed by a compact EMA encoder/pooling glyph, then a target hierarchy. Place the label “EMA target · stop-gradient” beside the encoder/pooling glyph. A small symbol-only circular-update mark inside that glyph is sufficient; do not draw any EMA parameter connector across the canvas.

The target hierarchy must be a visual twin of the predicted hierarchy below: identical order, size, spacing, and x positions for the global glyph and regional grids. Distinguish it only through sage color, dashed treatment, and the stop-gradient label. The target is not smaller or quieter in geometry; it is equal in length and scale so the comparison is immediately obvious.

Place one tiny, unlabeled pairwise-distance comparison glyph in the far-right column, vertically centered between target and prediction. Route the target downward and the prediction upward into this node with two short symmetric connectors. The comparison glyph must not resemble a pause button, power button, or playback control.

Draw the shortest available shallow, thin muted-clay feedback curve from the comparison node back to the outer midpoint of the vertical Rₜ brace. It may span at most 44% of the canvas width and must stay inside the lower center-right interior within approximately x = 54–98%. After leaving the comparison node, it turns inward immediately and never tracks a canvas edge. Label it “Future loss shapes Rₜ”. Its arrowhead lands only on the outer midpoint of the vertical Rₜ brace. It must not touch evidence, task tokens, task icons, or Q.

VISIBLE TEXT — STRICT 16-LABEL BUDGET
Render only the following 16 labels, each exactly once, spelled exactly, horizontally, and in clean sans-serif type:
“Evidence ≤ t”, “Medical 2D / 3D encoder”, “Dense memory  Dₜ¹:ᴸ”, “Resampler + projector”, “VLM + LoRA”, “8 generic slots  Sₜ”, “Rₜ = (Dₜ¹:ᴸ, Sₜ)”, “task- & horizon-independent”, “Read-only queries  Q”, “task q → e_q”, “transition q + e(h) → Aₜ,ₕ”, “Latent predictor”, “Prediction  Ŷₜ,ₕ”, “Future study  Xₜ⁺”, “EMA target · stop-gradient”, and “Future loss shapes Rₜ”.

Do not render any heading or prose from this prompt. Add no title, caption, panel letters, legend, color key, section header, explanatory sentence, citations, dataset names, numerical results, patient identifiers, logos, watermarks, modality abbreviations, report text, or invented labels. Every thumbnail, icon, token cap, clock, task glyph, ellipsis, arrow, and comparison symbol must remain symbol-only except for the exact whitelisted labels.

SCIENTIFIC INVARIANTS
- Only evidence available at or before t appears online.
- Medical images enter the 2D/3D encoder; report/history/context bypass it and enter the VLM as native text evidence.
- Each permitted image study is compressed into compact resampled visual evidence tokens; the VLM never receives every pixel or voxel token.
- Projected medical visual tokens enter the VLM directly; do not draw a native VLM visual encoder or any second vision encoder.
- Only the current Dₜ¹:ᴸ pyramid is retained as high-resolution dense spatial memory outside the VLM.
- The VLM evidence tokens appear first and exactly 8 generic learned slots appear last.
- The reusable representation is Rₜ = (Dₜ¹:ᴸ, Sₜ), not a pooled vector; it is formed before any task or horizon is specified.
- Q cross-attends only to Sₜ, is read-only, and never writes back.
- Dense tasks combine e_q with Dₜ¹:ᴸ outside Q.
- The single horizon embedding e(h) is introduced only after Sₜ is built and feeds both the forecast row and the latent predictor.
- The predictor receives Dₜ¹:ᴸ, Sₜ, Aₜ,ₕ, and e(h) through four separate ports.
- The observed future study Xₜ⁺ is target-side only. It is encoded by EMA components into stop-gradient global and regional latent targets.
- Prediction and target are matching global-plus-regional latent hierarchies, never reconstructed pixels or voxels.
- The target branch contains no VLM, resampler, patient-state slots, text encoder, or future report.

HARD EXCLUSIONS
No tall card on the left. No dark encoder tower. No saturated full-height block. No large enclosing evidence panel. No stacked glossy scan frames. No detached report icon. No coordinate-axis decoration. No VLM dashboard card. No nested query cards. No duplicate token sequence. No dangling metadata badges. No more or fewer than eight state slots. No multiple clocks. No semantic names on slots. No legend. No rainbow coding. No isolated narrow target card. No empty upper-right quadrant. No perimeter-shaped dashed EMA line. No giant bottom feedback arc. No future-to-online data arrow. No generated future scan. No image decoder, VAE, diffusion, flow, GMM, intervention, treatment, action, rollout, counterfactual, or causal claim.

FINAL COMPOSITION CHECK
The left side must feel light and compact, with a horizontal evidence ribbon and shallow encoder rather than a wall. The VLM sequence and eight slots form the central anchor. The right side must feel fully occupied and balanced: task icons above Q, prediction below, and a wide target lane above that mirrors the predicted latent hierarchy. The eye should move smoothly across the figure without encountering a dead upper-right zone, a heavy dark block, a rainbow legend, or a connector wrapping around the canvas.
```
