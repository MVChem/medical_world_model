# MedWorld-JEPA Figure 1 — GPT Image 2 prompt

```text
Create a publication-ready Figure 1 for “MedWorld-JEPA”, a medical world model that uses authentic patient time to shape a reusable representation of the present.

FORMAT AND STYLE
Ultrawide 2:1 landscape, warm off-white background (#FAFAF7), generous whitespace. Strictly flat 2D vector scientific illustration: crisp thin strokes, simple geometric pictograms, precise alignment, no title banner. It should look like a carefully art-directed Nature Methods figure, not a software architecture dashboard. Do not use gradients, glass effects, glow, shadows, 3D spheres, glossy cubes, stock imagery, decorative anatomy, or large text cards.

Use a balanced Nature/NPG palette consistently: deep blue #3C5488; cyan #4DBBD5; teal #00A087 only for the future target branch; coral #E64B35 for pathology/report; salmon-orange #F39B7F for lesion/localization; muted violet #8491B4 for trajectory/patient time; charcoal #303942 for labels and online arrows.

COMPOSITION — ONE CONTINUOUS LEFT-TO-RIGHT STORY

1. CURRENT EVIDENCE, left 22%.
Show three small flat medical-study pictograms—chest radiograph, axial CT, sagittal MRI—under the short label “CURRENT STUDY ≤ t”. A single blue arrow takes only these images into a compact flat grid-to-volume glyph labeled “2D / 3D JEPA”.

2. STRUCTURED CURRENT STATE, central 38%, the largest focal area.
Use one open oval field or subtle rounded outline, not a glass container. Label it “STRUCTURED CURRENT STATE” with a small “NO HORIZON” badge. Arrange six distinct, flat token glyphs as a coherent constellation:
- “Dense”: three cyan feature-map sheets;
- “Global”: one deep-blue disk;
- “Anatomy”: one cyan wireframe volume;
- “Pathology”: a coral cluster;
- “Lesion”: two salmon-orange localized slots;
- “Trajectory”: a violet curved timeline token.

Keep routing short and unmistakable; never run long lines across other modules:
- JEPA/Dense visual features are the shared visual backbone: at the state boundary, one thin pale-cyan spine sends short branches to all six token groups. Give Dense, Global, and Anatomy the clearest blue/cyan emphasis, while lighter cyan branches also reach Pathology, Lesion, and Trajectory.
- Place a small report-sheet icon labeled “Report” immediately beside Pathology; one short coral arrow supplements the shared visual route and points directly into Pathology. It does not touch JEPA.
- Place a small box/mask icon labeled “Regions” immediately beside Lesion; one short salmon-orange supervision arrow supplements the shared visual route and points directly into Lesion. It does not touch JEPA.
- Place a small two-timepoint icon labeled “Patient time” immediately beside Trajectory; one short violet arrow supplements the shared visual route and points directly into Trajectory. It does not touch JEPA.
These three source icons should hug the state perimeter so every colored route is short, separate, and non-crossing.

3. READ-ONLY TASK QUERIES, right 25%.
Fan four one-way charcoal arrows outward from the state to four compact circular icons labeled “CLS”, “SEG”, “RETRIEVAL”, and “FORECAST(h)”. Do not draw return arrows. Use tiny colored dots beside each query to show what it reads:
- CLS: deep blue + coral + violet;
- SEG: cyan + cyan + salmon-orange;
- RETRIEVAL: deep blue + coral;
- FORECAST(h): coral + salmon-orange + violet.
Attach the only clock symbol h to FORECAST(h), after the state. Put CLS, SEG, and RETRIEVAL in a quiet lower-right group. FORECAST(h) follows the upper-right route into simple token-based before/after symbols for pathology change and lesion birth, growth, shrinkage, and disappearance, labeled “CLINICAL TRANSITIONS”. Never draw a generated future scan.

4. TARGET-SIDE TRAINING BRANCH, upper-right background.
Use one pale-teal dashed enclosure, visually separate from the online flow. Inside, show an observed future study and report at t+h entering a small EMA encoder and a compact future token set. Use the heading “EMA TARGET · STOP-GRADIENT” and the label “FUTURE TARGET t+h”. Connect this target token set only to CLINICAL TRANSITIONS with one dashed teal comparison/alignment line ending at a compare symbol. No arrow from any future item may enter the current state or JEPA.

TEXT POLICY
Use only these exact short labels, rendered verbatim and horizontally in a clean sans-serif font: “CURRENT STUDY ≤ t”, “2D / 3D JEPA”, “STRUCTURED CURRENT STATE”, “NO HORIZON”, “Dense”, “Global”, “Anatomy”, “Pathology”, “Lesion”, “Trajectory”, “Report”, “Regions”, “Patient time”, “CLS”, “SEG”, “RETRIEVAL”, “FORECAST(h)”, “CLINICAL TRANSITIONS”, “FUTURE TARGET t+h”, “EMA TARGET · STOP-GRADIENT”. Add no sentences, legends, equations, numbers, citations, logos, watermarks, or invented text.

SCIENTIFIC INVARIANTS
All online evidence is available at or before t. The current state contains no requested horizon and no future observation. Trajectory encodes present/past factors informative about change; it is not a future token. Queries read but never rewrite the state. Only FORECAST(h) receives h. Future scans and reports are EMA stop-gradient targets only. Show latent/token and clinical-event prediction, never pixel/voxel reconstruction, treatment effects, interventions, counterfactuals, or causal inference.
```
