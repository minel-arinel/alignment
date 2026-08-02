const pptxgen = require("pptxgenjs");
const path = require("path");

const F = (n) => path.join(__dirname, "figures", n);

const INK = "141821";
const DARK = "12151C";
const ACCENT = "E4572E";      // the oblique cut / the method
const COOL = "2E86AB";        // the old way
const BODY = "33383F";
const MUTED = "6E7681";
const MUTED_D = "A3AAB8";
const RULE = "E6E8EC";

const HEAD = "Cambria";
const TEXT = "Calibri";
const MONO = "Courier New";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";        // 13.3 x 7.5
pres.author = "Elysia Ye";
pres.title = "Oblique registration of functional planes to mapZebrain";

const W = 13.3, H = 7.5;

/* ---------- helpers ---------- */

// Act marker: accent circle + number. The one repeated motif.
function actMark(s, n, label, onDark) {
  s.addShape(pres.ShapeType.ellipse, {
    x: 0.6, y: 0.42, w: 0.42, h: 0.42, fill: { color: ACCENT },
  });
  s.addText(String(n), {
    x: 0.6, y: 0.42, w: 0.42, h: 0.42, align: "center", valign: "middle",
    fontFace: TEXT, fontSize: 14, bold: true, color: "FFFFFF", margin: 0,
  });
  s.addText(label.toUpperCase(), {
    x: 1.16, y: 0.42, w: 8, h: 0.42, valign: "middle", margin: 0,
    fontFace: TEXT, fontSize: 11.5, bold: true, charSpacing: 1.6,
    color: onDark ? MUTED_D : MUTED,
  });
}

function slide(opts) {
  const { act, kicker, title, dark = false, notes } = opts;
  const s = pres.addSlide();
  if (dark) s.background = { color: DARK };
  if (act) actMark(s, act, kicker, dark);
  if (title) {
    s.addText(title, {
      x: 0.6, y: 1.0, w: 12.1, h: 0.95, valign: "top", margin: 0,
      fontFace: HEAD, fontSize: 32, bold: true, color: dark ? "FFFFFF" : INK,
    });
  }
  if (notes) s.addNotes(notes);
  return s;
}

// Bold accent takeaway line, bottom-left. No bar, no rule.
function takeaway(s, txt, dark = false) {
  s.addText(txt, {
    x: 0.6, y: 6.62, w: 12.1, h: 0.5, valign: "middle", margin: 0,
    fontFace: TEXT, fontSize: 15.5, bold: true, color: dark ? "F0A58C" : ACCENT,
  });
}

function caption(s, txt, x, y, w) {
  s.addText(txt, {
    x, y, w, h: 0.32, margin: 0, align: "center",
    fontFace: TEXT, fontSize: 10.5, italic: true, color: MUTED,
  });
}

function fig(s, name, box) {
  s.addImage({ path: F(name), ...box, sizing: { type: "contain", w: box.w, h: box.h } });
}

function bullets(s, items, box, opts = {}) {
  s.addText(
    items.map((t, i) => ({
      text: t,
      options: { bullet: true, breakLine: i !== items.length - 1, paraSpaceAfter: 10 },
    })),
    {
      ...box, margin: 0, valign: "top",
      fontFace: TEXT, fontSize: opts.fontSize || 15, color: BODY, lineSpacing: 22,
    }
  );
}

function card(s, box, tint) {
  s.addShape(pres.ShapeType.roundRect, {
    ...box, rectRadius: 0.06, fill: { color: tint || "F4F5F7" },
    line: { color: RULE, width: 1 },
  });
}

function statCard(s, x, y, w, big, label, color) {
  card(s, { x, y, w, h: 1.85 });
  s.addText(big, {
    x, y: y + 0.18, w, h: 0.85, align: "center", valign: "middle", margin: 0,
    fontFace: HEAD, fontSize: 38, bold: true, color: color || ACCENT,
  });
  s.addText(label, {
    x: x + 0.15, y: y + 1.02, w: w - 0.3, h: 0.68, align: "center", valign: "top", margin: 0,
    fontFace: TEXT, fontSize: 12, color: BODY,
  });
}

function arrow(s, x1, y1, x2, y2, color) {
  s.addShape(pres.ShapeType.line, {
    x: x1, y: y1, w: x2 - x1, h: y2 - y1,
    line: { color: color || MUTED, width: 2, endArrowType: "triangle" },
  });
}

/* =====================================================================
   ACT 0 — TITLE
   ===================================================================== */
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addImage({
    path: F("fig_hero.png"), x: 7.3, y: 0.45, w: 5.3, h: 6.6,
    sizing: { type: "contain", w: 5.3, h: 6.6 },
  });
  s.addText("Oblique registration", {
    x: 0.75, y: 2.15, w: 5.7, h: 0.85, margin: 0,
    fontFace: HEAD, fontSize: 40, bold: true, color: "FFFFFF",
  });
  s.addText("Mapping tilted functional planes\ninto the mapZebrain atlas", {
    x: 0.75, y: 3.05, w: 5.6, h: 1.2, margin: 0, lineSpacing: 30,
    fontFace: HEAD, fontSize: 21, color: "F0A58C",
  });
  s.addText("without pretending the brain is flat", {
    x: 0.75, y: 4.3, w: 5.6, h: 0.4, margin: 0,
    fontFace: TEXT, fontSize: 15, italic: true, color: MUTED_D,
  });
  s.addText("Elysia Ye  ·  larval zebrafish whole-brain imaging", {
    x: 0.75, y: 6.3, w: 5.6, h: 0.4, margin: 0,
    fontFace: TEXT, fontSize: 12.5, color: MUTED_D,
  });
  s.addNotes(
    "This is a plane from fish 32 with the atlas overlaid. It looks decent — hold that thought, " +
    "because 'the overlay looks good' turns out to be exactly what a wrong depth model also produces. " +
    "Today: why our planes don't match any single atlas slice, what everyone does about it, why that's " +
    "not good enough, and the method I've built instead."
  );
}

/* =====================================================================
   ACT I — THE PROBLEM
   ===================================================================== */
{
  const s = slide({
    act: 1, kicker: "The problem", title: "A neuron with no address is not a data point",
    notes:
      "Everything downstream — region-wise functional connectivity, comparing DOI vs eggwater by " +
      "anatomy — needs each ROI to carry an anatomical label. Registration is the step that turns " +
      "pixel coordinates into region identity. If it is wrong, every region-level result inherits " +
      "the error silently: nothing crashes, the numbers just quietly describe the wrong tissue.",
  });

  const steps = [
    ["Raw movie", "per plane, over time"],
    ["Motion correction", "mcorr → STD projection"],
    ["ROI extraction", "CaImAn → traces + COMs"],
    ["Registration", "← we are here"],
    ["Region-wise FC", "DOI vs eggwater"],
  ];
  const bw = 2.14, gap = 0.4;
  steps.forEach(([t, sub], i) => {
    const x = 0.65 + i * (bw + gap);
    const isUs = i === 3;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.35, w: bw, h: 1.75, rectRadius: 0.08,
      fill: { color: isUs ? ACCENT : "F4F5F7" },
      line: { color: isUs ? ACCENT : RULE, width: 1 },
    });
    s.addText(t, {
      x: x + 0.12, y: 2.6, w: bw - 0.24, h: 0.5, align: "center", margin: 0,
      fontFace: TEXT, fontSize: 14.5, bold: true, color: isUs ? "FFFFFF" : INK,
    });
    s.addText(sub, {
      x: x + 0.12, y: 3.1, w: bw - 0.24, h: 0.75, align: "center", valign: "top", margin: 0,
      fontFace: TEXT, fontSize: 11, color: isUs ? "FCE3DA" : MUTED,
    });
    if (i < steps.length - 1) arrow(s, x + bw + 0.06, 3.22, x + bw + gap - 0.06, 3.22, "C3C8D0");
  });

  s.addText(
    "The atlas is what makes one fish comparable to another. Registration is the only place that " +
    "correspondence gets established — and the only place it can be quietly lost.",
    { x: 0.65, y: 4.6, w: 11.9, h: 1.0, margin: 0,
      fontFace: TEXT, fontSize: 18, color: BODY, lineSpacing: 30 }
  );
  s.addText(
    "Nothing crashes when it is wrong. The numbers just quietly describe the wrong tissue.",
    { x: 0.65, y: 5.7, w: 11.9, h: 0.5, margin: 0,
      fontFace: TEXT, fontSize: 16, italic: true, color: MUTED }
  );
  takeaway(s, "Every region-level result rests on this one step.");
}

{
  const s = slide({
    act: 1, kicker: "The problem", title: "The scope is fixed. The fish is not.",
    notes:
      "The imaging plane is set by the objective and the piezo — it is horizontal in scope coordinates " +
      "and it never moves. The fish, mounted in agarose, sits at whatever angle it sits at. Nobody " +
      "mounts a fish perfectly level, and even if you did, the atlas's own axes are defined by a " +
      "different animal. So the plane cuts the brain obliquely. This isn't a mounting failure to be " +
      "fixed — it's the normal condition of every recording we have.",
  });
  fig(s, "fig_std_plane.png", { x: 8.9, y: 1.55, w: 3.8, h: 4.8 });
  bullets(s, [
    "The imaging plane is horizontal in scope coordinates and never tilts.",
    "The fish is mounted in agarose at whatever angle it lands at.",
    "The atlas axes were defined by a different animal entirely.",
    "So the plane enters dorsal tissue at one end and ventral tissue at the other.",
  ], { x: 0.6, y: 2.0, w: 7.9, h: 2.6 }, { fontSize: 16 });

  card(s, { x: 0.6, y: 4.85, w: 7.9, h: 1.35 });
  s.addText(
    "This is not a mounting error to be fixed. It is the normal condition of every recording we have.",
    { x: 0.95, y: 4.85, w: 7.2, h: 1.35, valign: "middle", margin: 0,
      fontFace: TEXT, fontSize: 15, italic: true, color: BODY, lineSpacing: 22 }
  );
  caption(s, "fish 32, plane 0 — STD projection", 8.9, 6.35, 3.8);
  takeaway(s, "The tilt is a property of the experiment, not a defect in it.");
}

{
  const s = slide({
    act: 1, kicker: "The problem", title: "One plane spans 85 µm of atlas depth",
    notes:
      "Here is the actual measurement for fish 32, drawn on a sagittal projection of mapZebrain. " +
      "The orange line is where our acquisition plane really sits. The dashed blue line is what a " +
      "flat atlas z-slice would be. Rostrally the plane is at z=242; caudally it has descended to " +
      "z=327. That's 85 microns of depth inside a single 2D image. For scale, a soma is about 7 " +
      "microns — so the two ends of one plane are separated by roughly twelve cell bodies of depth.",
  });
  fig(s, "fig_sagittal_cut.png", { x: 0.6, y: 1.85, w: 12.1, h: 3.7 });
  s.addText(
    "Sagittal projection of mapZebrain, with the fish-32 acquisition plane drawn through it.",
    { x: 0.7, y: 5.58, w: 11.9, h: 0.35, margin: 0, align: "center",
      fontFace: TEXT, fontSize: 11.5, italic: true, color: MUTED }
  );
  s.addText(
    [
      { text: "85 µm ", options: { fontSize: 22, bold: true, color: ACCENT } },
      { text: "of depth  ≈  ", options: { fontSize: 16, color: BODY } },
      { text: "12 somata ", options: { fontSize: 22, bold: true, color: ACCENT } },
      { text: "stacked, end to end, inside one image", options: { fontSize: 16, color: BODY } },
    ],
    { x: 0.7, y: 5.96, w: 11.9, h: 0.45, align: "center", valign: "middle", margin: 0, fontFace: TEXT }
  );
  takeaway(s, "No single atlas z-slice can be the right target for this plane.");
}

{
  const s = slide({
    act: 1, kicker: "The problem", title: "Pick a slice — either end matches, never both",
    notes:
      "Left and right are the two atlas slices at the ends of that range. Centre is our actual data. " +
      "The z=242 slice has the right forebrain but the wrong hindbrain. The z=327 slice has the right " +
      "hindbrain and almost no forebrain left. There is no slice in between that fixes it, because " +
      "the mismatch isn't a translation — it's that the target should be tilted.",
  });
  const boxes = [
    ["fig_atlas_ztop.png", 0.9, "atlas z = 242", "matches the rostral end", COOL],
    ["fig_experiment.png", 5.05, "our plane", "what we actually recorded", ACCENT],
    ["fig_atlas_zbot.png", 9.2, "atlas z = 327", "matches the caudal end", COOL],
  ];
  boxes.forEach(([f, x, lab, sub, col]) => {
    fig(s, f, { x, y: 2.0, w: 3.2, h: 3.55 });
    s.addText(lab, {
      x, y: 5.62, w: 3.2, h: 0.35, align: "center", margin: 0,
      fontFace: TEXT, fontSize: 14, bold: true, color: col,
    });
    s.addText(sub, {
      x, y: 5.97, w: 3.2, h: 0.35, align: "center", margin: 0,
      fontFace: TEXT, fontSize: 11.5, italic: true, color: MUTED,
    });
  });
  takeaway(s, "The mismatch is not a shift you can translate away — the target itself is tilted.");
}

/* =====================================================================
   ACT II — THE STAIRCASE
   ===================================================================== */
{
  const s = slide({
    act: 2, kicker: "The standard workaround", title: "Chop the plane into bands",
    notes:
      "The standard response, and what I did first: mask the plane into two or three bands, register " +
      "each band to its own atlas slice, and stitch the results. Each band gets a depth that is closer " +
      "to correct than one global slice was. This is a real improvement and it is what the field does.",
  });
  const bands = [
    ["rostral band", "→ atlas z ≈ 250"],
    ["middle band", "→ atlas z ≈ 285"],
    ["caudal band", "→ atlas z ≈ 320"],
  ];
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.85, y: 2.15, w: 3.1, h: 3.9, rectRadius: 0.06,
    fill: { color: "F4F5F7" }, line: { color: RULE, width: 1 },
  });
  s.addText("one functional plane", {
    x: 0.85, y: 2.25, w: 3.1, h: 0.35, align: "center", margin: 0,
    fontFace: TEXT, fontSize: 12, bold: true, color: MUTED,
  });
  bands.forEach((b, i) => {
    const y = 2.7 + i * 1.13;
    s.addShape(pres.ShapeType.roundRect, {
      x: 1.05, y, w: 2.7, h: 0.95, rectRadius: 0.06,
      fill: { color: COOL }, line: { color: COOL, width: 1 },
    });
    s.addText(b[0], {
      x: 1.05, y, w: 2.7, h: 0.95, align: "center", valign: "middle", margin: 0,
      fontFace: TEXT, fontSize: 13.5, bold: true, color: "FFFFFF",
    });
    arrow(s, 4.15, y + 0.47, 6.35, y + 0.47, COOL);
    s.addShape(pres.ShapeType.roundRect, {
      x: 6.55, y, w: 3.5, h: 0.95, rectRadius: 0.06,
      fill: { color: "FFFFFF" }, line: { color: COOL, width: 1.5 },
    });
    s.addText(b[1].replace("→ ", ""), {
      x: 6.55, y, w: 3.5, h: 0.95, align: "center", valign: "middle", margin: 0,
      fontFace: TEXT, fontSize: 13.5, bold: true, color: COOL,
    });
  });
  s.addText("each band registered\nindependently to its own\nflat atlas slice, then\nstitched back together", {
    x: 10.35, y: 3.1, w: 2.5, h: 2.0, valign: "top", margin: 0, lineSpacing: 21,
    fontFace: TEXT, fontSize: 13.5, color: BODY,
  });
  takeaway(s, "Sensible, widely used — and genuinely better than one flat slice.");
}

{
  const s = slide({
    act: 2, kicker: "The standard workaround", title: "And it does help",
    notes:
      "Be fair to the method. Masked registration visibly sharpens the fit — here the masked patch " +
      "recovers the top of the plane that a full-frame fit was smearing. I'm not about to tell you " +
      "this doesn't work. I'm going to tell you what it costs.",
  });
  [["fig_piece_mask.png", "top band, masked", 0.8],
   ["fig_piece_composite.png", "composite of both bands", 3.65],
   ["fig_piece_merge.png", "merge (R = atlas, G = data)", 6.5],
  ].forEach(([f, lab, x]) => {
    s.addText(lab, { x: x - 0.15, y: 2.0, w: 2.9, h: 0.32, margin: 0, align: "center",
      fontFace: TEXT, fontSize: 12.5, bold: true, color: MUTED });
    fig(s, f, { x, y: 2.4, w: 2.6, h: 3.6 });
  });
  bullets(s, [
    "Each band is closer to its true depth than one global slice was.",
    "Edges sharpen; the smearing from averaging across depth drops.",
    "It is cheap: same registration call, run more times.",
  ], { x: 9.55, y: 2.4, w: 3.2, h: 3.4 }, { fontSize: 14 });
  caption(s, "plane 2 — masked patch registration, fish 32", 0.8, 6.15, 8.3);
  takeaway(s, "The question isn't whether masking helps. It's what the leftover error looks like.");
}

{
  const s = slide({
    act: 2, kicker: "The standard workaround", title: "But the depth model is a staircase",
    notes:
      "This is the whole argument on one plot. Orange dashed is the true depth of the plane as you " +
      "walk along it — a straight ramp. Blue is what a three-band mask actually gives you: " +
      "piecewise constant. You are approximating a continuous ramp with a step function. " +
      "Bottom panel is the residual. Three bands leaves you 19 microns of depth error at its worst, " +
      "and crucially the error is worst in the MIDDLE of each band — where the tissue is, not at the " +
      "edges where you were looking.",
  });
  fig(s, "fig_staircase.png", { x: 1.6, y: 1.95, w: 10.1, h: 4.5 });
  takeaway(s, "The error is largest in the middle of each band — where the neurons are.");
}

{
  const s = slide({
    act: 2, kicker: "The standard workaround", title: "What the staircase costs",
    notes:
      "Three numbers. Three bands leaves 19 microns of residual depth error. To push that under one " +
      "soma you'd need about nine bands. But bands aren't free — each one is an independent " +
      "registration, so the in-plane pose is free to drift between neighbours, and every boundary is " +
      "a seam where two solutions meet. More bands buys depth accuracy with pose instability and more " +
      "seams. It does not converge on the truth; it trades one error for another.",
  });
  statCard(s, 0.75, 2.2, 3.75, "19 µm", "worst-case depth error with 3 bands — nearly 3 cell bodies", ACCENT);
  statCard(s, 4.78, 2.2, 3.75, "9 bands", "needed to push that error under a single soma", ACCENT);
  statCard(s, 8.8, 2.2, 3.75, "8 seams", "…and each one is an independent registration", ACCENT);

  bullets(s, [
    "Bands are registered independently, so in-plane pose drifts between neighbours — the stitched plane is no longer one rigid object.",
    "At a seam, two neurons a few microns apart get depths that differ by a whole step. That discontinuity is an artefact you introduced.",
    "More bands is not convergence: you buy depth resolution with pose instability and more seams.",
  ], { x: 0.75, y: 4.45, w: 11.8, h: 2.0 }, { fontSize: 14.5 });
  takeaway(s, "Refining the staircase does not approach the ramp. It trades one error for another.");
}

/* =====================================================================
   ACT III — THE METHOD
   ===================================================================== */
{
  const s = slide({
    act: 3, kicker: "The oblique method", title: "Don't approximate the ramp — model it",
    dark: false,
    notes:
      "Same axes as the last plot. The staircase goes away. If the depth of the plane is a smooth " +
      "linear function of position, then write down that function and use it. That's the entire idea. " +
      "Everything from here is the bookkeeping needed to make it work.",
  });
  fig(s, "fig_ramp.png", { x: 1.7, y: 2.0, w: 9.9, h: 3.7 });
  s.addText(
    "The depth of the plane is a smooth linear function of position. So write that function down and use it.",
    { x: 0.9, y: 5.85, w: 11.5, h: 0.6, align: "center", valign: "middle", margin: 0,
      fontFace: TEXT, fontSize: 16, color: BODY }
  );
  takeaway(s, "One continuous depth model instead of N independent registrations.");
}

{
  const s = slide({
    act: 3, kicker: "The oblique method", title: "Two numbers define the cut",
    notes:
      "The only manual step in the whole pipeline. Look at the atlas at the shallow end of your plane " +
      "and note the y-row where tissue starts. Look at the deep end and note the y-row where it ends. " +
      "That's it — two (y, z) anchor pairs. Everything downstream is derived. Note these are read off " +
      "the ATLAS, not off your data, which is why they're stable across all the planes of a fish.",
  });
  fig(s, "fig_band_pick.png", { x: 0.6, y: 1.85, w: 8.7, h: 4.6 });
  s.addText("The two anchors", {
    x: 9.5, y: 2.0, w: 3.2, h: 0.4, margin: 0,
    fontFace: HEAD, fontSize: 19, bold: true, color: INK,
  });
  card(s, { x: 9.5, y: 2.55, w: 3.2, h: 1.0 });
  s.addText("y = 250 → z = 242", {
    x: 9.5, y: 2.55, w: 3.2, h: 1.0, align: "center", valign: "middle", margin: 0,
    fontFace: MONO, fontSize: 14, bold: true, color: ACCENT,
  });
  card(s, { x: 9.5, y: 3.7, w: 3.2, h: 1.0 });
  s.addText("y = 850 → z = 327", {
    x: 9.5, y: 3.7, w: 3.2, h: 1.0, align: "center", valign: "middle", margin: 0,
    fontFace: MONO, fontSize: 14, bold: true, color: ACCENT,
  });
  s.addText(
    "Read off the atlas, not off your data — which is why the same two numbers serve every plane in the fish.",
    { x: 9.5, y: 4.9, w: 3.2, h: 1.5, valign: "top", margin: 0, lineSpacing: 21,
      fontFace: TEXT, fontSize: 13.5, color: BODY }
  );
  takeaway(s, "This is the only place a human is in the loop — and everything inherits its error.");
}

{
  const s = slide({
    act: 3, kicker: "The oblique method", title: "Depth is a function of atlas y",
    dark: true,
    notes:
      "THE slide. Every pixel in the functional plane gets its own atlas depth, interpolated linearly " +
      "between the two anchors. Continuous, not stepped. Three things people get wrong. One: y is the " +
      "ATLAS y — obtained after the affine — not the row index in your image. If you index depth by " +
      "image row, the tilt rotates whenever the affine changes and you get a different plane every " +
      "time you register. Two: the anchors describe where the cut sits in the atlas, which is what " +
      "makes them reusable across planes. Three: the line does not stop at the anchors — t is " +
      "unclamped, so the cut stays one flat tilted plane through the whole volume. Clamping would " +
      "give you a bent surface, which is physically wrong.",
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.75, y: 2.05, w: 11.8, h: 1.25, rectRadius: 0.06,
    fill: { color: "1D2230" }, line: { color: "39414F", width: 1 },
  });
  s.addText(
    [
      { text: "z", options: { color: ACCENT, bold: true } },
      { text: "  =  z_top  +  ", options: { color: "FFFFFF" } },
      { text: "( y − y_top ) / ( y_bot − y_top )", options: { color: "F0A58C" } },
      { text: "  ×  ( z_bot − z_top )", options: { color: "FFFFFF" } },
    ],
    { x: 0.75, y: 2.05, w: 11.8, h: 1.25, align: "center", valign: "middle", margin: 0,
      fontFace: MONO, fontSize: 21, bold: true }
  );
  fig(s, "fig_hinge_anchors_dark.png", { x: 0.75, y: 3.5, w: 7.4, h: 2.9 });

  const pts = [
    ["y is the atlas y", "obtained after the affine — not your image row index"],
    ["Anchors live in atlas space", "so the same pair serves every plane in the fish"],
    ["The line does not stop", "unclamped: one flat tilted plane through the volume"],
  ];
  pts.forEach(([h, b], i) => {
    const y = 3.55 + i * 0.98;
    s.addShape(pres.ShapeType.ellipse, {
      x: 8.5, y: y + 0.06, w: 0.2, h: 0.2, fill: { color: ACCENT },
    });
    s.addText(h, {
      x: 8.85, y, w: 3.9, h: 0.32, margin: 0,
      fontFace: TEXT, fontSize: 14, bold: true, color: "FFFFFF",
    });
    s.addText(b, {
      x: 8.85, y: y + 0.32, w: 3.9, h: 0.6, margin: 0, valign: "top", lineSpacing: 17,
      fontFace: TEXT, fontSize: 12, color: MUTED_D,
    });
  });
  takeaway(s, "Every pixel gets its own depth. Continuously.", true);
}

{
  const s = slide({
    act: 3, kicker: "The oblique method", title: "The affine and the anchors do different jobs",
    notes:
      "This is the part that surprises people, so I want to be explicit. We do run a 3D affine " +
      "registration — but we use only its in-plane output. The affine tells us where the plane sits " +
      "in x and y. Its z output is computed and then thrown away, replaced by the anchor rule. " +
      "That separation is deliberate: pose and depth are solved by different mechanisms, so they " +
      "can't trade off against each other. An optimiser that can move in z will happily buy metric " +
      "improvement by sliding to the wrong depth. This one can't.",
  });
  const cols = [
    ["3D affine registration", "solves in-plane pose", ["x, y position", "rotation in plane", "scale"],
      "Mattes MI, 500 iterations, against the atlas z-slab", COOL],
    ["The two anchors", "solve depth", ["atlas z per pixel", "the tilt", "shared by all planes"],
      "y → z, linear, unclamped", ACCENT],
  ];
  const CW = 3.85, CX = [0.75, 4.75, 8.75];
  cols.forEach(([t, role, items, foot, col], i) => {
    const x = CX[i];
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.15, w: CW, h: 3.35, rectRadius: 0.08,
      fill: { color: "FFFFFF" }, line: { color: col, width: 2 },
    });
    s.addText(t, { x: x + 0.28, y: 2.38, w: CW - 0.55, h: 0.4, margin: 0,
      fontFace: HEAD, fontSize: 17, bold: true, color: col });
    s.addText(role, { x: x + 0.28, y: 2.8, w: CW - 0.55, h: 0.35, margin: 0,
      fontFace: TEXT, fontSize: 13, italic: true, color: MUTED });
    bullets(s, items, { x: x + 0.28, y: 3.28, w: CW - 0.55, h: 1.5 }, { fontSize: 13.5 });
    s.addText(foot, { x: x + 0.28, y: 4.72, w: CW - 0.55, h: 0.65, margin: 0, valign: "top", lineSpacing: 16,
      fontFace: TEXT, fontSize: 11.5, color: MUTED });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: CX[2], y: 2.15, w: CW, h: 3.35, rectRadius: 0.08,
    fill: { color: "FBEAE4" }, line: { color: ACCENT, width: 1 },
  });
  s.addText("The affine's z output is discarded", {
    x: CX[2] + 0.28, y: 2.38, w: CW - 0.55, h: 0.85, margin: 0, lineSpacing: 24,
    fontFace: HEAD, fontSize: 17, bold: true, color: ACCENT,
  });
  s.addText(
    "This is the single line that separates the method from sitk.Resample, which would use that z " +
    "and hand you one depth for the whole plane.",
    { x: CX[2] + 0.28, y: 3.28, w: CW - 0.55, h: 1.5, margin: 0, valign: "top", lineSpacing: 19,
      fontFace: TEXT, fontSize: 13, color: BODY }
  );
  takeaway(s, "Pose and depth are solved separately, so they cannot trade off against each other.");
}

{
  const s = slide({
    act: 3, kicker: "The oblique method", title: "Build a synthetic atlas image that matches your geometry",
    notes:
      "Mechanically: take the grid of pixels in your functional plane. Push each one through the " +
      "affine to get atlas x and y. Use the anchor rule to get that pixel's atlas z. Sample the atlas " +
      "volume there, trilinearly. The result is a 2D image — an atlas view that has never existed as " +
      "a slice, cut along exactly the surface your microscope cut. Now the problem is 2D-to-2D and " +
      "ordinary Elastix can finish the job.",
  });
  const steps = [
    ["1", "Take every pixel", "(col, row) on the functional grid"],
    ["2", "Push through the affine", "→ atlas x, atlas y"],
    ["3", "Apply the anchor rule", "atlas y → atlas z, per pixel"],
    ["4", "Sample the volume", "trilinear, at (x, y, z)"],
  ];
  steps.forEach(([n, t, b], i) => {
    const y = 2.15 + i * 1.05;
    s.addShape(pres.ShapeType.ellipse, {
      x: 0.75, y: y + 0.14, w: 0.55, h: 0.55, fill: { color: ACCENT },
    });
    s.addText(n, { x: 0.75, y: y + 0.14, w: 0.55, h: 0.55, align: "center", valign: "middle",
      margin: 0, fontFace: TEXT, fontSize: 16, bold: true, color: "FFFFFF" });
    s.addText(t, { x: 1.55, y: y + 0.08, w: 4.3, h: 0.38, margin: 0,
      fontFace: TEXT, fontSize: 15.5, bold: true, color: INK });
    s.addText(b, { x: 1.55, y: y + 0.46, w: 4.6, h: 0.35, margin: 0,
      fontFace: TEXT, fontSize: 12.5, color: MUTED });
  });

  s.addShape(pres.ShapeType.roundRect, {
    x: 6.7, y: 2.3, w: 5.85, h: 3.55, rectRadius: 0.08,
    fill: { color: "F4F5F7" }, line: { color: RULE, width: 1 },
  });
  s.addText("The result", { x: 7.05, y: 2.55, w: 5.15, h: 0.4, margin: 0,
    fontFace: HEAD, fontSize: 18, bold: true, color: INK });
  s.addText(
    "A 2D atlas image that has never existed as a slice — cut along exactly the surface your " +
    "microscope cut.\n\nThe problem is now 2D-to-2D, and ordinary Elastix (affine + B-spline) " +
    "finishes the job on residual deformation only.",
    { x: 7.05, y: 3.05, w: 5.15, h: 2.5, margin: 0, valign: "top", lineSpacing: 23,
      fontFace: TEXT, fontSize: 14.5, color: BODY }
  );
  takeaway(s, "The geometry is solved before the deformable step ever runs.");
}

{
  const s = slide({
    act: 3, kicker: "The oblique method", title: "Oblique reslice vs. what SimpleITK would give you",
    notes:
      "Proof it does something. Same affine, same atlas, same fish. Left is the oblique reslice. " +
      "Middle is what a standard axis-aligned resample produces — essentially one atlas z for the " +
      "whole frame. Right is our data. Look at the forebrain and the hindbrain simultaneously: the " +
      "oblique version has both in register, the axis-aligned version can only ever have one.",
  });
  [["fig_cut_oblique.png", "oblique reslice", "z varies with atlas y", ACCENT, 1.6],
   ["fig_cut_axis.png", "axis-aligned resample", "one z ≈ 246 for the whole frame", COOL, 5.35],
   ["fig_experiment.png", "our data", "what we actually recorded", MUTED, 9.1],
  ].forEach(([f, lab, sub, col, x]) => {
    s.addText(lab, { x, y: 1.95, w: 2.7, h: 0.32, margin: 0, align: "center",
      fontFace: TEXT, fontSize: 14, bold: true, color: col });
    fig(s, f, { x, y: 2.35, w: 2.7, h: 3.5 });
    s.addText(sub, { x: x - 0.25, y: 5.92, w: 3.2, h: 0.32, margin: 0, align: "center",
      fontFace: TEXT, fontSize: 12, italic: true, color: MUTED });
  });
  s.addText(
    "Same affine, same atlas, same fish — the only difference is where z comes from.",
    { x: 0.9, y: 6.28, w: 11.5, h: 0.35, align: "center", margin: 0,
      fontFace: TEXT, fontSize: 12.5, italic: true, color: MUTED }
  );
  takeaway(s, "Forebrain and hindbrain in register at the same time — the axis-aligned cut can't do that.");
}

{
  const s = slide({
    act: 3, kicker: "The oblique method", title: "One affine for the whole stack, shifted in z",
    notes:
      "The planes are acquired by stepping the piezo along the imaging axis, so they are parallel by " +
      "construction: same in-plane pose, same tilt, different depth. So we register once and shift. " +
      "The arithmetic is clean — functional z spacing is 20 microns, mapZebrain is 1 micron isotropic, " +
      "so one plane step is exactly 20 atlas voxels. Re-registering each plane independently would let " +
      "the tilt drift plane to plane and produce a stack that no physical microscope could have " +
      "acquired. Only the sign is empirical: which end of the stack is deeper.",
  });
  fig(s, "fig_plane_stack.png", { x: 0.6, y: 1.9, w: 7.9, h: 4.4 });

  bullets(s, [
    "Stepped by the piezo along one axis — parallel by construction.",
    "Same in-plane pose, same tilt, different depth.",
    "So: register once, shift N times.",
    "Independent per-plane fits would let the tilt drift — a stack no microscope could have produced.",
  ], { x: 8.8, y: 2.0, w: 3.9, h: 2.7 }, { fontSize: 14 });

  card(s, { x: 8.8, y: 4.75, w: 3.9, h: 1.5 });
  s.addText("The arithmetic is exact", {
    x: 9.05, y: 4.92, w: 3.4, h: 0.32, margin: 0,
    fontFace: TEXT, fontSize: 12.5, bold: true, color: INK,
  });
  s.addText("20 µm / plane\n÷ 1 µm / voxel\n= 20 voxels", {
    x: 9.05, y: 5.28, w: 3.4, h: 0.85, margin: 0, lineSpacing: 15,
    fontFace: MONO, fontSize: 11.5, bold: true, color: ACCENT,
  });
  takeaway(s, "11 planes, one registration.");
}

/* =====================================================================
   ACT IV — VALIDATION
   ===================================================================== */
{
  const s = slide({
    act: 4, kicker: "Does it work", title: "Residual deformation is small",
    notes:
      "Left: experiment. Middle: the atlas after the deformable step. Right: merged, red atlas green " +
      "data. The point isn't that the merge looks yellow — it's that the B-spline barely had to do " +
      "anything, because the geometry was already right when it started. When the deformable step is " +
      "working hard, that's a sign the depth model is wrong and the warp is compensating for it.",
  });
  fig(s, "fig_qc_merge.png", { x: 0.6, y: 2.0, w: 7.5, h: 3.9 });
  caption(s, "experiment  |  atlas warped to it  |  merge (R = atlas, G = data)", 0.6, 6.05, 7.5);
  [["fig_pre_elastix.png", "before deformable", 8.5], ["fig_post_elastix.png", "after deformable", 10.7]].forEach(([f, lab, x]) => {
    s.addText(lab, { x: x - 0.15, y: 2.02, w: 2.3, h: 0.3, margin: 0, align: "center",
      fontFace: TEXT, fontSize: 12, italic: true, color: MUTED });
    fig(s, f, { x, w: 2.0, y: 2.4, h: 3.1 });
  });
  s.addText(
    "Before the deformable step vs. after — the correction is small because the geometry was " +
    "already right when Elastix started.",
    { x: 8.35, y: 5.62, w: 4.5, h: 0.9, margin: 0, valign: "top", lineSpacing: 19,
      fontFace: TEXT, fontSize: 13, color: BODY }
  );
  takeaway(s, "A hard-working B-spline is a symptom, not a success.");
}

{
  const s = slide({
    act: 4, kicker: "Does it work", title: "Consistent across the whole stack",
    notes:
      "All 11 planes of fish 32. One affine, one pair of anchors, z stepped by 20 voxels per plane. " +
      "Nothing here is tuned per plane. The deep planes get sparse because there's genuinely less " +
      "tissue there, not because the registration degraded.",
  });
  fig(s, "fig_all_planes.png", { x: 0.65, y: 2.0, w: 8.7, h: 3.73 });
  caption(s, "rows: experiment  |  oblique atlas reslice  |  overlay      columns: plane 0 (dorsal) → plane 10 (ventral)",
          0.65, 5.85, 8.7);
  bullets(s, [
    "All 11 planes, fish 32.",
    "One affine. One pair of anchors.",
    "z stepped by 20 voxels per plane.",
    "Nothing tuned per plane.",
  ], { x: 9.8, y: 2.15, w: 2.9, h: 2.5 }, { fontSize: 15 });
  s.addText(
    "Deep planes thin out because there is less tissue there — not because the registration degraded.",
    { x: 9.8, y: 4.8, w: 2.9, h: 1.5, margin: 0, valign: "top", lineSpacing: 22,
      fontFace: TEXT, fontSize: 14, color: BODY }
  );
  takeaway(s, "Same parameters top to bottom.");
}

{
  const s = slide({
    act: 4, kicker: "Does it work", title: "The check that isn't just an overlay",
    notes:
      "This is the slide that matters. Everything before this was 'the overlay looks good' — and I " +
      "want to be blunt: a wrong depth model plus a flexible enough B-spline ALSO produces an overlay " +
      "that looks good. It will warp the atlas to match intensity while sampling the wrong tissue, " +
      "and you will never see it in a merge. So: independent check. Take compact regions with " +
      "unambiguous centres — habenula, epiphysis — compute the centre of mass of the atlas mask, and " +
      "compare it to the centre of mass of the ROIs we mapped there. Those agree to within two or " +
      "three voxels, which at 1 micron isotropic is sub-somatic. That is a measurement, not an " +
      "impression.",
  });
  s.addShape(pres.ShapeType.roundRect, {
    x: 0.75, y: 1.9, w: 6.2, h: 1.2, rectRadius: 0.08,
    fill: { color: "FBEAE4" }, line: { color: ACCENT, width: 1.5 },
  });
  s.addText(
    "A wrong depth model plus a flexible B-spline also produces an overlay that looks good.",
    { x: 1.05, y: 1.9, w: 5.6, h: 1.2, valign: "middle", margin: 0, lineSpacing: 22,
      fontFace: TEXT, fontSize: 15, bold: true, color: "8C2F12" }
  );
  bullets(s, [
    "Take compact regions with unambiguous centres — habenula, epiphysis.",
    "Compute the centre of mass of the atlas region mask.",
    "Compare against the centre of mass of the ROIs we mapped into it.",
  ], { x: 0.75, y: 3.35, w: 6.2, h: 1.9 }, { fontSize: 14.5 });

  statCard(s, 0.75, 4.55, 6.2, "2–3 voxels", "agreement between mask CoM and mapped-ROI CoM, at 1 µm isotropic — sub-somatic", ACCENT);
  fig(s, "fig_rois_crop.png", { x: 7.45, y: 2.35, w: 5.2, h: 3.5 });
  s.addText("plane 0 — 1616 ROIs, coloured by the mapZebrain region they landed in", {
    x: 7.45, y: 1.95, w: 5.2, h: 0.32, margin: 0, align: "center",
    fontFace: TEXT, fontSize: 12, bold: true, color: MUTED });
  caption(s, "experiment  |  oblique atlas reslice", 7.45, 5.95, 5.2);
  takeaway(s, "That is a measurement, not an impression.");
}

{
  const s = slide({
    act: 4, kicker: "Does it work", title: "Masks didn't disappear — they got demoted",
    notes:
      "Honesty slide. I still use masked, piecewise registration. But it is no longer carrying the " +
      "depth model — it handles residual local deformation, mostly where the agarose or the eye " +
      "distorts one corner of the frame. And because the depth is already right everywhere, the seam " +
      "is a small intensity discontinuity rather than a jump of tens of microns in anatomy. I erode " +
      "the mask by an overlap band and discard that band when compositing, rather than registering " +
      "the sharp complement, which is what produces the visible seam.",
  });
  const then = ["carried the entire depth model", "seam = a jump of tens of µm in anatomy", "more bands → more pose drift"];
  const now = ["handles residual local deformation only", "seam = a small intensity discontinuity", "eroded overlap band, discarded at composite"];
  [["Before — masks as depth model", then, COOL], ["Now — masks as clean-up", now, ACCENT]].forEach(([t, items, col], i) => {
    const x = 0.75 + i * 6.05;
    s.addShape(pres.ShapeType.roundRect, {
      x, y: 2.4, w: 5.65, h: 2.9, rectRadius: 0.08,
      fill: { color: "FFFFFF" }, line: { color: col, width: 2 },
    });
    s.addText(t, { x: x + 0.3, y: 2.72, w: 5.05, h: 0.4, margin: 0,
      fontFace: HEAD, fontSize: 17.5, bold: true, color: col });
    bullets(s, items, { x: x + 0.3, y: 3.35, w: 5.05, h: 1.8 }, { fontSize: 15 });
  });
  takeaway(s, "The mask survived. It just stopped being the geometry.");
}

/* =====================================================================
   ACT V — LIMITS & STATUS
   ===================================================================== */
{
  const s = slide({
    act: 5, kicker: "Limits", title: "What this does not do",
    notes:
      "Naming the limits before anyone asks. It models a linear tilt about one axis. Roll about the " +
      "other axis isn't in the model — if the fish is also rolled left-right, that residual grows " +
      "toward the frame edges and the B-spline absorbs it. The plane is assumed flat; genuine optical " +
      "curvature isn't captured. And the anchors are set by eye, so the whole pipeline inherits " +
      "whatever error is in those two numbers — which is why the region centre-of-mass check matters " +
      "more than any overlay.",
  });
  const lims = [
    ["Linear tilt, one axis", "Roll about the other axis isn't modelled; the residual grows toward the frame edges."],
    ["The plane is assumed flat", "Genuine optical curvature is not captured — the B-spline absorbs it."],
    ["The anchors are manual", "Everything downstream inherits the error in those two numbers."],
    ["Intensity-based throughout", "A good-looking merge is necessary, not sufficient — hence the CoM check."],
  ];
  lims.forEach(([t, b], i) => {
    const x = 0.75 + (i % 2) * 6.05;
    const y = 2.15 + Math.floor(i / 2) * 2.1;
    s.addShape(pres.ShapeType.roundRect, {
      x, y, w: 5.65, h: 1.8, rectRadius: 0.08,
      fill: { color: "F4F5F7" }, line: { color: RULE, width: 1 },
    });
    s.addText(t, { x: x + 0.3, y: y + 0.22, w: 5.05, h: 0.4, margin: 0,
      fontFace: TEXT, fontSize: 15.5, bold: true, color: INK });
    s.addText(b, { x: x + 0.3, y: y + 0.68, w: 5.05, h: 0.95, margin: 0, valign: "top", lineSpacing: 20,
      fontFace: TEXT, fontSize: 13, color: BODY });
  });
  takeaway(s, "Everything inherits the two anchors — which is why validation can't be an overlay.");
}

{
  const s = slide({
    act: 5, kicker: "Status", title: "Where this goes next", dark: true,
    notes:
      "Status. The method is running across the DOI cohort and the aligned ROIs are written back into " +
      "the temporal pack with mapZebrain voxel coordinates and region labels, which is what the " +
      "region-wise functional connectivity work is now built on. Next: pulling the oblique functions " +
      "out into a standalone package with a written guide, so this is reproducible outside my notebooks. " +
      "Happy to take questions — especially on the anchor selection, since that's the part I'd most " +
      "like to automate.",
  });
  const items = [
    ["Running", "Applied across the DOI cohort; aligned ROIs written back to the temporal pack with atlas voxels + region labels."],
    ["Building on it", "Region-wise functional connectivity — the DOI vs eggwater difference-in-differences analysis."],
    ["Next", "Extract the oblique functions into a standalone package with a written guide, so it's reproducible outside my notebooks."],
  ];
  items.forEach(([t, b], i) => {
    const y = 2.45 + i * 1.15;
    s.addShape(pres.ShapeType.ellipse, { x: 0.8, y: y + 0.12, w: 0.24, h: 0.24, fill: { color: ACCENT } });
    s.addText(t, { x: 1.3, y, w: 3.0, h: 0.45, margin: 0,
      fontFace: TEXT, fontSize: 16, bold: true, color: ACCENT });
    s.addText(b, { x: 4.3, y, w: 8.2, h: 1.0, margin: 0, valign: "top", lineSpacing: 23,
      fontFace: TEXT, fontSize: 14.5, color: "E8EAEE" });
  });
  s.addText("Thank you — questions?", {
    x: 0.8, y: 6.35, w: 11.7, h: 0.5, margin: 0,
    fontFace: HEAD, fontSize: 22, bold: true, color: "FFFFFF",
  });
}

const out = path.join(__dirname, "oblique_registration.pptx");
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
