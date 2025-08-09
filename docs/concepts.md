# Concepts

The vocabulary the codebase uses. Most of these words also mean something in
ordinary badminton or computer vision; where the meaning here is narrower, that
is called out, because those are the ones that cause misreadings.

See [architecture.md](architecture.md) for how the pieces fit together.

## Court and geometry

**Court model** — the badminton court as the rule book defines it: a table of
vertices and the lines between them, in centimetres.

**Court model type** — `singles` or `doubles`. Chooses which lines are
boundaries. The same landing can be in for one and out for the other, so this is
an input to the verdict, not a display preference.

**Camera mode** — where the camera is positioned and how much it can see: the whole thing, or
a particular corridor. It determines which boundaries the system is entitled to
adjudicate against. A camera watching the left corridor may not call the far
baseline, because it cannot see it.

**Image coordinates** — pixels; origin top-left, y increasing downwards.

**Court coordinates** — centimetres on the real court. Camera-independent.
Distances are only meaningful here: a pixel covers far more floor at the far
baseline than it does near the camera.

**Homography** — the projective mapping between image and court coordinates.
Recovering it allows to show a top-down video on the court in the review video.

**Session court** — the one court object used for the whole clip. Because the
camera is static, the mapping is a property of the session; it is solved once
from evidence pooled across frames, validated, and then frozen. If it cannot be
validated there is no session court, and the run reports no verdicts.

## Finding the court

These terms name the stages of the solve, in order. The intermediate results are
kept on the solver and are what the court-solver walkthrough draws.

**Court mask** — the segmented region of the frame that is court. Used to
restrict where lines are looked for, and later to score candidate homographies.
A player standing on the court cuts a hole in it.

**Mask completeness** — how much of the court a given frame's mask actually
sees. Since the true footprint is identical on every frame, a smaller or more
ragged mask means something is standing in front of it. Used to pick the least
occluded frame to solve on.

**Line segment** — a raw straight piece of edge found inside the mask. Many per
painted line, and plenty of spurious ones.

**Line family** — segments split into two groups by orientation: those running
the length of the court and those running across it. The split is by image
angle, so "vertical" and "horizontal" refer to the picture, not the court.

**Stripe** — a painted court line has real width (4 cm), so it shows up as two
parallel edges, one per painted border.

**Line cluster** — a stripe's two edges collapsed into the single line through
its middle.

**Stripe centre vs boundary edge** — the crucial distinction. Detection finds
the *centre* of the painted stripe; the rule book defines boundaries at the
stripe's outer *edge*. Matching is done on centres, so that the solved boundary
lands on the paint rather than 2 cm off. Confusing the two produces a homography
that is subtly and consistently wrong.

**Consensus lines** — the lines that recur across many sampled frames, each kept
at its median position. Occlusions and shadows come and go; real lines do not.
The homography is solved from these rather than from any single frame.

**Line assignment** — a guess at *which* model lines the detected lines are. A
partial view does not say whether the two visible long lines are the sidelines
or a sideline and the centre line, so every order-preserving possibility is
tried and scored.

**Mask IoU** — the overlap between the court boundary as projected by a
candidate homography and the segmented mask. The score that picks the winning
assignment, and the validation gate: a homography that fits nothing is rejected
rather than used.

## The shuttle

**Detection** — a bounding box around the shuttle on one frame. The pipeline
follows a single shuttle and takes the most confident box.

**Cork** — the weighted nose of the shuttlecock. **This is the point that gets
judged**, not the centre of the bounding box: a shuttle lands cork-first and
tilts, so the box centre can sit centimetres away from the actual contact point.
Estimated from the pixels during the detection pass, with the direction of
travel as a hint; falls back to the box centre when the shuttle is too slow for
its heading to mean anything.

**Track** — one frame's worth of shuttle measurements: the box, a smoothed
speed, a smoothed direction, the turn angle against the previous detections, the
blur, and the cork. Carries no pixels, so the whole run's worth fits in memory
and can be saved, re-analysed, or swept without the video.

**Blur** — how smeared the shuttle looks inside its box. A motion cue that can
only be measured while the pixels are in hand, so the detection pass measures it
even though later stages are what use it.

**Flight** — a run of consecutive detections that could plausibly be the same
shuttle travelling. A flight is *broken* by a gap too long or a jump too large:
across a cut, or when the detector latches onto something else, the shuttle
appears to teleport, and the two sides must not be reasoned about as one motion.

**Trajectory** — a smooth run of tracks between impacts. What the drawn trail
follows.

## Adjudication

**Hit** — narrower here than in ordinary use. Sokil means **a contact with the
floor**, the event a line call depends on — not a racket stroke. Racket contacts
are impacts too and the detector sees them, but they are filtered out.

**Impact score** — how much a frame looks like a contact, combining how sharply
the direction turned and how much speed was lost. A shuttle in free flight keeps
going the same way; the frames where it stops doing so are the candidates.

**Contact frame** — the exact frame of touchdown. The detector fires slightly
early by construction, since it must look at frames on both sides of a
candidate, so the contact is located afterwards as the lowest point nearby.

**Non-maximum suppression** — one real impact makes several neighbouring frames
look suspicious. Only the strongest in each cluster is kept.

**Refractory period** — a cooldown after an accepted hit, so the bounce is not
reported as a second landing.

**Observable area** — the court plus a margin, as a region in the image. A
plausibility gate, not a verdict: a direction change outside it happened on
another court or too high to be a landing, and is never adjudicated.

**IN region** — the area a landing must touch to be called in, bounded only by
the lines this camera is entitled to judge. Sides the camera cannot see are
left open, so a shuttle deep in the unseen court is not called out on evidence
that does not exist.

**Owned boundary** — a line the camera can see and may therefore call against.
The complement of the open sides above.

**Verdict** — in or out, plus the margin in centimetres. The shuttle's real
diameter is part of the test: if any part of it touches the in region it is in,
even when the cork is on the wrong side of the line. Purely geometric, so it can
always be explained as a distance.

## Pipeline and output

**Review** — one run over one clip: detect, solve, find landings, judge, and
optionally render. Callable as a whole or a step at a time.

**Undistortion** — removing lens curvature, using an intrinsics matrix and
distortion coefficients measured beforehand from a checkerboard. Optional, but
without it straight court lines are not straight and the homography inherits the
error.

**Frame number** — the frame's position in decode order. The identifier every
stage joins on, and the same axis the annotation tool uses.

**Review video** — the rendered output: the tracking overlay, and on each
landing a magnified beat on the contact followed by the top-down close-up
carrying the verdict.

**Close-up** — the top-down view of a single landing at true scale, showing the
line that decided the call and the margin. Deliberately tight: the centimetres
that decide a call are invisible at whole-court scale.

**Stage** — one picture of one thing the pipeline does, used for debugging and
for the walkthroughs. Stages are images, never video; sequencing them is the
caller's job.
