# CREATOR BIBLE — "Tess"

The host of GREENLIGHT. A **working content creator**, not a model. The whole credibility of
the series rests on her looking like someone who has a job, an inbox, and a lukewarm coffee.

---

## 1. Who she is

| | |
|---|---|
| Name on screen | **Tess** (first name only; handle `@greenlight.desk`) |
| Age read | 33–38. Old enough to have opinions, young enough to be on the platform |
| Build | Average. Not tall, not styled, not posed |
| Heritage read | Mixed heritage, deliberately not pinned to one ethnicity |
| Register | UK English, soft northern vowels, fast-talker, dry |
| In-world role | Senior comms person who "got put on the AI thing." Not IT. Not legal. Not a boss. |

**She is the colleague who read the policy so you don't have to.** That is her entire
authority — not a title, not a certification.

---

## 2. The anti-brief (read this before any prompt)

Every generator drifts toward a beauty campaign. Fight it explicitly, every time:

| ❌ Not this | ✅ This |
|---|---|
| Ring-light circle catchlight in the eye | Soft rectangular window reflection, off-centre |
| Flawless retouched skin | Visible pores, texture, a spot on the jaw, faint under-eye shadow |
| Symmetrical face, glam makeup | Asymmetric brows, tinted lip balm at most, slight shine on the nose |
| 85mm portrait bokeh, subject isolated | 26mm phone lens, deep-ish focus, the room legible behind her |
| Hair styled and finished | Claw clip, flyaways, one strand she tucks mid-sentence |
| Teal-and-orange grade | Flat, phone-native colour; slightly warm mixed with cool window light |
| Centred, chest-up, still | Off-centre, head near the top of frame, small constant handheld drift |
| Clean white studio | An actual corner of an office with cables, a whiteboard, a radiator |

---

## 3. Physical continuity (lock this across all 6 episodes)

- Dark brown hair, shoulder length, pulled up in a tortoiseshell claw clip; flyaways at the temple.
- Small freckles across the nose and left cheek. A faint 1cm scar above the right eyebrow.
- Clear-frame rectangular glasses — worn in eps 1, 3, 5; pushed up on her head in 2, 4, 6.
- Wardrobe: oversized charcoal wool jumper over a plain white tee (ep 1), rotating to a
  washed-navy overshirt, a grey marl sweatshirt. **No logos, no branding, nothing new-looking.**
- Jewellery: two small gold studs in the left ear, one in the right. A thin steel watch.
- A lanyard, worn, card flipped to the blank side — never legible.
- Nails short, unpainted. One chipped index nail.

## 4. The rooms (rotate — never the same wall twice)

1. **Desk corner** — dual monitor bezel out of focus behind her, a sticky note fringe, a cable snake.
2. **Kitchen counter** — a used mug, a kettle, someone's abandoned lunchbox.
3. **Stairwell landing** — hard daylight from a high window, echo in the room tone.
4. **Meeting room, lights off** — she's ducked in between calls; screen glow on one cheek.
5. **Walking the corridor** — phone at arm's length, the frame bobbing with her step.

## 5. Camera, light, sound

- **Camera:** phone rear or front camera, 26mm equivalent, held at arm's length just
  *below* eye line so she looks very slightly down. Continuous handheld micro-drift.
  One refocus hunt per clip. Rolling-shutter skew if she pans.
- **Light:** north-facing window as key, camera-left, 3/4. Overhead office fluorescent
  spilling cool onto the shoulders. Mixed white balance, uncorrected. Slightly under-exposed
  on the shadow side. Never a beauty light.
- **Sound:** lav mic hidden under the jumper *or* phone mic — consistent per episode.
  Keep the room tone: a distant keyboard, a door, the HVAC. One mild plosive per clip.
- **Grade:** none. Phone-native. Mild highlight clipping on the window is a feature.

## 6. Voice and performance direction

- Opens mid-thought, as if the camera started late. No "Hi guys."
- Hook in the first 1.5 seconds, before she's even settled in frame.
- One self-interruption per script ("— right, no, the actual rule is —").
- Slows down and drops volume on the one line that is the rule. That's the beat that lands.
- Hands in shot. She points at the lens when she means *you*.
- Ends on an instruction or a question, never on "thanks for watching."
- **Never** says "leverage", "empower", "journey", "at Nteractive we believe".

---

## 7. Generation prompts

### 7.1 Character reference still (image model)

> Ultra-realistic candid smartphone photograph of a woman in her mid-thirties, mixed heritage,
> shoulder-length dark brown hair held in a tortoiseshell claw clip with visible flyaways,
> clear rectangular glasses, small freckles across the nose and left cheek, faint scar above the
> right eyebrow, minimal makeup, visible skin texture and pores, slight shine on the nose, faint
> under-eye shadow, asymmetric eyebrows. Wearing an oversized charcoal wool jumper over a plain
> white t-shirt, small gold ear studs, a worn blank lanyard. She is standing in the corner of a
> real working office: out-of-focus monitor bezels, a whiteboard with half-erased notes, a cable
> tray, a radiator. Shot on a phone at arm's length, 26mm wide lens, held slightly below eye line,
> deep depth of field so the room is legible. Soft daylight from a large window camera-left at
> three-quarters, cool fluorescent spill on the shoulders, mixed and uncorrected white balance,
> slightly underexposed shadows, mild highlight clipping on the window. Flat phone-native colour,
> no grade, no retouching, no beauty lighting, no ring-light catchlight, no bokeh portrait look,
> no studio backdrop, no fashion posing. Natural half-smile, mid-sentence expression, looking
> directly into the lens. Documentary realism. 9:16 vertical.

**Negative:** *ring light, ring-light catchlight, glamour, beauty campaign, fashion model,
retouched skin, airbrushed, symmetrical face, studio backdrop, seamless white, 85mm portrait,
shallow bokeh, teal and orange grade, HDR glow, influencer pose, perfect hair, editorial fashion,
stock photo smile, cinematic lens flare, plastic skin.*

### 7.2 Talking-head video clip (video model)

> [CHARACTER REFERENCE] speaking directly to a phone camera held at arm's length. Handheld, the
> frame drifting a few degrees, one autofocus hunt at the start. She is mid-sentence, talking
> quickly, gesturing with her free hand, glancing off-lens once then back. Natural blinks,
> micro-expressions, a small head tilt on emphasis. Real office corner behind her, soft window
> light camera-left, cool overhead spill. 26mm phone lens, deep focus, flat phone-native colour,
> no grade. Sync dialogue: "<LINE>". Ambient room tone, a distant keyboard, HVAC hum.
> Documentary realism, UGC selfie video, not a commercial.

### 7.3 Continuity note for multi-shot episodes

Lock the character reference still as image input for **every** shot. Re-state the wardrobe,
claw clip, glasses state and lanyard in each shot prompt — generators drop accessories first,
then hair, then wardrobe. Check ears (studs) and hands (nails) on every render; those fail first.

---

## 8. Two things to clear before a single frame renders

1. **Synthetic likeness.** Tess must be a composite who resembles no identifiable real person.
   Run a reverse-image check on the approved character still before it goes into production.
   If Nteractive would rather use a real colleague on camera, everything in this bible still
   applies as a casting and direction brief — swap §7 for a call sheet.
2. **Disclosure.** An AI-generated presenter explaining the AI policy is either the best joke in
   the series or the thing that torpedoes its credibility. Get ahead of it: a persistent
   `AI-generated presenter` corner mark on every episode, and let Tess say it out loud in the
   series trailer. The policy demands human review of AI output — the series has to visibly
   live by the rule it is teaching.
