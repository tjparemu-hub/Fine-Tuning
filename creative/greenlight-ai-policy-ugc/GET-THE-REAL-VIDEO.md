# Getting a real person, with real audio

Everything in this folder is pre-production: script, timings, framing, captions.
None of it is footage. Here are the three routes that produce footage, honestly
compared.

---

## Route A — Film a colleague on a phone · 30 minutes · £0 · **recommended**

For a compliance video this is very probably the right answer, and not as a
fallback. It is faster than generating, it costs nothing, and it removes every
approval in `compliance-checklist.md` except consent. It is also more persuasive:
the whole premise of the series is "a colleague who read the policy so you
didn't have to", and a colleague is the one thing AI can't fake better than the
real thing.

1. Cast anyone in the building who talks fast and isn't on the exec team.
   Not the Head of IT — the point is that this is peer-to-peer.
2. Phone on a small tripod or propped on a monitor, **rear camera**, 4K, 60fps.
   Lock exposure. Shoot in an actual office corner with the window to camera-left.
3. Script on a laptop just under the lens. Eight takes, one per shot in
   `ep01-work-vs-web-30s.md`. Three passes each — the third is always the one.
4. Audio: a £40 wireless lav, or the phone at chest distance in a quiet room.
   **Audio quality matters far more than image quality.** Bad picture reads as
   authentic; bad sound reads as amateur.
5. Cut to the timings in the beat sheet, burn `ep01.srt`, done.

Total kit: a phone, a lav mic, and 30 minutes of someone's afternoon.

## Route B — AI avatar tool · ~1 hour · ~£20-50/month

HeyGen, Arcads, Creatify, Captions.ai. You pick a stock UGC presenter and paste a
script; the tool produces a real-looking person with synced speech. This is the
fastest route to the thing you asked for.

→ Use `prompts/avatar-script.txt` — it has the avatar selection criteria (most
stock avatars look like spokespeople, which kills it), the voice settings, and
the script formatted for paste.

**Trade-off:** stock avatars are recognisable. Some of your staff will have seen
the same face selling a supplement. That undercuts a policy video more than it
would a product ad.

## Route C — Generative video model · ~2 hours · ~£50-150 in credits

Veo 3 (the only one that does convincing sync dialogue right now), Kling, Sora,
Runway Gen-4. Highest ceiling, highest effort, most variance — expect to
regenerate each shot three to six times, and expect continuity drift between
shots even with a locked character reference.

→ Use `prompts/veo3-shots.txt` — character still first, then the eight shot
prompts with dialogue, plus the negative prompt block.

**Trade-off:** eight separate generations of the same synthetic person will not
match perfectly. Cutting between them is where the illusion usually breaks.

---

## Why none of this happened in the Claude Code session

The session's container has no speech synthesis and no video model, and its
network policy blocks model-provider APIs outright — `api.openai.com` returns
`403 to CONNECT` at the gateway. An API key alone would not change that; the
environment's network policy has to be widened first. See
https://code.claude.com/docs/en/claude-code-on-the-web for how environments are
configured.

If that policy is widened and an approved provider key is available, the shot
prompts here can be driven from a session directly. Until then, the animatic
(`ep01_two-copilots_30s.mp4`) is previz for sign-off — it establishes pacing,
framing and captions, and it is deliberately marked "NOT FINAL FOOTAGE".

## Whichever route — before you start

Routes B and C put an AI tool at the centre of production, so guardrail 1 in
`compliance-checklist.md` applies to the tool itself: Head of IT and Head of ESG
approve it and it goes on the Trelica registry **before** the first render. It
would be a poor look for the AI policy video to be made with an unapproved AI
tool. Route A needs none of that — only the presenter's written consent.
