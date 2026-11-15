# Decisions

- 2026-11-15 (John: "Poinsettia seems decently built, so I want to move on to this idea. You are to start a new session on this computer and pass it all of the information about Poinsettia / allow it to use it. I want to reuse as much as we possibly can and use as much as we've learned to build this chip as efficiently as possible."): Farallon's command chip starts, with John's plan in PLAN.md (original PDF in docs/plan/).
  - Codename **Marigold** (John's pick, Nov 15). Repo D:\marigold, junction ~/Marigold, local git only.
  - Poinsetta (D:\poinsetta, final commit 1719528) and Imhotep (D:\imhotep) are read-only references. Copy from them; never write to them. docs/poinsetta-handoff.md maps what to reuse and the lessons learned.
  - Carried over from Poinsetta: weights hardwired in on-die ROM, one bit per cell (patent guardrail); software validation first (golden model, then bit-exact RTL); only 28 nm figures matter, with GF180 as the measured basis and a 45 nm layout for grounding; every agent on Opus 5.5 at max effort; usage stop at 88% (John's cap is 90%).
  - Silicon Chatbot and Imhotep are on hold (John, Nov 15).
