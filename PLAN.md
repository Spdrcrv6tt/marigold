# Command chip: build plan (Marigold)

Nov 15, 2026 · John. Codename Marigold (John, Nov 15). Transcribed from John's plan, docs/plan/command-chip-build-plan-2026-11-15.pdf; the PDF is the original. docs/decisions.md overrides this where they differ.

## Summary

The command chip is an always-listening voice-control chip that OEMs drop into products. It hears a wake word, works out what the person wants in their own words, and tells the product what to do. One piece of silicon serves every customer: a fan maker and a microwave maker each load their own wake word, actions and example phrases through an SDK, with no model retraining.

It is Poinsettia's smaller sibling. Poinsettia writes down anything said; the command chip skips text and goes straight from speech to an action plus its values ("fan speed up", "start, 120 s"). That keeps the die a few mm² instead of Poinsettia's 34.8 mm², which is what makes a few-dollar appliance part possible.

The opening is the combination nobody sells today in one cheap part: open phrasing instead of fixed phrases, OEM setup by typing examples, microwatt listening and a fast response. First market is home appliances (fans, purifiers, microwaves, ovens, washers); vehicles come later as an automotive-grade version.

Nothing here is built yet. Every number below is a target or an estimate until the software proof and the FPGA demo measure it.

## How it works

The core powers up only after a wake word; the OEM config picks the action.

- Always-on island (microwatts): PDM microphone (1-bit stream, always on), then the voice detector ("is someone talking?"), then the wake-word spotter ("was that the wake phrase?").
- Core (powered only after a wake): intent model (speech to a meaning vector, 2M to 5M weights in ROM), then the matcher (closest OEM action + values; no close match: do nothing), then a host message (I2C or UART: "start, 120 s") or GPIO pins (pulse, latch or toggle).
- OEM config: typed in the SDK on a PC (wake phrase, actions, values), loaded at boot.

The island runs at microwatts; the core, its ROM and its working memory stay off until a wake word is heard, and switch off again once the message is out.

- **Matching, not transcription.** The model turns speech into a meaning vector. The SDK turns each OEM example phrase into a vector on a PC, and the chip picks the closest action. Phrasing stays open; only the list of actions is fixed.
- **Values.** A small value head pulls out numbers, durations and levels, so "two minutes" arrives as start, 120 s.
- **Outputs.** Products with a controller get a short message over I2C or UART and decide for themselves, for example by checking the door is shut. Products with no controller get GPIO pins.
- **Wake word.** The OEM types its wake phrase in the SDK; the spotter matches its sound, with a sensitivity setting the OEM tunes against false wakes.

## Iteration 1 targets

Iteration 1 is done when the design hits these numbers in software and then bit for bit on the KV260, the same standard Poinsettia met. They are starting targets: revise them once the first software baseline exists.

| # | Item | Target | How it is measured |
|---|---|---|---|
| 1 | Intent accuracy, phrasings the OEM typed | ≥ 97% | Held-out recordings of the example phrases |
| 2 | Intent accuracy, phrasings the OEM never typed | ≥ 90% | Paraphrase test set ("crank it", "I'm roasting") |
| 3 | Value extraction | ≥ 95% exact | Numbers, durations, levels: "two minutes", "set it to 72", "level 3" |
| 4 | Rejecting non-commands after a wake | ≥ 95% rejected | Ordinary talk after the wake word fires nothing |
| 5 | Wake word detection | ≥ 95% at 1 to 3 m | Recorded wake phrases, several speakers and rooms |
| 6 | False wakes | ≤ 1 per 24 h | 24 h of TV, music and conversation |
| 7 | Response time | ≤ 300 ms median | End of speech to message on the output pins |
| 8 | Model size | ≤ 5M weights at 4-bit | ≤ 20 Mbit of ROM, about 1.7 mm² at 28 nm |
| 9 | Listening power, chip only | ≤ 100 µW at 28 nm (estimate) | Same method as Poinsettia's 13 to 94 µW island |
| 10 | Product noise | ≤ 5 points accuracy loss | Fan at full speed or microwave running, 10 dB below the voice |
| 11 | Same silicon, two products | Fan and microwave configs on one model | No retraining between them |
| 12 | Bit-exact hardware | RTL matches the golden model | Verilator, mic pin to output pins |

## Competition and positioning

Every piece of this exists somewhere; nobody sells the combination in one cheap part. The pitch is "here is what's out there, and here is where we beat it," never "this doesn't exist."

| Competitor | What it is | The gap we target |
|---|---|---|
| Picovoice Rhino | Speech-to-intent software that runs offline on microcontrollers, with intents and slots | Their docs say every anticipated variation must be written as an expression. We aim to generalize past the typed examples. |
| Picovoice Porcupine | Wake-word software; OEMs type a phrase and get a model in seconds | Type-to-train wake words are table stakes, not our edge. We match it on dedicated silicon. |
| Syntiant NDP120 | Dedicated always-on voice chip; reported under 500 µW always-on in specified configurations | Strong on listening power, built around wake words and command words. We target lower listening power plus open phrasing and values. |
| Chipintelli and Unisound | Cheap offline voice chips for appliances; Chipintelli lists up to 500 command words and 5,000+ customers | Fixed command words. They set the price floor we have to live near. |
| Sensory | On-device speech-to-text software for NPUs and microcontrollers, launched April 2026 | Software on someone else's chip. We win on power per command. |
| Moonshine Micro | Open-source voice interfaces for sub-$1 microcontrollers, from the team behind Moonshine | Same model family as Poinsettia: a possible partner as much as a rival. |

Our claim, once measured: open phrasing like a cloud assistant, setup by typing examples like Porcupine, listening power at or below Syntiant's, in a part priced near the appliance chips. The benchmark that proves it is Rhino on a Cortex-M versus our chip on the same commands (phase 6).

## Chip architecture and cost

The command chip is Poinsettia's always-on island plus a much smaller core: about 2.5 to 4 mm² at 28 nm, roughly $0.12 to $0.20 of silicon per die before packaging, test and masks. Areas scale Poinsettia's own 28 nm densities (ROM about 0.086 µm² per bit, SRAM about 1.6 mm² per MB); all are estimates.

| Block | Reused or new | Job | Area at 28 nm (est.) |
|---|---|---|---|
| PDM front end and voice detector | Reused | Mic bitstream to 16 kHz samples; "is someone talking?" | ~0.05 mm² |
| Capture buffer, 32 to 64 KB | Reused, shrunk from 128 KB | Keeps 1 to 2 s from before the wake word | 0.05 to 0.1 mm² |
| Wake-word spotter | New, inside the always-on island | "Was that the wake phrase?" with tens of thousands of weights | ~0.05 mm² |
| Intent model ROM | New model, Poinsettia's ROM style | One general model, 2M to 5M weights at 4-bit | 0.7 to 1.7 mm² |
| Working SRAM, ≤ 256 KB | Reused design | Features for a 2 to 3 s command | ~0.4 mm² |
| Config SRAM, ≤ 32 KB | New | The OEM's action vectors, value rules, wake template, thresholds | ~0.05 mm² |
| MAC arrays, per-element unit, RV32IM controller | Reused, fewer lanes | Runs the model and the matching | 0.2 to 0.4 mm² |
| Program memory, 32 KB | Reused | Firmware | 0.08 mm² |
| Pads and power switches | Reused method | About 20 pads | Takes the die to 2.5 to 4 mm² |

**Interfaces.**
- PDM microphone: clock out, 1-bit data in, same as Poinsettia.
- Host link: I2C target or UART, picked by a strap pin, plus an interrupt pin. Each command arrives as a short message: action ID, values, confidence.
- 8 GPIO for products with no controller: each action can pulse, latch or toggle a pin.
- Config: loaded by the host at boot over I2C, or read from a small SPI flash on standalone products.
- Clocks: 3.072 MHz always on, fast clock enabled only after a wake, as on Poinsettia.

Weights stay in ROM because every OEM runs the same general model; their customization is a few KB of data, not a new model. Flash only matters later, for language variants.

**Cost.** A TSMC 28 nm wafer runs about $3,000 (Silicon Analysts, Aug 2026 page citing 2024 pricing). At 2.5 to 4 mm² that is roughly 17,000 to 28,000 dies per wafer; at an assumed 90% yield, $0.12 to $0.20 each. Masks dominate at low volume: a 28 nm set runs $1M to $3M (Silicon Analysts), which adds $1 to $3 per chip over the first million units. A few-dollar appliance price needs volume in the millions.

## Build phases

Software first, then hardware, using Poinsettia's playbook: prove it on a laptop, freeze a golden integer model, match it bit for bit in RTL, then demo on the KV260. Phases 0 and 1 run side by side; the business track runs from week one.

1. **Phase 0: intent model on a laptop (the real research).**
    - Train a small audio encoder so its output vector lands near a text-embedding model's vector for the same sentence. The text model lives only in the SDK on a PC; the chip carries just the audio encoder.
    - Start from Moonshine's encoder (MIT license) and shrink toward ≤ 5M weights.
    - Add a value head for numbers, durations and levels ("two minutes" becomes 120 s), and a reject path for non-commands.
    - Train on commercially clean data: LibriSpeech and Speech Commands (both CC BY 4.0), plus synthetic command audio from many TTS voices mixed with fan, microwave and TV noise.
    - Evaluate only on SLURP (audio is CC BY-NC 4.0) and Fluent Speech Commands (academic research only) unless we license them.
    - Gate: targets 1 to 4, 8 and 11 met on held-out speakers.
2. **Phase 1: wake word on a laptop.**
    - Tiny phoneme model for the always-on island; the SDK turns the typed wake phrase into a phoneme template the chip matches against.
    - Gate: targets 5 and 6.
3. **Phase 2: golden model and RTL.**
    - Quantize to 4-bit with quantization-aware training and write the integer reference, as Poinsettia's golden format did.
    - SystemVerilog reusing Poinsettia's front end, voice detector, controller and MAC arrays; add the wake spotter, matcher, I2C, GPIO and config loader.
    - Gate: Verilator run matches the golden model bit for bit, mic pin to output pins, and negative controls fail (target 12).
4. **Phase 3: KV260 demo.**
    - PDM mic on the Pmod connector, a PWM fan, an LED, and a small MCU playing the microwave's controller over I2C.
    - Demo: "it's kinda hot in here" speeds the fan up; "two minutes" sends start, 120 s to the host.
    - Gate: target 7 measured on hardware, demo video recorded.
5. **Phase 4: silicon estimate.**
    - Poinsettia's method: GF180 power measured from simulation, a 45 nm layout, scaled to 28 nm.
    - Gate: target 9 and a die inside 2.5 to 4 mm².
6. **Phase 5: SDK v0.**
    - Desktop tool: type the wake phrase, actions, example phrases, values and GPIO mapping; it compiles a config blob and loads it over I2C or UART.
    - Built-in test with synthetic voices, so an OEM sees accuracy before touching hardware.
    - Gate: someone new configures a third product (an air purifier) in under an hour with no ML help.
7. **Phase 6: business track (from week one).**
    - Benchmark against Rhino on a Cortex-M: accuracy on unseen phrasings, power, parts cost.
    - Inventor's write-up of the text-enrolled intent matching, then a provisional patent with counsel.
    - Talk to 15 to 20 appliance hardware leads; aim for "we'd evaluate samples" letters.
    - SBIR/STTR with a TAMU faculty partner; reach out to Moonshine AI.

## Risks and open questions

The biggest risk is research, not hardware: open phrasing from a model this small is unproven, which is why phase 0 measures it before any RTL is written.

| Risk | Why it matters | What we do about it |
|---|---|---|
| Accuracy at ≤ 5M weights | The whole pitch is open phrasing | Measure first in phase 0; fallbacks are a bigger ROM or a narrower phrasing promise |
| Phrasings far from the OEM's examples | "I'm roasting" may not land near "fan up" | The SDK uses an LLM on the PC to expand each example into hundreds of paraphrases before embedding them |
| The product's own noise | A fan or microwave sits right next to the mic | Train with appliance noise; let OEMs add recordings of their product in the SDK |
| False wakes from TV and talk | One wrong wake and users turn it off | Two-stage check (spotter, then intent model confirms), tunable sensitivity |
| Masks and volume | $1M to $3M masks need millions of units to reach appliance prices | MPW runs until design wins justify full masks |
| Patents held by incumbents | Picovoice, Sensory and Syntiant have worked this space for years | Freedom-to-operate search before the provisional and before tapeout |
| Do people use voice on appliances? | Customers buy what end users actually use | Put the KV260 demo in front of real users early |
| Safety on heating or moving products | "Start" must never bypass a door interlock | The host MCU always makes the final call; the SDK docs say so plainly |
| Training data licenses | Some speech datasets bar commercial use | Train on CC BY data and synthetic audio; check TTS voice licenses |

Open questions: the codename (now Marigold), English-only at launch or a second language early, and whether chip sales or licensing the design comes first.

## First session

The first day should answer one question cheaply: can text matching alone hit 90% on phrasings the OEM never typed? If text can't, audio won't either, so fix the matching before training anything on audio.

- [ ] Pick a codename (Marigold), start a new repo, and copy the front end, voice detector, controller and MAC RTL from Poinsettia commit 1719528
- [ ] Write docs/iteration-1.md with the 12 targets above as the definition of done
- [ ] Log decision 1 in docs/decisions.md: speech-to-intent with text-enrolled matching, no transcription
- [ ] Define config format v0: wake phrase, actions, example phrases, values (number, duration, level), GPIO mapping
- [ ] Write two product configs: a fan (on, off, faster, slower, set level, oscillate) and a microwave (start for a time, add 30 s, stop, power level)
- [ ] Build a paraphrase test set: 50+ ways to say each action, none copied from the examples
- [ ] Text-only upper bound: embed examples and test phrases with a small text-embedding model, match by similarity, measure accuracy and rejection
- [ ] Download LibriSpeech and Speech Commands for training, SLURP audio for evaluation only
- [ ] Generate the first synthetic command audio: many TTS voices, every paraphrase, fan and microwave noise mixed in
- [ ] Record 30 minutes of a real fan and microwave running for the noise test set (John)
