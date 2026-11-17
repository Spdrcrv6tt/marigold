# Config blob v1 (what the host loads into config SRAM)

The coordinator fixed this on Nov 15 so the SDK, core and model tracks can work at the same time. Change it only through a dated entry in docs/decisions.md, and tell the other tracks. All fields are little endian. The blob is at most 32 KB. Version 1 (John's ruling K1, Nov 17, 2026) records the model and teacher the blob was built for; a version 0 blob is refused (error 2).

## Layout

| Offset | Field | Type | Meaning |
|---|---|---|---|
| 0x00 | magic | 4 B | "MGC0" |
| 0x04 | version | u16 | 1 |
| 0x06 | D | u16 | Embedding dimension. It equals the intent encoder's output dimension (the model agent sets it in docs/golden-format.md; 256 or less recommended). |
| 0x08 | N | u16 | Number of match vectors |
| 0x0A | n_actions | u8 | 1 to 64 |
| 0x0B | n_slots | u8 | 0 to 8 |
| 0x0C | n_gpio | u8 | Number of GPIO rules, 0 to 32 |
| 0x0D | wake_len | u8 | Length of the wake template in bytes |
| 0x0E | host | u8 | 0 none (GPIO only), 1 I2C, 2 UART |
| 0x0F | i2c_addr | u8 | 7-bit address |
| 0x10 | reject_thr | i32 | A score below this rejects |
| 0x14 | margin | i32 | Best score minus the best score of a *different* action; below this rejects |
| 0x18 | wake_sens | u16 | Wake sensitivity, units set by docs/golden-wake.md |
| 0x1A | flags | u16 | bit 0: IRQ pin used; bit 1: send WAKE events; bit 2: send NOMATCH events; bit 3: send TIMEOUT events (to the host; GPIO event rules work regardless) |
| 0x1C | conf_sh | u8 | Confidence shift: confidence = clamp(s_b >> conf_sh, 0, 255); 9 for D = 128 (golden format) |
| 0x1D | model | u8 | The chip model the blob was built for: its tag number (3 for mg-v3; sdk/chip_models.py) |
| 0x1E | teacher | u8 | The teacher whose embeddings the vectors are in (4 for teacher-v4; 0 for the SDK's test stub, which no chip loads) |
| 0x1F | reserved | u8 | 0 |
| 0x20 | vector table | N × 4 B | per vector: action u8 (0 = reject class, else an action's id), flags u8, bias i16 |
| ... | vectors | N × D × i8 | row-major, in the encoder output's integer scale |
| ... | action table | n_actions × 4 B | id u8, slot_mask u8 (bit k = slot k), gpio_first u8, gpio_count u8. Ids 0xF0 (WAKE), 0xF1 (NOMATCH) and 0xF2 (TIMEOUT) are event entries: no vectors point at them, and their GPIO rules fire on that event (e.g. a buzzer pin for a 'didn't catch that' ding). |
| ... | slot table | n_slots × 12 B | type u8 (0 number, 1 duration in s, 2 level), low u8, medium u8, high u8 (level slots: the values the spoken words low/medium/high map to; 0 = the value head's default rule; 0 for other types), min i32, max i32 |
| ... | gpio rules | n_gpio × 8 B | pin_mask u8, mode u8 (0 pulse, 1 latch, 2 toggle, 3 slot value in binary on pin_mask), level u8, slot u8, pulse_ms u16, pad u16 |
| ... | wake template | wake_len B | Format set by docs/golden-wake.md (phoneme ids and any extras) |
| end-4 | crc32 | u32 | CRC-32 (IEEE) over every byte before it |

## Matching rule (the golden model and the RTL must agree exactly)

1. q = the encoder's int8 output vector (D values).
2. For each vector i: s_i = sum(q[d] * v_i[d]) + bias_i, computed in int32.
3. b = argmax s_i (lowest index wins ties).
4. If action[b] == 0, or s_b < reject_thr, or s_b - max{s_j : action[j] != action[b]} < margin: reject. Rejection is reported as a NOMATCH event (below), never as an action.
5. Otherwise, emit action[b] together with the values of its slots that were said (docs/golden-format.md), each clamped to the slot's min and max. A slot the value head found no value for (raw code INT32_MIN, 0x80000000) is left out: not clamped, not sent (R16, John's ruling of Nov 17, 2026), so a value nobody said never looks like the slot's minimum.

This covers nearest-prototype matching with reject prototypes (several vectors per action plus action-0 vectors) and linear heads (one vector per action, with bias). The SDK picks what to emit.

## Events and host message (I2C read or UART frame)

The chip reports what happened after every wake, so a product can show it: a light while listening, a 'didn't catch that' ding, and so on.

| Event | When | Type byte |
|---|---|---|
| COMMAND | A confident match: action plus values | 0 |
| WAKE | Wake word detected, core powering up (listening) | 1 |
| NOMATCH | Speech was heard after the wake but nothing matched confidently (step 4 rejected it) | 2 |
| TIMEOUT | Woke, but no speech followed within the listen window (default 4 s) | 3 |

Frame: `0xA5, type u8, action id u8 (0 for non-command events), n_values u8, then for each value: slot u8 + i32 LE, confidence u8 (0 to 255; for NOMATCH the best rejected score, scaled), crc8`. n_values counts only the values that were said: a slot of the action that is missing from the frame was not said ("start" with no time is start with no values; the host applies its own default). A GPIO binary rule (mode 3) on a slot that was not said leaves its pins as they were. COMMAND is always sent. WAKE, NOMATCH and TIMEOUT go to the host only if their flag bits are set. The IRQ pin rises when a frame is ready and falls when the host reads it. In GPIO-only products, the event entries' rules (ids 0xF0 to 0xF2) drive the pins, e.g. pulse a buzzer on NOMATCH.

Pairing (review fixes, Nov 17): every WAKE frame is followed by exactly one decision frame (COMMAND, NOMATCH or TIMEOUT), and decisions come in segment order. The next segment's WAKE may arrive before the previous segment's decision (the chip boots its core again for a segment that opened while it finished the previous one), so a host pairs WAKEs and decisions first-in, first-out. With WAKE, NOMATCH or TIMEOUT frames switched off by the flags, only the frames that are on are sent, in the same order.

Config errors (STATUS bits 7:4 after a commit): 1 magic, 2 version (a version 0 blob too), 3 field range, 4 length, 5 CRC, 6 D is not the chip model's output size (review R4), 7 built for another model: the model or teacher byte is not the chip's (K1; models v1, v2 and v3 all have D = 128, so a blob compiled with the wrong teacher would otherwise load and match garbage). The checks run in that order.

Safety: the host always makes the final call (door interlocks and the like). A COMMAND event is a report of what was heard, not a power switch.
