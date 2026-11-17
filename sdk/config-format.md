# Marigold config format v0

One JSON file per product, written by the OEM in the SDK. The SDK compiles it into a config blob (action vectors, value rules, wake template, thresholds; at most 32 KB of config SRAM) that the host loads at boot over I2C or UART, or that the chip reads from SPI flash. v0 is the source format only; the blob layout comes later.

```json
{
  "format": "marigold-config/0",
  "product": "Desk fan",
  "wake": { "phrase": "hey breeze", "sensitivity": 0.5 },
  "match": { "reject_threshold": null },
  "output": { "host": "i2c", "i2c_address": "0x2A", "irq_pin": true },
  "values": {
    "speed": { "type": "level", "min": 1, "max": 5,
               "words": { "low": 1, "medium": 3, "high": 5 } }
  },
  "actions": [
    { "id": 1, "name": "on",
      "description": "Turn the fan on",
      "examples": ["turn the fan on", "switch it on"],
      "values": [],
      "gpio": { "pins": [0], "mode": "latch", "level": 1 } }
  ]
}
```

## Fields

| Field | Meaning |
|---|---|
| `format` | Always `marigold-config/0` for this version. |
| `product` | Free text, for the OEM's own records. |
| `wake.phrase` | The wake phrase as typed. The SDK turns it into a phoneme template for the spotter. |
| `wake.sensitivity` | 0 to 1. Higher wakes more easily and false-wakes more often. |
| `match.reject_threshold` | Similarity below which a command is rejected. `null` means the SDK calibrates it on its built-in non-command set. |
| `output.host` | `i2c`, `uart` or `none` (GPIO only). Picked on the chip by a strap pin; this records the product's choice. |
| `output.i2c_address` | 7-bit target address, when `host` is `i2c`. |
| `output.irq_pin` | Whether the interrupt pin is used. |
| `values.<name>` | A value slot an action can carry. `type` is `number` (`min`, `max`, optional `unit`), `duration` (`min_s`, `max_s`) or `level` (`min`, `max`, optional `words` mapping spoken words to levels). |
| `actions[]` | Up to 64 actions (including event entries, below). `id` (1 to 239; 0xF0 and up are reserved for events) goes in the host message. `name` and `description` are for people. `examples` are the typed phrases (5 to 30 recommended), and `values` names the slots the action can carry. |
| `actions[].gpio` | Optional, one rule or a list, applied in order when the action is accepted (docs/architecture.md 8.5). `pins` are distinct GPIO numbers 0 to 7. `mode` is `pulse` (with `pulse_ms` 1 to 65535 and optional `level`, the active level, default 1: the pins go to `level` for pulse_ms, then to the opposite level), `latch` (with `level` 0 or 1) or `toggle`. A slot value can drive a pin group with `"encode": "binary", "value": "<slot>"`: least significant bit on the lowest pin, clamped to 0 to 2^k - 1 for k pins; the pins do not change if the action does not carry that slot. |

| `events` | Optional. What the chip reports besides commands: `wake` (wake word heard, listening), `nomatch` (speech after the wake matched nothing confidently) and `timeout` (woke, but no speech within the listen window). Each takes `host` (true: send the event frame to the host; sets flags bit 1, 2 or 3 in the blob) and an optional `gpio` rule or list of rules in the action format (pulse, latch or toggle; no value encoding). The SDK compiles an event with GPIO rules into an action-table entry with id 0xF0 (wake), 0xF1 (nomatch) or 0xF2 (timeout); no match vector points at it. Example: a buzzer on pin 4 for a "didn't catch that" ding. |

### Events example

```json
"events": {
  "wake":    { "host": true, "gpio": { "pins": [4], "mode": "latch", "level": 1 } },
  "nomatch": { "host": true, "gpio": { "pins": [4], "mode": "pulse", "pulse_ms": 150 } },
  "timeout": { "host": false }
}
```

The reference configs use it: the GPIO-only fan pulses pin 4 on NOMATCH; the I2C microwave sends WAKE, NOMATCH and TIMEOUT to its controller.

## Host message (v0)

Defined in docs/config-blob.md: `0xA5, type u8 (0 COMMAND, 1 WAKE, 2 NOMATCH, 3 TIMEOUT), action id u8 (0 for events), n_values u8, then for each value: slot u8 + int32 LE, confidence u8, crc8`. Durations travel in seconds, levels as integers. A rejected command is a NOMATCH event, sent only when `events.nomatch.host` is true. Every WAKE is followed by exactly one decision (COMMAND, NOMATCH or TIMEOUT) and decisions come in segment order, but the next WAKE may arrive before the previous decision: pair them first-in, first-out (docs/config-blob.md). The host always makes the final call: a microwave's controller checks the door before it starts.

## Compiling

`python sdk/compile.py sdk/configs/<product>.json` writes data/sdk/<product>.mgc (config blob v0) and a JSON report next to it; `--parse BLOB` prints a blob back. Tests: `python -m pytest -q sdk/tests`. Placeholders until docs/golden-format.md and docs/golden-wake.md exist are listed at the top of sdk/compile.py.

## Rules the SDK enforces

- Every action needs at least 5 examples, and no example may appear under two actions.
- Action ids are unique, 1 to 239; actions plus event entries are at most 64, GPIO rules at most 32, pins 0 to 7.
- Example phrases may name the value slot with a number ("cook for two minutes"); the SDK keeps the words, and the value head reads the number.
- Safety: the SDK docs say plainly that voice commands never bypass interlocks; the chip only reports what it heard.
