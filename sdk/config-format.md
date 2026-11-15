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
| `actions[]` | Up to 64 actions. `id` (1 to 255) goes in the host message. `name` and `description` are for people. `examples` are the typed phrases (5 to 30 recommended), and `values` names the slots the action can carry. |
| `actions[].gpio` | Optional. `pins` are GPIO numbers 0 to 7. `mode` is `pulse` (with `pulse_ms`), `latch` (with `level` 0 or 1) or `toggle`. A slot value can drive a pin group with `"encode": "binary"`. |

## Host message (v0)

`action id (1 byte), value count (1 byte), each value as slot index (1 byte) + int32 little-endian, confidence (1 byte, 0 to 255)`. Durations travel in seconds, levels as integers. A rejected command sends nothing. The host always makes the final call: a microwave's controller checks the door before it starts.

## Rules the SDK enforces

- Every action needs at least 5 examples, and no example may appear under two actions.
- Example phrases may name the value slot with a number ("cook for two minutes"); the SDK keeps the words, and the value head reads the number.
- Safety: the SDK docs say plainly that voice commands never bypass interlocks; the chip only reports what it heard.
