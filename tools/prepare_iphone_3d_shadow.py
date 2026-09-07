#!/usr/bin/env python3.11
"""Stage the local ExtremeWeather shader for a physical-device shadow test."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import textwrap


def shader_packet(source: bytes) -> dict:
    text = source.decode("utf-8")
    blocks = re.findall(r'let source = """\n(.*?)\n[ \t]*"""', text, re.S)
    blocks = [block for block in blocks if "fragment float4 regional_volume_fragment(" in block]
    if len(blocks) != 1 or "\\" in blocks[0]:
        raise ValueError("expected one literal, interpolation-free weather shader")
    shader = textwrap.dedent(blocks[0]) + "\n"
    for entry in ("regional_volume_vertex", "regional_precipitation_fragment"):
        if entry not in shader:
            raise ValueError(f"shader entry point missing: {entry}")
    return {
        "schema_version": 1,
        "renderer": "ExtremeWeather Regional3DMetalVolumeView",
        "input_kind": "synthetic_fixed_cloud_fields_v1",
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "shader_sha256": hashlib.sha256(shader.encode()).hexdigest(),
        "shader": shader,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                        / "ios/ClaimBodyDeviceHarness/Resources/Shadow/shadow_renderer.json")
    args = parser.parse_args()
    packet = shader_packet(args.source.read_bytes())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in packet.items() if k != "shader"}))


if __name__ == "__main__":
    main()
