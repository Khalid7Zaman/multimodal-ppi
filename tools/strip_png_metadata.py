#!/usr/bin/env python
"""Strip ALL metadata from PNG figures, keeping only the chunks that define the image.

Standard figure-preparation step before release: it removes PNG text chunks (tEXt / zTXt / iTXt)
and any non-standard ancillary chunks, so no software tag, tool name, or provenance text remains in
the file. The image pixels (IHDR / PLTE / IDAT) are copied through untouched, so the picture is
identical; only hidden metadata is removed. Pure standard library, no dependencies.

Usage:
    python tools/strip_png_metadata.py reports/figures/*.png
"""
import sys, struct

# Chunks that define how the image renders - kept. Everything else (text + custom) is dropped.
KEEP = {b"IHDR", b"PLTE", b"IDAT", b"IEND", b"tRNS", b"gAMA", b"cHRM",
        b"sRGB", b"iCCP", b"sBIT", b"bKGD", b"hIST", b"pHYs", b"sPLT"}
SIG = b"\x89PNG\r\n\x1a\n"


def strip(path):
    data = open(path, "rb").read()
    if data[:8] != SIG:
        print(f"{path}: not a PNG, skipped")
        return
    out = bytearray(SIG)
    i, dropped = 8, []
    while i < len(data):
        (length,) = struct.unpack(">I", data[i:i + 4])
        ctype = data[i + 4:i + 8]
        if ctype in KEEP:
            out += data[i:i + 12 + length]
        else:
            dropped.append(ctype.decode("latin1"))
        i += 12 + length
    open(path, "wb").write(out)
    print(f"{path}: dropped {dropped or 'nothing'}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        strip(p)
    print("done")
