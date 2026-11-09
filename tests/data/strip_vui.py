"""Remove the VUI from the SPS of an H.264 Annex B stream.

x264 always writes a VUI with a bitstream restriction, so a stream without
`max_num_reorder_frames` has to be made from one that has it. The decoder then
has to infer the reorder depth (see `Sps::reorder_depth`).

Usage:
    python3 strip_vui.py in.264 out.264

    novui.264:      from bframes.264 (reorder depth from the level)
    novui_poc2.264: from zerolatency.264 (POC type 2, reorder depth 0)

Handles what x264 writes for 4:2:0 frame-coded streams: no scaling matrices.
"""

import sys


class Bits:
    def __init__(self, data):
        self.bits = "".join(f"{b:08b}" for b in data)
        self.pos = 0

    def u(self, n):
        value = int(self.bits[self.pos:self.pos + n], 2)
        self.pos += n
        return value

    def ue(self):
        zeros = 0
        while self.bits[self.pos] == "0":
            zeros += 1
            self.pos += 1
        self.pos += 1
        return (1 << zeros) - 1 + (self.u(zeros) if zeros else 0)


def unescape(data):
    out = bytearray()
    zeros = 0
    for b in data:
        if zeros >= 2 and b == 3:
            zeros = 0
            continue
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def escape(data):
    out = bytearray()
    zeros = 0
    for b in data:
        if zeros >= 2 and b <= 3:
            out.append(3)
            zeros = 0
        out.append(b)
        zeros = zeros + 1 if b == 0 else 0
    return bytes(out)


def strip(sps):
    """`sps` is the NAL unit with its header byte, emulation bytes included."""
    r = Bits(unescape(sps[1:]))
    profile = r.u(8)
    r.u(16)
    r.ue()
    if profile in (100, 110, 122, 244, 44, 83, 86, 118, 128, 138, 139, 134, 135):
        chroma = r.ue()
        assert chroma != 3, "4:4:4 is not handled"
        r.ue()
        r.ue()
        r.u(1)
        assert r.u(1) == 0, "scaling matrices are not handled"
    r.ue()
    poc_type = r.ue()
    if poc_type == 0:
        r.ue()
    else:
        assert poc_type == 2, "POC type 1 is not handled"
    r.ue()
    r.u(1)
    r.ue()
    r.ue()
    if r.u(1) == 0:
        r.u(1)
    r.u(1)
    if r.u(1):
        for _ in range(4):
            r.ue()
    assert r.u(1) == 1, "the SPS has no VUI"
    # vui_parameters_present_flag = 0, then the RBSP stop bit and alignment.
    bits = r.bits[:r.pos - 1] + "0" + "1"
    bits += "0" * (-len(bits) % 8)
    rbsp = bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))
    return sps[:1] + escape(rbsp)


def main():
    data = open(sys.argv[1], "rb").read()
    # Split at start codes, keeping each one with the NAL unit that follows.
    starts = []
    i = 0
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0:
            break
        starts.append(i)
        i += 3
    out = bytearray(data[:starts[0]])
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(data)
        nal = data[start + 3:end]
        # A four-byte start code leaves its leading zero on the previous unit.
        tail = b""
        if n + 1 < len(starts) and nal.endswith(b"\x00"):
            nal, tail = nal[:-1], b"\x00"
        if nal[0] & 0x1F == 7:
            nal = strip(nal)
        out += b"\x00\x00\x01" + nal + tail
    open(sys.argv[2], "wb").write(out)


if __name__ == "__main__":
    main()
