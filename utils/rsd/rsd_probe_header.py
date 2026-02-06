#!/usr/bin/env python3
"""
Probe Garmin `.RSD` *file header* to list channel metadata.

This does NOT rely on PINGMapper/PINGVerter heuristics (like "if 2 channels then
downscan"). Instead it parses the header area described in:
  "Garmin Recorded Sonar Data (RSD) File Format" (Herbert Oppmann, 2024-03-19)

It extracts, per channel:
- channel_id
- first_chunk_offset
- transducer_port
- transducer_freq (mode, start_freq, end_freq)
- channel_capabilities (raw)

Usage:
  python rsd_probe_header.py /path/to/file.RSD
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


def read_varuint(buf: bytes, i: int) -> Tuple[int, int]:
    """Google-protobuf style varint (unsigned)."""
    shift = 0
    val = 0
    while True:
        if i >= len(buf):
            raise ValueError("Unexpected EOF while reading varuint")
        b = buf[i]
        i += 1
        val |= (b & 0x7F) << shift
        if (b & 0x80) == 0:
            return val, i
        shift += 7
        if shift > 63:
            raise ValueError("varuint too large")


def parse_var_struct(buf: bytes, i: int) -> Tuple[Dict[int, List[bytes]], int]:
    """
    Parse Garmin "variable structure" (field_count + fields).
    Returns a dict mapping field_number -> list of raw value bytes.
    """
    nfields, i = read_varuint(buf, i)
    out: Dict[int, List[bytes]] = {}
    for _ in range(nfields):
        key, i = read_varuint(buf, i)
        field_no = key >> 3
        len_code = key & 0x07
        if len_code == 7:
            length, i = read_varuint(buf, i)
        else:
            length = len_code
        if i + length > len(buf):
            raise ValueError("Unexpected EOF while reading field bytes")
        val = buf[i : i + length]
        i += length
        out.setdefault(field_no, []).append(val)
    return out, i


def parse_array(buf: bytes, i: int) -> Tuple[List[bytes], int]:
    """
    Parse Garmin "array": VarUInt32 count, followed by 'count' elements.
    Elements here are returned as raw bytes; the caller must know element type.
    """
    n, i = read_varuint(buf, i)
    elems: List[bytes] = []
    for _ in range(n):
        # For many arrays in header, elements are complex structures with their own
        # serialization. We can't know length here unless the surrounding encoding
        # provides it (varray). So this helper is used only where element sizes are fixed.
        raise NotImplementedError("Use varray parsing helpers for header arrays.")
    return elems, i


def parse_varray_of_length_prefixed_blobs(buf: bytes, i: int) -> Tuple[List[bytes], int]:
    """
    Parse Garmin "varray" where:
      VarUInt32 overall_length_bytes
      VarUInt32 n_elems
      for each elem: VarUInt32 elem_len + elem_bytes

    Returns list of elem_bytes.
    """
    overall_len, i = read_varuint(buf, i)
    end = i + overall_len
    if end > len(buf):
        raise ValueError("varray overall_len beyond buffer")

    n, i = read_varuint(buf, i)
    elems: List[bytes] = []
    for _ in range(n):
        elen, i = read_varuint(buf, i)
        elems.append(buf[i : i + elen])
        i += elen

    # Some varrays may include padding; move to end if needed.
    if i < end:
        i = end
    return elems, i


def parse_len_delimited_varray_of_length_prefixed_blobs(buf: bytes, i: int) -> Tuple[List[bytes], int]:
    """
    Garmin often stores arrays inside an *already length-delimited* field value.
    In that case, the varray does NOT start with overall_len; it starts with:
      VarUInt32 n_elems
      for each elem: VarUInt32 elem_len + elem_bytes
    """
    n, i = read_varuint(buf, i)
    elems: List[bytes] = []
    for _ in range(n):
        elen, i = read_varuint(buf, i)
        elems.append(buf[i : i + elen])
        i += elen
    return elems, i


def parse_data_info_channel_id(data_info_bytes: bytes) -> int:
    """
    DataInfo element is length-prefixed in examples; inside, it's just a VarUInt32 channel_id.
    """
    # In practice this is just a VarUInt32 channel_id stored inside a length-prefixed element.
    cid, _ = read_varuint(data_info_bytes, 0)
    return cid


@dataclass
class ChannelSummary:
    channel_id: int
    first_chunk_offset: int | None
    transducer_port: int | None
    freq_mode: int | None
    start_freq_hz: int | None
    end_freq_hz: int | None
    channel_capabilities: int | None


def parse_header_channels(rsd_path: Path) -> Tuple[Dict[str, int], List[ChannelSummary]]:
    """
    Parse the 0x5000 header area and extract per-channel summaries.
    """
    header_area_size = 0x5000
    buf = rsd_path.read_bytes()[:header_area_size]

    header_struct, i = parse_var_struct(buf, 0)

    # Header has a trailing CRC (uint32 little-endian)
    if i + 4 > len(buf):
        raise ValueError("Header CRC missing")
    _crc = struct.unpack_from("<I", buf, i)[0]
    i += 4

    # Basic header fields
    def u32(field_no: int) -> int | None:
        vs = header_struct.get(field_no)
        if not vs:
            return None
        v = vs[0]
        if len(v) != 4:
            return None
        return struct.unpack("<I", v)[0]

    def u16(field_no: int) -> int | None:
        vs = header_struct.get(field_no)
        if not vs:
            return None
        v = vs[0]
        if len(v) != 2:
            return None
        return struct.unpack("<H", v)[0]

    def u8(field_no: int) -> int | None:
        vs = header_struct.get(field_no)
        if not vs:
            return None
        v = vs[0]
        if len(v) != 1:
            return None
        return v[0]

    header_info = {
        "magic_number": u32(0) or -1,
        "format_version": u16(1) or -1,
        "channel_count": u32(2) or -1,
        "max_channel_count": u8(3) or -1,
    }

    chan_info_vals = header_struct.get(6, [])
    if not chan_info_vals:
        return header_info, []

    # Field 6 is "channel_information_array" encoded as an array[channel_count] of ChannelInformation structures.
    # The field value itself contains: VarUInt32 n_elems, then each ChannelInformation structure serialized inline.
    # The overall length is known from the field container, so we can parse sequentially.
    chan_block = chan_info_vals[0]
    j = 0
    n_chan, j = read_varuint(chan_block, j)

    channels: List[ChannelSummary] = []
    for _ in range(n_chan):
        ch_struct, j = parse_var_struct(chan_block, j)

        # Field 0 data_info is a varray of DataInfo element(s).
        channel_id = -1
        if 0 in ch_struct:
            # ch_struct[0][0] is a length-delimited field value; inside it starts with n_elems.
            elems, _ = parse_len_delimited_varray_of_length_prefixed_blobs(ch_struct[0][0], 0)
            if elems:
                channel_id = parse_data_info_channel_id(elems[0])

        # Field 1 first_chunk_offset is an 8-byte little-endian ulong
        first_chunk_offset = None
        if 1 in ch_struct and ch_struct[1] and len(ch_struct[1][0]) == 8:
            first_chunk_offset = struct.unpack("<Q", ch_struct[1][0])[0]

        transducer_port = None
        freq_mode = None
        start_freq_hz = None
        end_freq_hz = None
        channel_capabilities = None

        # Field 2 prop_chan_info is a varray of DpsChannelInformation structures
        if 2 in ch_struct and ch_struct[2]:
            elems, _ = parse_len_delimited_varray_of_length_prefixed_blobs(ch_struct[2][0], 0)
            if elems:
                dps_struct, _k = parse_var_struct(elems[0], 0)

                # Field 0 transducer_port: VarUInt32 stored in 1-? bytes
                if 0 in dps_struct and dps_struct[0]:
                    transducer_port, _ = read_varuint(dps_struct[0][0], 0)

                # Field 2 channel_capabilities: VarUInt32
                if 2 in dps_struct and dps_struct[2]:
                    channel_capabilities, _ = read_varuint(dps_struct[2][0], 0)

                # Field 1 transducer_freq: nested variable structure
                if 1 in dps_struct and dps_struct[1]:
                    tf_struct, _ = parse_var_struct(dps_struct[1][0], 0)
                    if 0 in tf_struct and tf_struct[0]:
                        freq_mode, _ = read_varuint(tf_struct[0][0], 0)
                    if 1 in tf_struct and tf_struct[1]:
                        start_freq_hz, _ = read_varuint(tf_struct[1][0], 0)
                    if 2 in tf_struct and tf_struct[2]:
                        end_freq_hz, _ = read_varuint(tf_struct[2][0], 0)

        channels.append(
            ChannelSummary(
                channel_id=channel_id,
                first_chunk_offset=first_chunk_offset,
                transducer_port=transducer_port,
                freq_mode=freq_mode,
                start_freq_hz=start_freq_hz,
                end_freq_hz=end_freq_hz,
                channel_capabilities=channel_capabilities,
            )
        )

    return header_info, channels


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python rsd_probe_header.py /path/to/file.RSD", file=sys.stderr)
        return 2

    rsd_path = Path(argv[1]).expanduser().resolve()
    if not rsd_path.exists():
        print(f"Not found: {rsd_path}", file=sys.stderr)
        return 2

    header_info, chans = parse_header_channels(rsd_path)
    print("Header:")
    for k, v in header_info.items():
        print(f"  {k}: {v}")

    print("\nChannels:")
    if not chans:
        print("  (none parsed)")
        return 0

    for c in chans:
        freq = None
        if c.start_freq_hz is not None or c.end_freq_hz is not None:
            freq = f"{c.start_freq_hz}-{c.end_freq_hz} Hz (mode={c.freq_mode})"
        print(
            "  "
            + f"channel_id={c.channel_id} "
            + f"first_chunk_offset={c.first_chunk_offset} "
            + f"transducer_port={c.transducer_port} "
            + f"freq={freq} "
            + f"capabilities={c.channel_capabilities}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

