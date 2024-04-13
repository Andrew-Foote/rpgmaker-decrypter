import struct
import cython as c
from cython.cimports.libc.stdint import int32_t, uint32_t

def parse_encrypted_file(
    content: c.char[:],
    name: bytearray,
    pos: c.int,
    key: uint32_t
) -> tuple[c.int, c.int, uint32_t, c.int]:

    name_len: int32_t = struct.unpack('<i', content[pos:pos + 4])[0]
    name_len ^= key
    pos += 4
    key = key * 7 + 3

    for _ in range(name_len):
        z: c.char = content[pos] 
        z ^= key
        name.append(z)
        pos += 1
        key = key * 7 + 3

    size: int32_t = struct.unpack('<i', content[pos:pos + 4])[0]
    size ^= key
    pos += 4
    key = key * 7 + 3

    return pos, size, key

def decrypt_file_content(content: c.char[:], key: uint32_t) -> c.int:
    pos: c.int

    for pos in range(0, len(content), 4):
        content[pos    ] ^= key         
        content[pos + 1] ^= (key >>  8)
        content[pos + 2] ^= (key >> 16) 
        content[pos + 3] ^= (key >> 24)
        key = key * 7 + 3

    return 0