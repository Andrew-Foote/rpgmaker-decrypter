import cython as c
from cython.imports.libc.stdint import uint32_t

def parse_encrypted_file(
    content: c.char[:],
    name: bytearray,
    pos: c.int,
    key: uint32_t
) -> tuple[c.int, c.int, uint32_t]:
    ...

def decrypt_file_content(content: c.char[:], key: uint32_t) -> c.int:
    ...
