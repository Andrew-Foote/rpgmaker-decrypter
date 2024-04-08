# Ported from the C# implementation at
# https://github.com/uuksu/RPGMakerDecrypter/tree/master.

# rewrite in cython?

from dataclasses import dataclass
from functools import partial
import itertools as it
from pathlib import Path
import numpy as np
import numpy.typing as npt
from typing import Iterator
import cython as c
from cython.cimports.libc.stdint import uint8_t, int32_t, uint32_t

RGSSAD_V1_KEY: uint32_t = 0xdeadcafe

@c.cfunc
def next_key(key: uint32_t) -> uint32_t:
    return key * 7 + 3

class ParseError(Exception):
    pass

@c.cfunc
def read_int(content: bytes, pos: c.int) -> tuple[int32_t, c.int]:
    if len(content) < pos + 4:
        raise ParseError(
            'reached end of file unexpectedly; file may be corrupt'
        )

    z = np.frombuffer(content[pos:pos + 4], dtype=np.dtype('<i4'))

    return (
        z[0],
        pos + 4
    )

@c.cfunc
def read_byte(content: bytes, pos: c.int) -> tuple[uint8_t, c.int]:
    if len(content) < pos:
        raise ParseError(
            'reached end of file unexpectedly; file may be corrupt'
        )

    return content[pos], pos + 1

@c.cfunc
def decrypt_int(value: int32_t, key: uint32_t) -> tuple[int32_t, uint32_t]:
    return value ^ key, next_key(key)

@c.cfunc
def decrypt_byte(value: uint8_t, key: uint32_t) -> tuple[uint8_t, uint32_t]:
    return value ^ (key & 0xff), next_key(key)

@c.cfunc
def parse_int(
    content: bytes, pos: c.int, key: uint32_t
) -> tuple[int32_t, c.int, uint32_t]:

    encrypted, pos = read_int(content, pos)
    result, key = decrypt_int(encrypted, key)
    return result, pos, key

@c.cfunc
def parse_byte(
    content: bytes, pos: c.int, key: uint32_t
) -> tuple[uint8_t, c.int, uint32_t]:
    
    encrypted, pos = read_byte(content, pos)
    result, key = decrypt_byte(encrypted, key)
    return result, pos, key

@dataclass
class EncryptedFile:
    name: str
    content: bytes
    key: uint32_t

def parse_encrypted_files(content: bytes) -> Iterator[EncryptedFile]:
    header = content[:8]

    if header != b'RGSSAD\0\1':
        raise ValueError('no RGSSAD version 1 header detected')

    key = RGSSAD_V1_KEY
    pos: c.int = 8

    while pos < len(content):
        length, pos, key = parse_int(content, pos, key)
        name = bytearray()

        for _ in range(length):
            char, pos, key = parse_byte(content, pos, key)
            name.append(char)

        size, pos, key = parse_int(content, pos, key)
        file_content = content[pos:pos + size]
        pos += int(size)
        
        yield EncryptedFile(
            name.decode('utf-8', 'surrogateescape'),
            file_content,
            key
        )

@dataclass
class DecryptedFile:
    name: str
    content: bytes

@c.cfunc
def decrypt_bytearray(array: bytearray, key: uint32_t) -> c.int:
    pos: c.int

    for pos in range(0, len(array), 4):
        array[pos    ] ^= key        
        array[pos + 1] ^= (key >>  8)
        array[pos + 2] ^= (key >> 16)
        array[pos + 3] ^= (key >> 24)
        key = key * 7 + 3

    return 0

def decrypt_file(encrypted: EncryptedFile) -> DecryptedFile:
    encrypted_content = encrypted.content
    size = len(encrypted_content)
    
    decrypted_content = bytearray(
        encrypted_content + bytes((0,) * (4 - size % 4))
    )

    decrypt_bytearray(decrypted_content, encrypted.key)
    return DecryptedFile(encrypted.name, bytes(decrypted_content[:size]))

LIMIT_WHEN_PROFILING = 100

def main(
    input_file: Path, output_dir: Path,
    *, overwrite: bool, profile: bool, verbose: bool
) -> None:
    
    if input_file.suffix != '.rgssad':
        print('Warning: input file does not have the .rgssad extension')

    with input_file.open('rb') as ifh:
        content = ifh.read()
    
    encrypted_files = parse_encrypted_files(content)

    if profile:
        encrypted_files = it.islice(encrypted_files, LIMIT_WHEN_PROFILING)

    for ef in encrypted_files:
        if verbose:
            print(f'Decrypting file {ef.name} (key: {hex(ef.key)})...')

        df = decrypt_file(ef)
        path = (output_dir / df.name).absolute()

        if not overwrite and path.exists():
            print(
                f'Path {path} already exists. Use the -o option to overwrite '
                'existing files.'
            )
            
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('wb') as ofh:
            ofh.write(df.content)
