# Ported from the C# implementation at
# https://github.com/uuksu/RPGMakerDecrypter/tree/master.

# rewrite in cython?

from dataclasses import dataclass
import itertools as it
from pathlib import Path
import struct
import numpy as np
from typing import Iterator
import cython as c
from cython.cimports.libc.stdint import int32_t, uint32_t

RGSSAD_V1_KEY: uint32_t = 0xdeadcafe

class ParseError(Exception):
    pass

@dataclass
class EncryptedFile:
    name: str
    content: bytes
    key: uint32_t

def parse_encrypted_files(content: bytes) -> Iterator[EncryptedFile]:
    header = content[:8]
    contentlen = len(content)

    if header != b'RGSSAD\0\1':
        raise ValueError('no RGSSAD version 1 header detected')

    key: uint32_t = RGSSAD_V1_KEY
    pos: c.int = 8

    while pos < contentlen:
        length: int32_t = struct.unpack('<i', content[pos:pos + 4])[0]
        length ^= key
        pos += 4
        key = key * 7 + 3

        if pos + length > contentlen:
            raise ParseError('corrupt file')

        name: c.char[:] = bytearray(content[pos:pos + length])

        for i in range(length):
            name[i] ^= key
            pos += 1
            key = key * 7 + 3

        size: int32_t = struct.unpack('<i', content[pos:pos + 4])[0]
        size ^= key
        pos += 4
        key = key * 7 + 3

        if pos + size > contentlen:
            raise ParseError('corrupt file')

        yield EncryptedFile(
            bytes(name).decode('utf-8', 'surrogateescape'),
            content[pos:pos + size],
            key
        )

        pos += size

@dataclass
class DecryptedFile:
    name: str
    content: bytes

@c.cfunc
def decrypt_array(array: c.char[:], key: uint32_t) -> c.int:
    pos: c.int

    for pos in range(0, len(array), 4):
        array[pos    ] ^= key         
        array[pos + 1] ^= (key >>  8)
        array[pos + 2] ^= (key >> 16) 
        array[pos + 3] ^= (key >> 24)
        key = key * 7 + 3

    return 0

def decrypt_file(encrypted: EncryptedFile) -> DecryptedFile:
    content = encrypted.content
    size = len(content)
    a: c.char[:] = bytearray(content + b'\0' * (4 - size % 4))
    decrypt_array(a, encrypted.key)
    return DecryptedFile(encrypted.name, bytes(a[:size]))

LIMIT_WHEN_PROFILING = 500

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
