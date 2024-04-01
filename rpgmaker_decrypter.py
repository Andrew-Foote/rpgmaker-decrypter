# Ported from the C# implementation at
# https://github.com/uuksu/RPGMakerDecrypter/tree/master.

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from typing import Iterator

RGSSAD_V1_KEY = np.uint32(0xdeadcafe)

THREE = np.uint32(3)
SEVEN = np.uint32(7)

def next_key(key: np.uint32) -> np.uint32:
    return key * SEVEN + THREE

class ParseError(Exception):
    pass

def read_int(content: bytes, pos: int) -> tuple[np.int32, int]:
    if len(content) < pos + 4:
        raise ParseError(
            'reached end of file unexpectedly; file may be corrupt'
        )

    return (
        np.frombuffer(content[pos:pos + 4], dtype=np.dtype('<u4'))[0],
        pos + 4
    )

def read_byte(content: bytes, pos: int) -> tuple[np.uint8, int]:
    if len(content) < pos:
        raise ParseError(
            'reached end of file unexpectedly; file may be corrupt'
        )

    return np.uint8(content[pos]), pos + 1

def decrypt_int(value: np.int32, key: np.uint32) -> tuple[np.int32, np.uint32]:
    return value ^ key, next_key(key)

def decrypt_byte(value: np.uint8, key: np.uint32) -> tuple[np.uint8, np.uint32]:
    return value ^ np.uint8(key), next_key(key)

def parse_int(
    content: bytes, pos: int, key: np.uint32
) -> tuple[np.int32, int, np.uint32]:

    encrypted, pos = read_int(content, pos)
    result, key = decrypt_int(encrypted, key)
    return result, pos, key

def parse_byte(
    content: bytes, pos: int, key: np.uint32
) -> tuple[np.uint8, int, np.uint32]:
    
    encrypted, pos = read_byte(content, pos)
    result, key = decrypt_byte(encrypted, key)
    return result, pos, key

def parse_byte_fast(content: bytes, pos: int, key: np.uint32) -> np.uint8:
    encrypted, pos = read_byte(content, pos)
    return encrypted ^ np.uint8(key)

@dataclass
class EncryptedFile:
    name: str
    content: bytes
    key: np.uint32

def parse_encrypted_files(content: bytes) -> Iterator[EncryptedFile]:
    header = content[:8]

    if header != b'RGSSAD\0\1':
        raise ValueError('no RGSSAD version 1 header detected')

    key = RGSSAD_V1_KEY
    pos = 8

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

def decrypt_file(encrypted: EncryptedFile) -> DecryptedFile:
    # encrypted_content = encrypted.content
    # size = len(encrypted_content)
    # nwords = size // 4 + (size % 4 > 0)
    
    # keys = [encrypted.key]

    # for _ in range(nwords - 1):
    #     keys.append(next_key(key))

    

    # keys = np.ndarray([key * 7 + 3])

    encrypted_content = encrypted.content
    decrypted_content = bytearray()
    pos = 0
    key = encrypted.key

    while pos < len(encrypted.content):
        char, pos, _ = parse_byte(
            encrypted_content, pos,
            key >> np.uint32((pos % 4) * 8)
        )

        if pos % 4 == 0:
            key = next_key(key)

        decrypted_content.append(char)

    return DecryptedFile(encrypted.name, bytes(decrypted_content))

def main(
    input_file: Path, output_dir: Path,
    *, verbose: bool, overwrite: bool
) -> None:
    
    if input_file.suffix != '.rgssad':
        print('Warning: input file does not have the .rgssad extension')

    with input_file.open('rb') as ifh:
        content = ifh.read()

    encrypted_files = parse_encrypted_files(content)
    
    import itertools as it

    for ef in it.islice(encrypted_files, 10):
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

if __name__ == '__main__':
    import argparse
    np.seterr(over='ignore')

    arg_parser = argparse.ArgumentParser(
        prog='rpgmaker_decrypter',
        description='Decrypts RPG Maker XP game data files.'
    )

    arg_parser.add_argument('input_file', type=Path)
    arg_parser.add_argument('output_dir', type=Path)
    arg_parser.add_argument('-v', '--verbose', action='store_true')
    arg_parser.add_argument('-o', '--overwrite', action='store_true')
    parsed_args = arg_parser.parse_args()
    
    main(
        parsed_args.input_file, parsed_args.output_dir,
        verbose=parsed_args.verbose,
        overwrite=parsed_args.overwrite
    )