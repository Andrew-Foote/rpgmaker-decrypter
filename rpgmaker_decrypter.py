import argparse
from dataclasses import dataclass
import itertools as it
from pathlib import Path, PureWindowsPath
import shutil
from typing import Iterator
from crpgmaker_decrypter import parse_encrypted_file, decrypt_file_content

DECRYPT_LIMIT_WHEN_PROFILING = 500
INITIAL_KEY = 0xdeadcafe

class ParseError(Exception):
    pass

@dataclass
class EncryptedFile:
    name: str
    content: memoryview
    key: int

def parse_encrypted_files(content: memoryview) -> Iterator[EncryptedFile]:
    if bytes(content[:8]) != b'RGSSAD\0\1':
        raise ValueError('no RGSSAD version 1 header detected')

    content_len: int = len(content)
    key = INITIAL_KEY
    pos = 8

    while pos < content_len:
        name = bytearray()
        pos, size, key = parse_encrypted_file(content, name, pos, key)
        
        yield EncryptedFile(
            name.decode('utf-8', 'surrogateescape'),
            content[pos:pos + size],
            key
        )

        pos += size
        
@dataclass
class DecryptedFile:
    name: str
    content: bytes

def decrypt_file(encrypted: EncryptedFile) -> DecryptedFile:
    content = encrypted.content
    size = len(content)
    a = bytearray(bytes(content) + b'\0' * (4 - size % 4))
    decrypt_file_content(memoryview(a), encrypted.key)
    return DecryptedFile(encrypted.name, bytes(a[:size]))

def main(
    input_file: Path, output_dir: Path,
    *, overwrite: bool, wipe: bool, profile: bool, verbose: bool
) -> None:
    
    if input_file.suffix != '.rgssad':
        print('Warning: input file does not have the .rgssad extension')

    with input_file.open('rb') as ifh:
        content = memoryview(bytearray(ifh.read()))

    if wipe:
        if input(f'Wipe directory {output_dir}? (y/n) ') == 'y':
            shutil.rmtree(output_dir)
        else:
            print('Decryption aborted')
    
    encrypted_files = parse_encrypted_files(content)

    if profile:
        encrypted_files = it.islice(
            encrypted_files,
            DECRYPT_LIMIT_WHEN_PROFILING
        )

    for ef in encrypted_files:
        if verbose:
            print(f'Decrypting file {ef.name}...')

        # I'm assuming here that the paths will always be given in Windows
        # syntax, since RPG Maker XP doesn't work natively on Linux
        df = decrypt_file(ef)
        dfname = PureWindowsPath(df.name).as_posix()
        path = (output_dir / dfname).absolute()

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
    arg_parser = argparse.ArgumentParser(
        prog='rpgmaker_decrypter',
        description='Decrypts RPG Maker XP game data files.'
    )

    arg_parser.add_argument('input_file', type=Path, help=(
        'the .rgssad file to parse'
    ))

    arg_parser.add_argument('output_dir', type=Path, help=(
        'the output directory to store the decrypted files in'
    ))
    
    arg_parser.add_argument('-o', '--overwrite', action='store_true', help=(
        'overwrite files in the output directory if they have the same name as '
        'one of the decrypted files (redundant if -w is used)'
    ))
    
    arg_parser.add_argument('-w', '--wipe', action='store_true', help=(
        'wipe output directory completely before writing into it'
    ))

    arg_parser.add_argument('-v', '--verbose', action='store_true', help=(
        'print messages to standard output showing which files are being '
        'decrypted'
    ))

    arg_parser.add_argument('-p', '--profile', action='store_true', help=(
        f'only decrypt a limited number of files '
        f'(namely {DECRYPT_LIMIT_WHEN_PROFILING}), to make the program run '
        'quicker when profiling/testing'
    ))

    parsed_args = arg_parser.parse_args()
    
    main(
        parsed_args.input_file, parsed_args.output_dir,
        overwrite=parsed_args.overwrite,
        wipe=parsed_args.wipe,
        profile=parsed_args.profile,
        verbose=parsed_args.verbose
    )