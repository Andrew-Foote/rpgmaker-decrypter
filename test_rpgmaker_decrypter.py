# The 'example/expected-output' folder contains the output from the C# 
# implementation run on the file 'example/input.rgssad'. We test that the result
# from our implementation matches this output.

from pathlib import Path
import random
from typing import Iterator
import pytest
from rpgmaker_decrypter import decrypt_file, parse_encrypted_files

EXPECTED_OUTPUT_PATH = Path('example/expected-output')

@pytest.fixture(scope='session')
def input_content() -> Iterator[bytes]:
    with Path('example/input.rgssad').open('rb') as ifh:
        yield memoryview(bytearray(ifh.read()))

def test_filenames_match(input_content: memoryview) -> None:
    actual_filenames = {ef.name for ef in parse_encrypted_files(input_content)}

    expected_filenames = {
        str(path.relative_to(EXPECTED_OUTPUT_PATH))
        for path in EXPECTED_OUTPUT_PATH.glob('**/*') 
        if not path.is_dir()
    }

    assert actual_filenames == expected_filenames

def test_contents_match(input_content: memoryview) -> None:
    numfiles = sum(1 for _ in parse_encrypted_files(input_content))
    sample = random.sample(range(numfiles), 100)

    for i, ef in enumerate(parse_encrypted_files(input_content)):
       if i in sample:
            with (EXPECTED_OUTPUT_PATH / ef.name).open('rb') as efh:
                expected_content = efh.read()

            actual_content = decrypt_file(ef).content
            assert actual_content == expected_content
    