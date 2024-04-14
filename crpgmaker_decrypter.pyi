def parse_encrypted_file(
    content: memoryview,
    name: bytearray,
    pos: int,
    key: int
) -> tuple[int, int, int]:
    ...

def decrypt_file_content(content: memoryview, key: int) -> int:
    ...
