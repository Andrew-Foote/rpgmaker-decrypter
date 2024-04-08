def decrypt_file(encrypted: EncryptedFile) -> DecryptedFile:
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
