import sys
from typing import List

IP = [
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
]

FP = [
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25
]

E = [
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
]

P = [
    16, 7, 20, 21,
    29, 12, 28, 17,
    1, 15, 23, 26,
    5, 18, 31, 10,
    2, 8, 24, 14,
    32, 27, 3, 9,
    19, 13, 30, 6,
    22, 11, 4, 25
]

PC1 = [
    57, 49, 41, 33, 25, 17, 9,
    1, 58, 50, 42, 34, 26, 18,
    10, 2, 59, 51, 43, 35, 27,
    19, 11, 3, 60, 52, 44, 36,
    63, 55, 47, 39, 31, 23, 15,
    7, 62, 54, 46, 38, 30, 22,
    14, 6, 61, 53, 45, 37, 29,
    21, 13, 5, 28, 20, 12, 4
]

PC2 = [
    14, 17, 11, 24, 1, 5,
    3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8,
    16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
]

SHIFTS = [1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1]

SBOX = [
    # S1
    [
        [14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
        [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
        [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
        [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
    ],
    # S2
    [
        [15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
        [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
        [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
        [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
    ],
    # S3
    [
        [10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
        [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
        [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
        [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
    ],
    # S4
    [
        [7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
        [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
        [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
        [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
    ],
    # S5
    [
        [2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
        [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
        [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
        [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
    ],
    # S6
    [
        [12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
        [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
        [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
        [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
    ],
    # S7
    [
        [4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
        [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
        [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
        [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
    ],
    # S8
    [
        [13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
        [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
        [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
        [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
    ],
]

MASK64 = (1 << 64) - 1


def permute(inp: int, tbl: List[int], n: int, in_bits: int) -> int:
    out = 0
    for i in range(n):
        src_pos = tbl[i]
        bit = (inp >> (in_bits - src_pos)) & 1
        out = (out << 1) | bit
    return out & ((1 << n) - 1)


def rol28(x: int, s: int) -> int:
    x &= 0x0FFFFFFF
    return ((x << s) | (x >> (28 - s))) & 0x0FFFFFFF


def bytes_to_u64_be(b: bytes, off: int) -> int:
    x = 0
    for i in range(8):
        x = (x << 8) | b[off + i]
    return x & MASK64


def u64_to_bytes_be(x: int) -> bytes:
    x &= MASK64
    out = bytearray(8)
    for i in range(7, -1, -1):
        out[i] = x & 0xFF
        x >>= 8
    return bytes(out)


def hex_u64(x: int) -> str:
    x &= MASK64
    hexchars = "0123456789ABCDEF"
    s = ["0"] * 16
    for i in range(15, -1, -1):
        s[i] = hexchars[x & 0xF]
        x >>= 4
    return "".join(s)


def make_subkeys(key64: int) -> List[int]:
    key64 &= MASK64
    key56 = permute(key64, PC1, 56, 64)
    C = (key56 >> 28) & 0x0FFFFFFF
    D = (key56 >> 0) & 0x0FFFFFFF

    sub = [0] * 16
    for r in range(16):
        C = rol28(C, SHIFTS[r])
        D = rol28(D, SHIFTS[r])
        CD = ((C << 28) | D) & ((1 << 56) - 1)
        sub[r] = permute(CD, PC2, 48, 56)
    return sub


def feistel(R: int, K48: int) -> int:
    R &= 0xFFFFFFFF
    ER = permute(R, E, 48, 32)
    x = ER ^ (K48 & ((1 << 48) - 1))

    out32 = 0
    for i in range(8):
        chunk = (x >> (42 - 6 * i)) & 0x3F
        row = ((chunk & 0x20) >> 4) | (chunk & 0x01)
        col = (chunk >> 1) & 0x0F
        sval = SBOX[i][row][col]
        out32 = ((out32 << 4) | sval) & 0xFFFFFFFF

    out32 = permute(out32, P, 32, 32) & 0xFFFFFFFF
    return out32


def des_encrypt_block(block: int, sub: List[int]) -> int:
    block &= MASK64
    ip = permute(block, IP, 64, 64)
    L = (ip >> 32) & 0xFFFFFFFF
    R = ip & 0xFFFFFFFF

    for i in range(16):
        newL = R
        newR = (L ^ feistel(R, sub[i])) & 0xFFFFFFFF
        L, R = newL, newR

    preout = ((R << 32) | L) & MASK64
    return permute(preout, FP, 64, 64) & MASK64


def des_decrypt_block(block: int, sub: List[int]) -> int:
    block &= MASK64
    ip = permute(block, IP, 64, 64)
    L = (ip >> 32) & 0xFFFFFFFF
    R = ip & 0xFFFFFFFF

    for i in range(15, -1, -1):
        newL = R
        newR = (L ^ feistel(R, sub[i])) & 0xFFFFFFFF
        L, R = newL, newR

    preout = ((R << 32) | L) & MASK64
    return permute(preout, FP, 64, 64) & MASK64


def read_file(path: str) -> bytearray:
    try:
        with open(path, "rb") as f:
            return bytearray(f.read())
    except OSError:
        raise RuntimeError("Cannot open input file: " + path)


def write_file(path: str, data: bytes) -> None:
    try:
        with open(path, "wb") as f:
            f.write(data)
    except OSError:
        raise RuntimeError("Cannot open output file: " + path)


def pkcs7_pad(data: bytearray) -> None:
    pad = 8 - (len(data) % 8)
    if pad == 0:
        pad = 8
    data.extend([pad] * pad)


def pkcs7_unpad(data: bytearray) -> None:
    if len(data) == 0 or (len(data) % 8) != 0:
        raise RuntimeError("Bad padding size")
    pad = data[-1]
    if pad < 1 or pad > 8:
        raise RuntimeError("Bad padding value")
    for i in range(pad):
        if data[len(data) - 1 - i] != pad:
            raise RuntimeError("Bad padding bytes")
    del data[-pad:]


def is_hex_string(s: str) -> bool:
    for c in s:
        if not (("0" <= c <= "9") or ("a" <= c <= "f") or ("A" <= c <= "F")):
            return False
    return True


def parse_key64(raw: str) -> int:
    s = raw.strip()
    if s.startswith("0x") or s.startswith("0X"):
        s = s[2:]

    if len(s) == 16 and is_hex_string(s):
        key = 0
        for c in s:
            key <<= 4
            if "0" <= c <= "9":
                key |= (ord(c) - ord("0"))
            elif "a" <= c <= "f":
                key |= (ord(c) - ord("a") + 10)
            else:
                key |= (ord(c) - ord("A") + 10)
        return key & MASK64

    a = s
    if len(a) < 8:
        a = a + ("\0" * (8 - len(a)))
    if len(a) > 8:
        a = a[:8]

    key = 0
    for i in range(8):
        key = (key << 8) | (ord(a[i]) & 0xFF)
    return key & MASK64


def mode_encrypt_ecb(sub: List[int]) -> None:
    in_file = input("Input file to encrypt: ")
    out_file = input("Output encrypted file: ")

    data = read_file(in_file)
    pkcs7_pad(data)

    out = bytearray(len(data))
    for i in range(0, len(data), 8):
        b = bytes_to_u64_be(data, i)
        c = des_encrypt_block(b, sub)
        out[i:i+8] = u64_to_bytes_be(c)

    write_file(out_file, bytes(out))
    print(f"[OK] Encrypted -> {out_file}")


def mode_decrypt_ecb(sub: List[int]) -> None:
    in_file = input("Input file to decrypt: ")
    out_file = input("Output decrypted file: ")

    data = read_file(in_file)
    if (len(data) % 8) != 0:
        raise RuntimeError("Ciphertext size is not multiple of 8")

    out = bytearray(len(data))
    for i in range(0, len(data), 8):
        b = bytes_to_u64_be(data, i)
        p = des_decrypt_block(b, sub)
        out[i:i+8] = u64_to_bytes_be(p)

    pkcs7_unpad(out)
    write_file(out_file, bytes(out))
    print(f"[OK] Decrypted -> {out_file}")


def mode_signature_cbc(sub: List[int]) -> None:
    in_file = input("Input file to sign: ")
    sig_file = input("Output signature file (hex): ")

    data = read_file(in_file)
    pkcs7_pad(data)

    iv = 0
    for i in range(0, len(data), 8):
        b = bytes_to_u64_be(data, i)
        x = (b ^ iv) & MASK64
        iv = des_encrypt_block(x, sub)

    sig_hex = hex_u64(iv)
    print(f"Signature (CBC-MAC, hex): {sig_hex}")

    try:
        with open(sig_file, "w", encoding="utf-8") as f:
            f.write(sig_hex)
    except OSError:
        raise RuntimeError("Cannot write signature file")

    print(f"[OK] Signature saved to: {sig_file}")


def main() -> int:
    try:
        print("=== Lab 4: DES encryption/decryption + CBC signature ===")
        print("Modes:")
        print("1 - Encrypt file (DES, ECB)")
        print("2 - Decrypt file (DES, ECB)")
        print("3 - Create digital signature (DES CBC-MAC)")
        mode_str = input("Choose mode: ").strip()
        if mode_str == "":
            return 0
        mode = int(mode_str)

        key_str = input("Enter DES key (8 ASCII chars OR 16 hex digits): ")
        key64 = parse_key64(key_str)

        sub = make_subkeys(key64)

        if mode == 1:
            mode_encrypt_ecb(sub)
        elif mode == 2:
            mode_decrypt_ecb(sub)
        elif mode == 3:
            mode_signature_cbc(sub)
        else:
            print("Unknown mode.")
        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
