import os
import sys
import math
import time
from dataclasses import dataclass
from typing import List, Tuple


def add_mod(a: int, b: int, m: int) -> int:
    if m <= 0:
        raise ValueError("mod must be > 0")
    r = (a + b) % m
    if r < 0:
        r += m
    return r


def mul_mod(a: int, b: int, m: int) -> int:
    if m <= 0:
        raise ValueError("mod must be > 0")
    r = (a * b) % m
    if r < 0:
        r += m
    return r


def square_mod(a: int, m: int) -> int:
    if m <= 0:
        raise ValueError("mod must be > 0")
    r = (a * a) % m
    if r < 0:
        r += m
    return r


def modexp(base: int, exp: int, mod: int) -> int:
    if mod <= 0:
        raise ValueError("mod must be > 0")
    if exp < 0:
        raise ValueError("exp must be >= 0")

    base %= mod
    if base < 0:
        base += mod

    result = 1 % mod
    while exp > 0:
        if (exp & 1) != 0:
            result = (result * base) % mod
        base = (base * base) % mod
        exp >>= 1

    if result < 0:
        result += mod
    return result


def gcd_cppint(a: int, b: int) -> int:
    while b != 0:
        t = a % b
        a = b
        b = t
    return -a if a < 0 else a


def egcd(a: int, b: int) -> Tuple[int, int, int]:
    if b == 0:
        return a, 1, 0
    g, x1, y1 = egcd(b, a % b)
    x = y1
    y = x1 - (a // b) * y1
    return g, x, y


def modinv(a: int, m: int) -> int:
    aa = (a % m + m) % m
    g, x, _y = egcd(aa, m)
    if g != 1:
        raise RuntimeError("mod inverse does not exist")
    r = x % m
    if r < 0:
        r += m
    return r


def dec_digits(x: int) -> int:
    t = -x if x < 0 else x
    return len(str(t))


class RNG:
    def __init__(self) -> None:
        seed = int.from_bytes(os.urandom(8), "big", signed=False)
        if seed == 0:
            seed = time.time_ns() & ((1 << 64) - 1)
        self._state = seed & ((1 << 64) - 1)

    def u64(self) -> int:
        x = self._state
        x ^= (x >> 12) & ((1 << 64) - 1)
        x ^= (x << 25) & ((1 << 64) - 1)
        x ^= (x >> 27) & ((1 << 64) - 1)
        self._state = x & ((1 << 64) - 1)
        return (x * 2685821657736338717) & ((1 << 64) - 1)


RNG_GLOBAL = RNG()


def random_bits(bits: int) -> int:
    if bits == 0:
        return 0
    words = (bits + 63) // 64
    x = 0
    for _ in range(words):
        x <<= 64
        x += RNG_GLOBAL.u64()

    extra = words * 64 - bits
    if extra > 0:
        x >>= extra

    x |= (1 << (bits - 1))
    return x


def random_odd_with_digits(digits: int) -> int:
    LOG2_10 = 3.32192809488736234787
    bits = max(4, int(math.ceil(digits * LOG2_10)))
    x = random_bits(bits)
    if (x & 1) == 0:
        x += 1
    return x


SMALL_PRIMES: List[int] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29,
    31, 37, 41, 43, 47, 53, 59, 61, 67, 71,
    73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
    283, 293, 307, 311, 313, 317, 331, 337, 347, 349,
    353, 359, 367, 373, 379, 383, 389, 397
]


def quick_small_prime_screen(n: int) -> bool:
    if n < 2:
        return False
    for p in SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return n == p
    return True


def fermat_test(n: int, bases: List[int]) -> bool:
    if n < 4:
        return n == 2 or n == 3
    for a in bases:
        if a <= 1:
            continue
        if (a % n) == 0:
            continue
        if modexp(a, n - 1, n) != 1:
            return False
    return True


def miller_rabin_round(n: int, a: int, d: int, r: int) -> bool:
    x = modexp(a % n, d, n)
    if x == 1 or x == n - 1:
        return True
    for _i in range(1, r):
        x = (x * x) % n
        if x == n - 1:
            return True
    return False


def is_probable_prime_mr(n: int, rounds: int = 16) -> bool:
    if n < 2:
        return False
    if (n & 1) == 0:
        return n == 2

    d = n - 1
    r = 0
    while (d & 1) == 0:
        d >>= 1
        r += 1

    FIXED_BASES = [2, 3, 5, 7, 11, 13, 17]
    used = 0

    for fb in FIXED_BASES:
        if fb >= n:
            break
        if not miller_rabin_round(n, fb, d, r):
            return False
        used += 1
        if used >= rounds:
            return True

    for _ in range(used, rounds):
        a = RNG_GLOBAL.u64()
        if a < 2:
            a = 2
        a = 2 + (a % (n - 3))
        if not miller_rabin_round(n, a, d, r):
            return False

    return True


def is_probable_prime(n: int, mr_rounds: int = 16) -> bool:
    if not quick_small_prime_screen(n):
        return False
    if not fermat_test(n, [2, 3, 5, 7, 11]):
        return False
    return is_probable_prime_mr(n, mr_rounds)


def generate_prime_with_digits(digits: int, mr_rounds: int = 16) -> int:
    if digits < 2:
        digits = 2
    while True:
        cand = random_odd_with_digits(digits)
        while dec_digits(cand) < digits:
            cand = (cand << 1) | 1

        divisible_small = False
        for p in SMALL_PRIMES:
            if cand % p == 0:
                divisible_small = True
                break
        if divisible_small:
            continue

        if is_probable_prime(cand, mr_rounds):
            return cand


@dataclass
class RSAKeyPair:
    p: int
    q: int
    n: int
    phi: int
    e: int
    d: int


def generate_rsa(prime_digits: int, mr_rounds: int = 16, e: int = 65537) -> RSAKeyPair:
    while True:
        p = generate_prime_with_digits(prime_digits, mr_rounds)
        q = generate_prime_with_digits(prime_digits, mr_rounds)
        if p == q:
            continue

        n = p * q
        phi = (p - 1) * (q - 1)
        if gcd_cppint(e, phi) != 1:
            continue

        d = modinv(e, phi)
        return RSAKeyPair(p=p, q=q, n=n, phi=phi, e=e, d=d)


def rsa_encrypt(m: int, e: int, n: int) -> int:
    if m >= n:
        raise ValueError("message m must be < n")
    return modexp(m, e, n)


def rsa_decrypt(c: int, d: int, n: int) -> int:
    return modexp(c, d, n)


def num_bytes(x: int) -> int:
    if x == 0:
        return 1
    t = -x if x < 0 else x
    bytes_cnt = 0
    while t > 0:
        t >>= 8
        bytes_cnt += 1
    return bytes_cnt


def read_file(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        raise RuntimeError("Cannot open input file: " + path)


def write_file(path: str, data: bytes) -> None:
    try:
        with open(path, "wb") as f:
            f.write(data)
    except OSError:
        raise RuntimeError("Cannot open output file: " + path)


def encrypt_file_rsa(in_path: str, out_path: str, n: int, e: int) -> None:
    plain = read_file(in_path)
    orig_size = len(plain)

    mod_bytes = num_bytes(n)
    if mod_bytes < 2:
        raise RuntimeError("Modulus too small")
    block_bytes = mod_bytes - 1

    try:
        out = open(out_path, "wb")
    except OSError:
        raise RuntimeError("Cannot open output file: " + out_path)

    with out:
        tmp_len = orig_size
        lenbuf = bytearray(8)
        for i in range(7, -1, -1):
            lenbuf[i] = tmp_len & 0xFF
            tmp_len >>= 8
        out.write(lenbuf)

        pos = 0
        while pos < len(plain):
            chunk = min(block_bytes, len(plain) - pos)

            block = bytearray(block_bytes)
            for i in range(chunk):
                block[i] = plain[pos + i]
            pos += chunk

            m = 0
            for i in range(block_bytes):
                m <<= 8
                m += block[i]

            c = rsa_encrypt(m, e, n)

            cipher = bytearray(mod_bytes)
            t = c
            for i in range(mod_bytes):
                cipher[mod_bytes - 1 - i] = t & 0xFF
                t >>= 8

            out.write(cipher)

    print(f"[OK] File encrypted to: {out_path}")


def decrypt_file_rsa(in_path: str, out_path: str, n: int, d: int) -> None:
    enc = read_file(in_path)
    if len(enc) < 8:
        raise RuntimeError("Encrypted file too short")

    orig_size = 0
    for i in range(8):
        orig_size = (orig_size << 8) | enc[i]

    mod_bytes = num_bytes(n)
    if mod_bytes < 2:
        raise RuntimeError("Modulus too small")
    block_bytes = mod_bytes - 1

    offset = 8
    if (len(enc) - offset) % mod_bytes != 0:
        raise RuntimeError("Encrypted data size is not a multiple of block size")

    blocks = (len(enc) - offset) // mod_bytes
    plain = bytearray()

    for b in range(blocks):
        base = offset + b * mod_bytes
        cipher_bytes = enc[base:base + mod_bytes]

        c = 0
        for i in range(mod_bytes):
            c <<= 8
            c += cipher_bytes[i]

        m = rsa_decrypt(c, d, n)

        block = bytearray(block_bytes)
        t = m
        for i in range(block_bytes):
            block[block_bytes - 1 - i] = t & 0xFF
            t >>= 8

        plain.extend(block)

    if orig_size > len(plain):
        raise RuntimeError("Original size larger than decrypted data")

    del plain[orig_size:]
    write_file(out_path, bytes(plain))

    print(f"[OK] File decrypted to: {out_path}")



def main() -> int:
    try:
        print("=== Lab 3: RSA key generation + file encryption/decryption ===")
        print("Enter DECIMAL DIGITS for each prime p and q (key base length): ", end="")
        s = input().strip()
        if s == "":
            return 0
        digits = int(s)

        print("Miller-Rabin rounds (default 16): ", end="")
        rounds_s = input().strip()
        if rounds_s == "":
            rounds = 16
        else:
            rounds = int(rounds_s)

        print("\n[+] Generating RSA primes p and q... This may take some time.")
        kp = generate_rsa(digits, rounds, 65537)

        print("\n=== RSA Keys ===")
        print(f"p  ({dec_digits(kp.p)} digits): {kp.p}")
        print(f"q  ({dec_digits(kp.q)} digits): {kp.q}")
        print(f"n  ({dec_digits(kp.n)} digits): {kp.n}")
        print(f"phi: {kp.phi}")
        print("Public key  (n, e):")
        print(f"  e = {kp.e}")
        print(f"  n = {kp.n}")
        print("Private key (n, d):")
        print(f"  d = {kp.d}")
        print(f"  n = {kp.n}")

        print("\nNow we will encrypt and then decrypt a file using this RSA key.")
        in_file = input("Input (plaintext) file name: ")
        enc_file = input("Output encrypted file name: ")
        dec_file = input("Output decrypted file name: ")

        encrypt_file_rsa(in_file, enc_file, kp.n, kp.e)
        decrypt_file_rsa(enc_file, dec_file, kp.n, kp.d)

        orig = read_file(in_file)
        dec = read_file(dec_file)

        if orig == dec:
            print("[OK] Decrypted file is identical to the original.")
        else:
            print("[!] Decrypted file differs from the original.")

        return 0

    except Exception as ex:
        print(f"Error: {ex}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
