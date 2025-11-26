from __future__ import annotations

import secrets
from math import gcd, isqrt
from typing import Tuple, List

def _sieve_primes_upto(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = bytearray(b"\x01") * (n + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, isqrt(n) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n + 1:step] = b"\x00" * (((n - start) // step) + 1)
    return [i for i, v in enumerate(sieve) if v]

_SMALL_PRIMES: List[int] = _sieve_primes_upto(10000)

def is_probable_prime_fermat(n: int, k: int = 16) -> bool:

    if n < 2:
        return False
    for p in (2, 3, 5):
        if n == p:
            return True
        if n % p == 0:
            return False

    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        if pow(a, n - 1, n) != 1:
            return False
    return True


def is_probable_prime_mr(n: int, k: int = 8) -> bool:
    """Ймовірнісна перевірка простоти Міллера–Рабіна (опційно)."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES[:25]:
        if n == p:
            return True
        if n % p == 0:
            return False

    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1

    for _ in range(k):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def _random_ndigit_int(num_digits: int) -> int:
    """Криптостійко генерує випадкове n-значне число у десятковій системі."""
    if num_digits < 1:
        raise ValueError("num_digits має бути >= 1")
    low = 10 ** (num_digits - 1)
    high = 10 ** num_digits - 1
    span = high - low + 1
    return low + secrets.randbelow(span)


def _passes_small_prime_sieve(n: int) -> bool:
    """Швидко відсікає кандидата, якщо він ділиться на малий простий."""
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    return True


def generate_large_prime_digits(
    num_digits: int = 50,
    *,
    fermat_rounds: int = 24,
    use_mr: bool = True,
    mr_rounds: int = 8,
) -> int:
    if num_digits < 2:
        raise ValueError("Для RSA варто брати принаймні 2 цифри; зазвичай 20+ цифр.")

    while True:
        n = _random_ndigit_int(num_digits)
        n |= 1

        if not _passes_small_prime_sieve(n):
            continue

        if not is_probable_prime_fermat(n, k=fermat_rounds):
            continue

        if use_mr and not is_probable_prime_mr(n, k=mr_rounds):
            continue

        return n

def _mod_inverse(a: int, m: int) -> int:
    try:
        return pow(a, -1, m)
    except TypeError:
        def egcd(x, y):
            if x == 0:
                return y, 0, 1
            g, b, a = egcd(y % x, x)
            return g, a - (y // x) * b, b

        g, x, _ = egcd(a, m)
        if g != 1:
            raise ValueError("Оберненого елемента не існує")
        return x % m


def generate_rsa_keys(
    num_digits: int = 50,
    *,
    e: int = 65537,
    use_mr: bool = True,
    fermat_rounds: int = 24,
    mr_rounds: int = 8,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:

    if num_digits < 2:
        raise ValueError("num_digits має бути >= 2 для RSA.")

    while True:
        p = generate_large_prime_digits(
            num_digits, fermat_rounds=fermat_rounds, use_mr=use_mr, mr_rounds=mr_rounds
        )
        q = generate_large_prime_digits(
            num_digits, fermat_rounds=fermat_rounds, use_mr=use_mr, mr_rounds=mr_rounds
        )
        if p == q:
            continue

        n = p * q
        phi = (p - 1) * (q - 1)

        if gcd(e, phi) != 1:
            for cand in (3, 5, 17, 257, 65537):
                if gcd(cand, phi) == 1:
                    e = cand
                    break
            else:
                continue

        d = _mod_inverse(e, phi)
        return (e, n), (d, n)


def rsa_encrypt_int(m: int, pubkey: Tuple[int, int]) -> int:
    """Шифрує ціле m (0 <= m < n) з використанням відкритого ключа (e, n)."""
    e, n = pubkey
    if not (0 <= m < n):
        raise ValueError("m має задовольняти 0 <= m < n")
    return pow(m, e, n)


def rsa_decrypt_int(c: int, privkey: Tuple[int, int]) -> int:
    """Дешифрує ціле c (0 <= c < n) з використанням закритого ключа (d, n)."""
    d, n = privkey
    if not (0 <= c < n):
        raise ValueError("c має задовольняти 0 <= c < n")
    return pow(c, d, n)

def _print_run_random(bits: int = 128) -> None:
    rnd = secrets.randbits(bits)
    print(f"[RNG] Випадкове число ({bits} біт): {rnd}")

def _input_choice(prompt: str, choices: dict) -> str:
    """Питає у користувача вибір зі словника choices (key->description). Повертає key."""
    while True:
        print(prompt)
        for k, desc in choices.items():
            print(f"  {k} — {desc}")
        s = input("> ").strip()
        if s in choices:
            return s
        print("Некоректний вибір. Спробуйте ще раз.\n")

def _input_int(prompt: str, min_value: int = 1) -> int:
    while True:
        s = input(prompt).strip()
        if s.isdigit():
            v = int(s)
            if v >= min_value:
                return v
        print(f"Введіть ціле число \u2265 {min_value}.\n")

def _main():

    mode = _input_choice(
        "Оберіть режим роботи:",
        {"1": "Згенерувати просте число", "2": "Згенерувати RSA ключі"}
    )

    algo = _input_choice(
        "Оберіть алгоритм перевірки простоти:",
        {"1": "Лише Ферма", "2": "Ферма + Міллер–Рабін"}
    )
    use_mr = (algo == "2")

    digits = _input_int("Вкажіть кількість десяткових цифр (наприклад, 40): ", min_value=2)

    if mode == "1":
        p = generate_large_prime_digits(digits, fermat_rounds=24, use_mr=use_mr, mr_rounds=8)
        print(f"\n[OK] Просте число з ~{digits} цифр (algo={algo}):")
        print(p)
    else:
        pub, priv = generate_rsa_keys(
            num_digits=digits,
            e=65537,
            use_mr=use_mr,
            fermat_rounds=24,
            mr_rounds=8,
        )
        e, n = pub
        d, _ = priv
        print(f"\n[OK] Згенеровано RSA-ключі (algo={algo}, digits≈{digits}):")
        print(f"Public (e, n): e={e}, n={n}")
        print(f"Private (d, n): d={d}, n={n}")

if __name__ == "__main__":
    _main()
