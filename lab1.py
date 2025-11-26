from typing import Tuple

def _strip_leading_zeros(s: str) -> str:
    s = s.lstrip('0')
    return s if s else '0'

def _validate_dec_str(s: str) -> str:
    if s is None:
        raise ValueError("Порожнє значення")
    s = str(s).strip()
    if not s:
        raise ValueError("Порожній рядок")
    if any(ch < '0' or ch > '9' for ch in s):
        raise ValueError(f"Невалідні символи {s!r}")
    return _strip_leading_zeros(s)

def cmp_dec(a: str, b: str) -> int:
    a = _validate_dec_str(a)
    b = _validate_dec_str(b)
    if len(a) != len(b):
        return -1 if len(a) < len(b) else 1
    if a == b:
        return 0
    return -1 if a < b else 1


def add_dec(a: str, b: str) -> str:
    a = _validate_dec_str(a)
    b = _validate_dec_str(b)

    i, j = len(a) - 1, len(b) - 1
    carry = 0
    out = []

    while i >= 0 or j >= 0 or carry:
        da = ord(a[i]) - 48 if i >= 0 else 0
        db = ord(b[j]) - 48 if j >= 0 else 0
        s = da + db + carry
        out.append(chr((s % 10) + 48))
        carry = s // 10
        i -= 1
        j -= 1

    return _strip_leading_zeros(''.join(reversed(out)))

def sub_dec(a: str, b: str) -> str:
    a = _validate_dec_str(a)
    b = _validate_dec_str(b)
    if cmp_dec(a, b) < 0:
        raise ValueError("sub_dec вимагає a >= b.")

    i, j = len(a) - 1, len(b) - 1
    borrow = 0
    out = []

    while i >= 0:
        da = (ord(a[i]) - 48) - borrow
        db = ord(b[j]) - 48 if j >= 0 else 0
        if da < db:
            da += 10
            borrow = 1
        else:
            borrow = 0
        out.append(chr((da - db) + 48))
        i -= 1
        j -= 1

    return _strip_leading_zeros(''.join(reversed(out)))

def mul_small_dec(a: str, k: int) -> str:
    a = _validate_dec_str(a)
    if k < 0:
        raise ValueError("k від'ємне")
    if a == '0' or k == 0:
        return '0'

    carry = 0
    out = []
    for i in range(len(a) - 1, -1, -1):
        da = ord(a[i]) - 48
        prod = da * k + carry
        out.append(chr((prod % 10) + 48))
        carry = prod // 10
    while carry:
        out.append(chr((carry % 10) + 48))
        carry //= 10
    return _strip_leading_zeros(''.join(reversed(out)))

def mul_dec(a: str, b: str) -> str:
    a = _validate_dec_str(a)
    b = _validate_dec_str(b)
    if a == '0' or b == '0':
        return '0'

    res = [0] * (len(a) + len(b))

    for i in range(len(a) - 1, -1, -1):
        da = ord(a[i]) - 48
        carry = 0
        for j in range(len(b) - 1, -1, -1):
            db = ord(b[j]) - 48
            k = i + j + 1
            tmp = res[k] + da * db + carry
            res[k] = tmp % 10
            carry  = tmp // 10
        res[i] += carry

    s = ''.join(chr(d + 48) for d in res)
    return _strip_leading_zeros(s)

# >>> ДОДАНО: зведення в квадрат
def square_dec(a: str) -> str:
    a = _validate_dec_str(a)
    return mul_dec(a, a)
# <<<

def mod_dec(a: str, m: str) -> str:
    a = _validate_dec_str(a)
    m = _validate_dec_str(m)
    if m == '0':
        raise ZeroDivisionError("Модуль дорівнює 0.")

    r = '0'
    for ch in a:
        r = add_dec(mul_small_dec(r, 10), str(ord(ch) - 48))
        while cmp_dec(r, m) >= 0:
            r = sub_dec(r, m)
    return _strip_leading_zeros(r)

def is_odd(a: str) -> bool:
    a = _validate_dec_str(a)
    return (ord(a[-1]) - 48) % 2 == 1

def div2_dec(a: str) -> str:
    a = _validate_dec_str(a)
    if a == '0':
        return '0'
    carry = 0
    out = []
    for ch in a:
        cur = carry * 10 + (ord(ch) - 48)
        out.append(chr((cur // 2) + 48))
        carry = cur % 2
    return _strip_leading_zeros(''.join(out))

def pow_mod(base: str, exp: str, mod: str) -> str:
    base = mod_dec(_validate_dec_str(base), mod)
    exp  = _validate_dec_str(exp)
    mod  = _validate_dec_str(mod)
    if mod == '0':
        raise ZeroDivisionError("Модуль дорівнює 0.")
    if exp == '0':
        return '1' if mod != '1' else '0'

    result = '1'
    cur = base
    while exp != '0':
        if is_odd(exp):
            result = mod_dec(mul_dec(result, cur), mod)
        cur = mod_dec(mul_dec(cur, cur), mod)
        exp = div2_dec(exp)
    return _strip_leading_zeros(result)


def read_dec_input(prompt: str, nonzero: bool = False) -> str:
    while True:
        s = input(prompt).strip()
        try:
            s = _validate_dec_str(s)
            if nonzero and s == '0':
                print("Значення 0")
                continue
            return s
        except ValueError as e:
            print(f"Помилка {e}")

def op_add():
    a = read_dec_input("Перше число a: ")
    b = read_dec_input("Друге число b: ")
    print("a + b =", add_dec(a, b))

def op_mul():
    a = read_dec_input("Перше число a: ")
    b = read_dec_input("Друге число b: ")
    print("a * b =", mul_dec(a, b))

# >>> ДОДАНО: хендлер для квадрата
def op_square():
    a = read_dec_input("Число a: ")
    print("a^2 =", square_dec(a))
# <<<

def op_mod():
    a = read_dec_input("Число a: ")
    m = read_dec_input("Модуль m (m>0): ", nonzero=True)
    print("a mod m =", mod_dec(a, m))

# (op_pow_mod лишаємо в коді, але не показуємо в меню)

def print_menu():
    print("\nМЕНЮ")
    print("1 — Додавання")
    print("2 — Множення")
    print("3 — Зведення в квадрат: a^2")
    print("4 — Остача: a mod m")
    print("0 — Вихід")

def main_loop():
    ops = {
        '1': op_add,
        '2': op_mul,
        '3': op_square,
        '4': op_mod,
        '0': None
    }
    while True:
        print_menu()
        choice = input("Операції (0–4): ").strip()
        if choice == '0':
            break
        fn = ops.get(choice)
        if fn is None:
            print("Помилка")
            continue
        try:
            fn()
        except ZeroDivisionError as zde:
            print("Модуль - нуль", zde)
        except ValueError as ve:
            print("Помилка значень:", ve)
        except Exception as e:
            print("unlucky", e)


if __name__ == "__main__":
    main_loop()
