"""Лабораторная: одномерная оптимизация.
Тестовая функция f(x) = x^3 - 3 sin(x) на [0; 1].
"""
from math import sin, cos, atan, log1p
import numpy as np
import matplotlib.pyplot as plt

global GOLDEN
GOLDEN = (1+5**.5) / 2


def f(x):   return x**3-3*sin(x)
def df(x):  return 3*x**2-3*cos(x)
def d2f(x): return 6*x+3*sin(x)
A, B = 0, 1


class Counted:
    """wrapper  counts how many times a function was called"""
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, x):
        self.n += 1
        return self.fn(x)



H1, H2 = 1e-6, 1e-4

def fd_first(F, kind, h=H1):
    if kind == 'left':    return lambda x: (F(x) - F(x - h)) / h
    if kind == 'right':   return lambda x: (F(x+h) - F(x)) / h
    if kind == 'central': return lambda x: (F(x+h) - F(x-h)) / (2*h)
    raise ValueError(kind)

def fd_second(F, h=H2):
    return lambda x: (F(x+h) - 2*F(x) + F(x-h)) / h**2



# без производных
def brute_force(F, a, b, eps):
    "Brute forcing through a grid with the step of eps. pick the lowest num"
    n = int((b - a) / eps) + 1
    xs = [a + i * (b - a) / n for i in range(n + 1)]
    return min(xs, key=F)


def bitwise(F, a, b, eps):
    """Поразрядный поиск: идём шагом h, пока f убывает; иначе
    разворачиваемся и уменьшаем шаг в 4 раза."""
    x, h = a, (b - a) / 4
    fx = F(x)
    while abs(h) > eps / 4:
        xn = x + h
        if a <= xn <= b:
            fn = F(xn)
            if fn < fx:
                x, fx = xn, fn
                continue
        h = -h / 4
    return x


def dichotomy(F, a, b, eps):
    """Дихотомия - логика из старого 4.py (сетка из 5 точек, отрезок
    сжимается вдвое), только f передаётся параметром и нет печати."""
    def build_grid(a, b):
        return [a + i * (b - a) / 4 for i in range(5)]

    grid = build_grid(a, b)
    phi_vals = list(map(F, grid))
    while (b - a) > eps:
        min_ind = phi_vals.index(min(phi_vals))
        if min_ind == 0:
            a_new, b_new = grid[0], grid[1]
        elif min_ind == 4:
            a_new, b_new = grid[-2], grid[-1]
        else:
            a_new, b_new = grid[min_ind - 1], grid[min_ind + 1]
        a, b = a_new, b_new
        grid = build_grid(a, b)
        phi_vals = list(map(F, grid))
    return (a + b) / 2


def golden(F, a, b, eps):
    """Золотое сечение - та же логика, что в старом 4.py, но значения
    f(x1), f(x2) запоминаются: одно вычисление f на итерацию."""
    x1 = b - (b - a) / GOLDEN
    x2 = a + (b - a) / GOLDEN
    f1, f2 = F(x1), F(x2)
    while (b - a) > eps:
        if f1 < f2:
            b = x2
            x2, f2 = x1, f1
            x1 = b - (b - a) / GOLDEN
            f1 = F(x1)
        else:
            a = x1
            x1, f1 = x2, f2
            x2 = a + (b - a) / GOLDEN
            f2 = F(x2)
    return (a + b) / 2


def doubling_step(F, a, b, h0, eps):
    """Удвоение шага (исправленный вариант): от левого конца идём вправо,
    удваивая шаг, пока f убывает. Минимум оказывается на [prev, xn];
    дальше сужаем отрезок золотым сечением."""
    x, h = a, h0
    fx = F(x)
    prev = x
    while True:
        xn = min(x + h, b)
        fn = F(xn)
        if fn >= fx or xn == b:
            break
        prev, x, fx = x, xn, fn
        h *= 2
    return golden(F, prev, xn, eps)


def parabolas(F, a, b, eps, max_iter=100):
    """Метод парабол: вершина параболы через 3 точки -> новая точка."""
    x1, x2, x3 = a, (a + b) / 2, b
    f1, f2, f3 = F(x1), F(x2), F(x3)
    for _ in range(max_iter):
        den = (x2 - x1) * (f2 - f3) - (x2 - x3) * (f2 - f1)
        if den == 0:
            break
        u = x2 - 0.5 * ((x2 - x1) ** 2 * (f2 - f3) - (x2 - x3) ** 2 * (f2 - f1)) / den
        if not (x1 < u < x3):
            break
        fu = F(u)
        if abs(u - x2) < eps:
            return u if fu < f2 else x2
        if u < x2:
            if fu < f2: x3, f3, x2, f2 = x2, f2, u, fu
            else:       x1, f1 = u, fu
        else:
            if fu < f2: x1, f1, x2, f2 = x2, f2, u, fu
            else:       x3, f3 = u, fu
    return min(((f1, x1), (f2, x2), (f3, x3)))[1]


# ---- с производными --------------------------------------------------------
def midpoint(dF, a, b, eps):
    """Средней точки: бисекция по знаку f'."""
    while (b - a) > eps:
        c = (a + b) / 2
        d = dF(c)
        if d > 0:   b = c
        elif d < 0: a = c
        else:       return c
    return (a + b) / 2


def chord(dF, a, b, eps, max_iter=200):
    """Хорд: нуль f' ищем по секущей через (a, f'(a)), (b, f'(b)).
    Остановка по |f'(x)| < eps."""
    da, db = dF(a), dF(b)
    x = a
    for _ in range(max_iter):
        x = a - da * (b - a) / (db - da)
        dx = dF(x)
        if abs(dx) < eps:
            break
        if dx > 0: b, db = x, dx
        else:      a, da = x, dx
    return x


def newton(dF, d2F, x0, eps, max_iter=100):
    """Ньютон: x <- x - f'(x)/f''(x)."""
    x = x0
    for _ in range(max_iter):
        d2 = d2F(x)
        if d2 == 0:                           # на разошедшемся x разностная f'' может занулиться
            break
        step = dF(x) / d2
        x -= step
        if abs(step) < eps or abs(x) > 1e8:   # сошлись / разошлись
            break
    return x


def newton_raphson(F, dF, d2F, x0, eps, max_iter=200):
    """Ньютон-Рафсон: x <- x - alpha*f'/f'', alpha дробим пополам,
    пока f не уменьшится."""
    x = x0
    for _ in range(max_iter):
        d2 = d2F(x)
        if d2 == 0:
            break
        step = dF(x) / d2
        if abs(step) < eps:
            break
        alpha = 1.0
        while F(x - alpha * step) >= F(x) and alpha > 1e-10:
            alpha /= 2
        x -= alpha * step
    return x


def marquardt(F, dF, d2F, x0, eps, lam=1.0, max_iter=200):
    """Марквардт: x <- x - f'/(f'' + lambda). Шаг удачный (f уменьшилась) ->
    lambda/2, неудачный -> lambda*2 и пересчёт."""
    x = x0
    for _ in range(max_iter):
        g = dF(x)
        if abs(g) < eps:
            break
        while True:
            xn = x - g / (d2F(x) + lam)
            if F(xn) < F(x):
                lam /= 2
                x = xn
                break
            lam *= 2
            if lam > 1e12:
                return x
    return x


# ---- глобальный поиск (п.6) -----------------------------------------------
def piyavskii(F, a, b, L, eps, max_iter=5000):
    """Метод ломаных (Пиявского): L - константа Липшица, eps - точность по
    значению функции. Возвращает (x, f(x), число вычислений f)."""
    pts = [(a, F(a)), (b, F(b))]
    for _ in range(max_iter):
        best = None
        for i, ((x1, f1), (x2, f2)) in enumerate(zip(pts, pts[1:])):
            x = (x1 + x2) / 2 + (f1 - f2) / (2 * L)           # пересечение «зубьев»
            low = (f1 + f2) / 2 - L * (x2 - x1) / 2           # значение ломаной там
            if best is None or low < best[0]:
                best = (low, x, i)
        low, x, i = best
        if min(p[1] for p in pts) - low < eps:
            break
        pts.insert(i + 1, (x, F(x)))
    xm, fm = min(pts, key=lambda p: p[1])
    return xm, fm, len(pts)


# ============================================================ ЭКСПЕРИМЕНТЫ
EPS_LIST = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]
XSTAR = newton(df, d2f, 1.0, 1e-14)        # «точное» решение для сравнения

NO_DER = {
    'перебор':         lambda F, dF, d2F, eps: brute_force(F, A, B, eps),
    'поразрядный':     lambda F, dF, d2F, eps: bitwise(F, A, B, eps),
    'дихотомия':       lambda F, dF, d2F, eps: dichotomy(F, A, B, eps),
    'золотое сечение': lambda F, dF, d2F, eps: golden(F, A, B, eps),
    'парабол':         lambda F, dF, d2F, eps: parabolas(F, A, B, eps),
    'удвоение шага':   lambda F, dF, d2F, eps: doubling_step(F, A, B, eps, eps),
}
WITH_DER = {
    'средней точки':   lambda F, dF, d2F, eps: midpoint(dF, A, B, eps),
    'хорд':            lambda F, dF, d2F, eps: chord(dF, A, B, eps),
    'Ньютона':         lambda F, dF, d2F, eps: newton(dF, d2F, B, eps),
}


def run(method, kind, eps):
    """kind: 'analytic' | 'left' | 'right' | 'central'. Возвращает (x, N)."""
    cf = Counted(f)
    if kind == 'analytic':
        cdf, cd2 = Counted(df), Counted(d2f)
    else:                                   # производные через разности от f
        cdf, cd2 = fd_first(cf, kind), fd_second(cf)
    x = method(cf, cdf, cd2, eps)
    n = cf.n + (cdf.n + cd2.n if kind == 'analytic' else 0)
    return x, n


def plot_test_function():
    xs = np.linspace(A, B, 400)
    plt.figure()
    plt.plot(xs, [f(x) for x in xs])
    plt.plot(XSTAR, f(XSTAR), 'ro', label=f'x* = {XSTAR:.6f}')
    plt.title('f(x) = x^3 - 3 sin x'); plt.grid(); plt.legend()


def part3_4():
    table = {}
    for name, m in NO_DER.items():
        table[(name, '-')] = [run(m, 'analytic', e) for e in EPS_LIST]
    for name, m in WITH_DER.items():
        for k in ('analytic', 'left', 'right', 'central'):
            table[(name, k)] = [run(m, k, e) for e in EPS_LIST]

    head = f"{'метод':<18}{'произв.':<10}" + ''.join(f'{e:>10.0e}' for e in EPS_LIST)
    print('\nN - число вычислений f и/или производных'); print(head)
    for (name, k), res in table.items():
        print(f'{name:<18}{k:<10}' + ''.join(f'{n:>10d}' for _, n in res))
    print('\n|x - x*|'); print(head)
    for (name, k), res in table.items():
        print(f'{name:<18}{k:<10}' + ''.join(f'{abs(x - XSTAR):>10.1e}' for x, _ in res))

    plt.figure()
    for (name, k), res in table.items():
        if k in ('-', 'analytic'):
            plt.loglog(EPS_LIST, [n for _, n in res], 'o-', label=name)
    plt.xlabel('eps'); plt.ylabel('N'); plt.grid(); plt.legend()
    plt.title('N(eps), аналитические производные')


# ---- п.5 --------------------------------------------------------------------
def g(x):   return x * atan(x) - 0.5 * log1p(x * x)   # log1p - чтобы не терять точность
def dg(x):  return atan(x)
def d2g(x): return 1 / (1 + x * x)


def part5():
    x0s = [0.5, 1.0, 1.3, 1.4, 1.5, 2.0, 5.0]
    variants = {'аналит. производные': (dg, d2g),
                'разностные (центр.)': (fd_first(g, 'central'), fd_second(g))}
    for vname, (d1, d2) in variants.items():
        print(f'\nп.5, {vname}; x* = 0')
        print(f"{'x0':>6}{'Ньютон':>14}{'Марквардт':>14}{'Ньютон-Рафсон':>16}")
        for x0 in x0s:
            print(f'{x0:6.2f}{newton(d1, d2, x0, 1e-8):14.3e}'
                  f'{marquardt(g, d1, d2, x0, 1e-8):14.3e}'
                  f'{newton_raphson(g, d1, d2, x0, 1e-8):16.3e}')
    xs = np.arange(0, 3, 0.01)
    bad = next((x for x in xs if abs(newton(dg, d2g, x, 1e-8)) > 1e-4), None)
    print(f'\nметод Ньютона сходится при |x0| < {bad} (первое плохое x0 на сетке 0.01)')


# ---- п.6 --------------------------------------------------------------------
def part6():
    # L: |(cos x/x^2)'| = |sin x/x^2 + 2cos x/x^3| <= 1 + 2 = 3 при x >= 1
    #    |(x/10 + 2 sin 4x)'| = |0.1 + 8 cos 4x| <= 8.1
    tasks = [('cos(x)/x^2',      lambda x: cos(x) / x ** 2,        1, 12, 3.0),
             ('x/10 + 2 sin 4x', lambda x: x / 10 + 2 * sin(4 * x), 0, 4, 8.1)]
    eps = 1e-3
    for name, F, a, b, L in tasks:
        cF = Counted(F)
        xb = brute_force(cF, a, b, eps)
        xp, fp, n = piyavskii(F, a, b, L, eps)
        print(f'\n{name} на [{a}; {b}]')
        print(f'  перебор:  x = {xb:.5f}, f = {F(xb):.6f}, N = {cF.n}')
        print(f'  ломаных:  x = {xp:.5f}, f = {fp:.6f}, N = {n}')
        xs = np.linspace(a, b, 1000)
        plt.figure()
        plt.plot(xs, [F(x) for x in xs])
        plt.plot(xb, F(xb), 'go', label='перебор')
        plt.plot(xp, fp, 'rx', label='ломаных')
        plt.title(name); plt.grid(); plt.legend()


if __name__ == '__main__':
    plot_test_function()
    part3_4()
    part5()
    part6()
    plt.show()