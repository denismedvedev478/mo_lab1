from math import exp
import numpy as np
import matplotlib.pyplot as plt
import lab_opt   # общие методы (golden используется при уточнении в удвоении шага)

global E, R, Is, phi_t, eps
E = 5
R = 1
Is = 1.2E-10
phi_t = 0.025
eps = 0.001

global GOLDEN
GOLDEN = (1 + 5 ** 0.5) / 2

# I1 - I2 = 0
# I1 = (E-phi)/R ; I2 = Is*(exp(phi/phi_t) - 1)
# Ф(phi) = abs(I1 - I2)

def Phi(phi):
    I1 = (E-phi)/R 
    I2 = Is*(exp(phi/phi_t) - 1)
    return abs(I1-I2)


def calc_by_dichotomy(a: float, b: float):
    print('-' * 18)
    print("Dichotomy")
    def build_grid(a, b):
        return [a + i*(b - a)/4 for i in range(5)]

    grid = build_grid(a, b)
    phi_vals = list(map(Phi, grid))
    counter = 0

    while (b - a) > eps:
        counter += 1

        min_ind = phi_vals.index(min(phi_vals))

        if min_ind == 0:
            a_new, b_new = grid[0], grid[1]
        elif min_ind == 4:
            a_new, b_new = grid[-2], grid[-1]
        else:
            a_new, b_new = grid[min_ind - 1], grid[min_ind + 1]

        arg1 = '{0:0.3E}'.format(a)
        arg2 = '{0:0.3E}'.format(b)
        arg3 = '{0:0.3E}'.format(grid[min_ind])
        arg4 = '{0:0.3E}'.format(Phi(a))
        arg5 = '{0:0.3E}'.format(Phi(b))
        arg6 = '{0:0.3E}'.format(Phi(grid[min_ind]))
        print(f"{counter}) \t a: {arg1}, b: {arg2}, res_x: {arg3} | Ф(а): {arg4}, Ф(b): {arg5}, Ф(res_x): {arg6}")

        a, b = a_new, b_new
        new_grid = build_grid(a, b)
        phi_vals = list(map(Phi, new_grid))
        grid = new_grid

    result_x = (a + b) / 2
    result_f = Phi(result_x)
    print(f"result: Iterations: {counter}, x: {result_x}, Phi(x): {result_f}")


def calc_by_golden_ratio(a:float, b:float):
    x1 = b - (b - a) / GOLDEN
    x2 = a + (b - a) / GOLDEN

    res_x = 0
    res = 0

    counter = 0
    print('-' * 18)
    print("Golden ratio")
    while (b-a) > eps:
        if Phi(x1) < Phi(x2):
            b = x2
            x2 = x1
            x1 = b - (b-a)/GOLDEN 
            res_x = x1
            res = Phi(x1)
        else:
            a = x1
            x1 = x2
            x2 = a + (b-a)/GOLDEN 
            res_x = x2
            res = Phi(x2)

        counter += 1

        arg1 = '{0:0.3E}'.format(a)
        arg2 = '{0:0.3E}'.format(b)
        arg3 = '{0:0.3E}'.format(res_x)
        arg4 = '{0:0.3E}'.format(Phi(a))
        arg5 = '{0:0.3E}'.format(Phi(b))
        arg6 = '{0:0.3E}'.format(res)
        print(f"{counter}) \t a: {arg1}, b: {arg2}, res_x: {arg3} | Ф(а): {arg4}, Ф(b): {arg5}, Ф(res_x): {arg6}")
    
    print(f"result: \t{counter, res_x, res}")


def calc_by_doubling_step(a: float, b: float, h0: float):
    # Идём от a вправо, удваивая шаг, пока Phi убывает.
    # Как только Phi перестала убывать, минимум лежит на [prev, xn];
    # дальше сужаем этот отрезок золотым сечением (lab_opt.golden).
    print('-'*18)
    print("Doubling Step")
    x, h = a, h0
    fx = Phi(x)
    prev = x
    counter = 0
    while True:
        counter += 1
        xn = min(x + h, b)
        fn = Phi(xn)

        arg1 = '{0:0.3E}'.format(h)
        arg2 = '{0:0.3E}'.format(xn)
        arg3 = '{0:0.3E}'.format(fn)
        print(f"{counter}) \t h: {arg1}, x: {arg2}, Phi(x): {arg3}")

        if fn >= fx or xn == b:
            break
        prev, x, fx = x, xn, fn
        h *= 2

    result_x = lab_opt.golden(Phi, prev, xn, eps)
    print(f"bracket: [{prev:.6f}, {xn:.6f}]")
    print(f"result: Iterations: {counter}, x: {result_x}, Phi(x): {Phi(result_x)}")


if __name__ == '__main__':
    calc_by_dichotomy(0, 5)
    calc_by_golden_ratio(0, 5)
    calc_by_doubling_step(0, 5, 0.00001)

    xb1 = 0.607
    xb2 = 0.609
    plt.semilogy(np.linspace(xb1, xb2, 100), list(map(Phi, np.linspace(xb1, xb2, 100))))
    plt.show()
