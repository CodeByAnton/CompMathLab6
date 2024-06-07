import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def function1(x, y):
    return x ** 2 - 2 * y


def function2(x, y):
    return y + sp.cos(x)


def function3(x, y):
    return y + (1 + x) * y * y


def function4(x, y):
    return 2 * x - 3 * y


def get_exact_solution(func, x0, y0):
    # Определяем переменные
    x = sp.symbols('x')
    y = sp.Function('y')(x)

    # Определяем дифференциальное уравнение
    ode = sp.Eq(y.diff(x), func(x, y))

    # Решаем уравнение с начальными условиями
    solution = sp.dsolve(ode, y, ics={y.subs(x, x0): y0})

    # Получаем правую часть решения (y(x) = ...)
    y_solution = solution.rhs

    # Преобразуем символическое решение в функцию
    y_func = sp.lambdify(x, y_solution, 'numpy')

    return y_func


def input_data():
    print("Ввод данных")
    y0 = float(input("Введите значение y0: "))
    x0 = float(input("Введите значение x0: "))
    xn = float(input("Введите значение xn: "))
    h = float(input("Введите значение шага: "))
    eps = float(input("Введите погрешность: "))
    return y0, x0, xn, h, eps


def validate_data(y0, x0, xn, h):
    if xn < x0:
        raise ValueError("xn должно быть больше x0")

    if xn < x0 + h * 3:
        raise ValueError("Уменьшите шаг или увеличьте xn")
    if h <= 0:
        raise ValueError("Шаг должен быть больше нуля")


def improved_euler(f, y0, x0, xn, h):
    n = int((xn - x0) / h) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    y[0] = y0

    for i in range(n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h, y[i] + h * k1)
        y[i + 1] = y[i] + h * (k1 + k2) / 2

    return x, y


def runge_kutta_4th(f, y0, x0, xn, h):
    n = int((xn - x0) / h) + 1

    x = np.linspace(x0, xn, n)
    y = np.zeros(n)

    y[0] = y0

    for i in range(n - 1):
        k1 = f(x[i], y[i])
        k2 = f(x[i] + h / 2, y[i] + h * k1 / 2)
        k3 = f(x[i] + h / 2, y[i] + h * k2 / 2)
        k4 = f(x[i] + h, y[i] + h * k3)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x, y


def adams(f, y0, x0, xn, h, eps, k=4):
    x_rk, y_rk, h_2, _ = compute_with_runge_rule(f, y0, x0, x0 + (k - 1) * h, h, eps, "Метод Рунге-Кутта 4го порядка")
    n = int((xn - x0) / h_2) + 1
    x = np.linspace(x0, xn, n)
    y = np.zeros(n)
    for i in range(k):
        y[i] = y_rk[i]

    for i in range(k - 1, n - 1):
        y_pred = y[i] + h_2 / 24 * (
                55 * f(x[i], y[i]) - 59 * f(x[i - 1], y[i - 1]) + 37 * f(x[i - 2], y[i - 2]) - 9 * f(x[i - 3],
                                                                                                     y[i - 3]))
        f_pred = f(x[i + 1], y_pred)
        y_corr = y[i] + h_2 / 24 * (
                9 * f_pred + 19 * f(x[i], y[i]) - 5 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2]))

        while abs(y_pred - y_corr) > eps:
            y_pred = y_corr
            y_corr = y[i] + h_2 / 24 * (
                    9 * f_pred + 19 * f(x[i], y[i]) - 5 * f(x[i - 1], y[i - 1]) + f(x[i - 2], y[i - 2]))
        y[i + 1] = y_corr

    y_real_solution = get_exact_solution(f, x0, y0)(x)

    return x, y, y_real_solution


map_methods = {"Усовершенствованный метод Эйлера": improved_euler,
               "Метод Рунге-Кутта 4го порядка": runge_kutta_4th,
               "Метод Адамса": adams}


def compute_with_runge_rule(f, y0, x0, xn, h, eps, method_name):
    runge_coefficients = {"Усовершенствованный метод Эйлера": 3,
                          "Метод Рунге-Кутта 4го порядка": 15}
    k = runge_coefficients[method_name]
    current_method = map_methods[method_name]
    x, y = current_method(f, y0, x0, xn, h)
    h /= 2
    x_next, y_next = current_method(f, y0, x0, xn, h)
    while abs(y[-1] - y_next[-1]) / k > eps:
        h /= 2
        y = y_next
        x = x_next
        x_next, y_next = current_method(f, y0, x0, xn, h)
    y_real_solution = get_exact_solution(f, x0, y0)(x_next)

    return x_next, y_next, h, y_real_solution


def compute_eps_for_multi_step_method(y, y_exact):
    epsilon = abs(y[0] - y_exact[0])
    for i, j in zip(y, y_exact):
        epsilon = max(epsilon, abs(i - j))
    return epsilon


def compute_adams_method(f, y0, x0, xn, h, eps):
    x_curr,y_curr, y_real_solution = adams(f, y0, x0, xn, h, eps)
    i=0
    while True:
        # i+=1
        # print(i)
        # print(compute_eps_for_multi_step_method(y_curr, y_real_solution))
        if compute_eps_for_multi_step_method(y_curr, y_real_solution) < eps:
            return x_curr, y_curr, y_real_solution
        h/=2
        x_curr, y_curr, y_real_solution = adams(f, y0, x0, xn, h, eps)




def plot_graphik(x_data, y_data, method_name):
    plt.plot(x_data, y_data, color='blue')
    # plt.scatter(x_data, y_data, color='red')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(method_name)
    plt.show()


def main():
    try:
        y0, x0, xn, h, eps = input_data()
        validate_data(y0, x0, xn, h)
        n = int((xn - x0) / h) + 1

        print("Выберпите функцию (введите номер функции)")
        print("1. f1=x^2-2y")
        print("2. f2=y+cos(x)")
        print("3. f3=y+(1+x)*y^2")
        print("4. f4=2*x-3*y")

        chosen_func = int(input())

        if chosen_func == 1:
            function = function1
        elif chosen_func == 2:
            function = function2
        elif chosen_func == 3:
            function = function3
        elif chosen_func == 4:
            function = function4
        else:
            print("Такой функции нет")
            return

        # Усовершенствованный метод эйлера

        x_improved_euler, y_improved_euler, _, y_real_solution = compute_with_runge_rule(function, y0, x0, xn, h, eps,
                                                                                         "Усовершенствованный метод Эйлера")
        plt.plot(x_improved_euler, y_improved_euler, label="Усовершенствованный метод Эйлера")
        print("Усовершенствованный метод Эйлера")
        headers = ["X", "Y", "Точное решение"]
        data = list(zip(x_improved_euler, y_improved_euler, y_real_solution))
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print()

        # Метод Рунге-Кутта
        x_runge_kutta, y_runge_kutta, h1, y_real_solution = compute_with_runge_rule(function, y0, x0, xn, h, eps,
                                                                                    "Метод Рунге-Кутта 4го порядка")
        plt.plot(x_runge_kutta, y_runge_kutta, label="Метод Рунге-Кутта 4го порядка")
        print("Метод Рунге-Кутта 4го порядка")
        headers = ["X", "Y", "Точное решение"]
        data = list(zip(x_runge_kutta, y_runge_kutta, y_real_solution))
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print()

        # Точное решение
        function_exact_solution = get_exact_solution(function, x0, y0)
        x_exact_values = np.linspace(x0, xn, int((xn - x0) / h1) + 1)
        y_exact_values = function_exact_solution(x_exact_values)
        plt.plot(x_exact_values, y_exact_values, label="Точное решение")

        # Метод Адамса
        x_adams, y_adams, y_real_solution = compute_adams_method(function,y0,x0,xn,h,eps)
        plt.plot(x_adams, y_adams, label="Метод Адамса")

        print("Метод Адамса")
        headers = ["X", "Y", "Точное решение"]
        data = list(zip(x_adams, y_adams, y_real_solution))
        print(tabulate(data, headers=headers, tablefmt="grid"))

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(loc='best', fontsize='small')
        plt.title("Все методы")
        plt.show()

        plot_graphik(x_exact_values, y_exact_values, "Точное решение")
        plot_graphik(x_improved_euler, y_improved_euler, "Усовершенствованный метод Эйлера")
        plot_graphik(x_runge_kutta, y_runge_kutta, "Метод Рунге-Кутта 4го порядка")
        plot_graphik(x_adams, y_adams, "Метод Адамса")






    except ValueError as ve:
        print("\n Измените входные данные\n")


if __name__ == '__main__':
    main()
