from decimal import getcontext
import numpy as np
import sympy as sp
from scipy.optimize import fsolve

# 设置 Decimal 的默认精度
getcontext().prec = 100


# 用户自定义 ODE 函数
def get_ode_function():
    """
    提示用户输入一个 ODE 函数，并返回该函数。
    支持任何形式的表达式，包括 sin, cos 等。
    """
    print("Choose a formula (function input format is f(t, y))")
    print("1. dy/dt = -2y")
    print("2. dy/dt = t * sqrt(y)")
    print("3. dy/dt = y * (1 - y)  (e.g. population models)")
    print("4. dy/dt = y - t^2 + 1 (equations in questions)")
    print("5. Customized function (You can input any formula like t*sin(y) + cos(t))")

    choice = input("Please input your option (1/2/3/4/5): ").strip()

    # 默认函数是 dy/dt = -2y
    if choice == '1':
        def f(t, y):
            return -2 * y

        return f

    elif choice == '2':
        def f(t, y):
            return t * np.sqrt(y)

        return f

    elif choice == '3':
        def f(t, y):
            return y * (1 - y)

        return f

    elif choice == '4':
        def f(t, y):
            return y - t ** 2 + 1

        return f

    elif choice == '5':
        print("You chose to input a customized function.")
        print("You can use the following mathematical functions and operators:")
        print("  - Trigonometric functions: sin(y), cos(y), tan(y), etc.")
        print("  - Exponential, log, sqrt, etc.")

        allowed_functions = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'asin': sp.asin,
            'acos': sp.acos,
            'atan': sp.atan,
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'exp': sp.exp,
            'log': sp.log,
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi,
            'e': sp.E,
        }

        while True:
            expr = input("Please input your customized expression for dy/dt: ").strip()

            if expr.lower() in ['exit', 'quit']:
                print("Exiting function input.")
                return None

            try:
                ode_expr = sp.sympify(expr, locals=allowed_functions)
                symbols_in_expr = ode_expr.free_symbols
                allowed_free_symbols = {sp.Symbol('t'), sp.Symbol('y'), sp.Symbol('pi'), sp.Symbol('e')}
                if not symbols_in_expr.issubset(allowed_free_symbols):
                    raise ValueError("Only 't', 'y', 'pi', and 'e' are allowed variables.")

                ode_func = sp.lambdify((sp.Symbol('t'), sp.Symbol('y')), ode_expr, modules=["numpy"])
                ode_func(1, 1)  # Test the function
                return ode_func

            except Exception as e:
                print(f"Error: {e}")


# 求解常微分方程（ODE）的数值解
def solve_ode(f, t0, y0, t_end, h, known_w_values=None, init_method=None):
    t0, y0, h, t_end = map(float, (t0, y0, h, t_end))
    n_steps = int((t_end - t0) / h) + 1
    t_values = np.linspace(t0, t_end, num=n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0

    if known_w_values is not None and len(known_w_values) >= 3:
        for i in range(3):
            if i + 1 < n_steps:
                y_values[i + 1] = known_w_values[i]
    else:
        # 使用 Runge-Kutta-Fehlberg 方法初始化前三步
        for i in range(1, 4):
            if i >= n_steps:
                break  # 防止索引超出范围
            t_current = t_values[i - 1]
            y_current = y_values[i - 1]

            # 计算 RKF45 的 k1 到 k6
            k1 = h * f(t_current, y_current)
            k2 = h * f(t_current + h / 4, y_current + (1 / 4) * k1)
            k3 = h * f(t_current + 3 * h / 8, y_current + (3 / 32) * k1 + (9 / 32) * k2)
            k4 = h * f(t_current + 12 * h / 13,
                       y_current + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
            k5 = h * f(t_current + h,
                       y_current + (439 / 216) * k1 - (8 / 21) * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
            k6 = h * f(t_current + h / 2,
                       y_current - (8 / 27) * k1 + (2 / 9) * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (
                               11 / 40) * k5)

            # 计算五阶估计值 y_rk5
            y_rk5 = y_current + (16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55)

            # 将 y_rk5 赋值给 y_values
            y_values[i] = y_rk5

    for n in range(3, n_steps - 1):
        # 当前步
        t_n = t_values[n]
        y_n = y_values[n]
        t_n_minus_1 = t_values[n - 1]
        y_n_minus_1 = y_values[n - 1]
        t_n_minus_2 = t_values[n - 2]
        y_n_minus_2 = y_values[n - 2]
        t_n_minus_3 = t_values[n - 3]
        y_n_minus_3 = y_values[n - 3]

        t_pred = t_n + h
        y_pred = y_n + (h / 24) * (
                55 * f(t_n, y_n) -
                59 * f(t_n_minus_1, y_n_minus_1) +
                37 * f(t_n_minus_2, y_n_minus_2) -
                9 * f(t_n_minus_3, y_n_minus_3)
        )
        print(f"Prediction step for t = {t_pred:<10.2f}: y_pred = {y_pred:<15.8f}")

        def implicit_correction(y_next):
            return y_next - y_n - (h / 720) * (
                    251 * f(t_pred, y_next) +
                    646 * f(t_n, y_n) -
                    264 * f(t_n_minus_1, y_n_minus_1) +
                    106 * f(t_n_minus_2, y_n_minus_2) -
                    19 * f(t_n_minus_3, y_n_minus_3)
            )

        y_corrected = fsolve(implicit_correction, y_pred)[0]
        y_values[n + 1] = y_corrected

    return t_values, y_values


def main():
    try:
        significant_digits = int(input("Please enter the number of significant digits required: "))
        if significant_digits <= 0:
            raise ValueError("Significant digits must be positive.")
    except ValueError as ve:
        print(f"Invalid input for significant digits: {ve}")
        return

    getcontext().prec = significant_digits + 2
    f = get_ode_function()

    if f is None:
        return

    try:
        t0 = float(input("Please input initial time t0: "))
        y0 = float(input("Please input initial condition y0: "))
        t_end = float(input("Please input end time t_end: "))
        h = float(input("Please input step length h: "))
        if h <= 0:
            raise ValueError("Step length h must be positive.")
    except ValueError as ve:
        print(f"Invalid input for numerical parameters: {ve}")
        return

    _w = input("W values already known? (y/n): ").strip().lower()
    known_w_values = []

    if _w == 'y':
        try:
            print("Please input three known W values (W1, W2, W3):")
            for i in range(1, 4):
                w = float(input(f"Please input the value of W{i}: "))
                known_w_values.append(w)
        except ValueError:
            print("Invalid input for W values. Please enter valid numbers.")
            return
    elif _w != 'n':
        print("Invalid choice for W values. Please enter 'y' or 'n'.")
        return

    exact_solution = None
    exact_expr = None
    exact_symbols = {'t', 'pi', 'e'}
    print("Do you know the exact solution for the ODE? (y/n)")
    if input().strip().lower() == 'y':
        while True:
            expr = input("Please input the exact solution as a function of t: ").strip()
            if expr.lower() in ['exit', 'quit']:
                break
            try:
                exact_expr = sp.sympify(expr, locals={**sp.functions.__dict__, 'pi': sp.pi, 'e': sp.E})
                if not exact_expr.free_symbols.issubset({sp.Symbol('t')}):
                    raise ValueError("Exact solution can only depend on 't'.")
                exact_solution = sp.lambdify(sp.Symbol('t'), exact_expr, modules=['numpy'])
                break
            except Exception as e:
                print(f"Error in parsing exact solution: {e}")

    try:
        t_values, y_values = solve_ode(f, t0, y0, t_end, h, known_w_values=known_w_values if _w == 'y' else None)
    except Exception as e:
        print(f"An error occurred while solving the ODE: {e}")
        return

    print("\nNumerical solution result:")
    print(f"{'t':<15} {'y(t)':<20} {'Exact y(t)':<20} {'Absolute Error':<20}")

    for t, y in zip(t_values, y_values):
        exact_y = exact_solution(t) if exact_solution else None
        error = abs(y - exact_y) if exact_y is not None else None
        t_str = f"{t:.{significant_digits}f}"
        y_str = f"{y:.{significant_digits}f}"
        exact_str = f"{exact_y:.{significant_digits}f}" if exact_y is not None else "N/A"
        error_str = f"{error:.{significant_digits}f}" if error is not None else "N/A"
        print(f"{t_str:<15} {y_str:<20} {exact_str:<20} {error_str:<20}")


if __name__ == "__main__":
    main()
