from decimal import getcontext
import numpy as np
import sympy as sp
from scipy.optimize import fsolve

# 设置 Decimal 的默认精度（例如设置精度为100位，之后会根据需要调整）
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
            return -2 * y  # 示例：dy/dt = -2y

        return f

    elif choice == '2':
        def f(t, y):
            return t * np.sqrt(y)  # 示例：dy/dt = t * sqrt(y)

        return f

    elif choice == '3':
        def f(t, y):
            return y * (1 - y)  # 示例：dy/dt = y * (1 - y) (用于人口模型)

        return f

    elif choice == '4':
        # 这是问题中的微分方程: y' = y - t^2 + 1
        def f(t, y):
            return y - t ** 2 + 1

        return f

    elif choice == '5':
        print("You chose to input a customized function.")
        print("You can use the following mathematical functions and operators:")
        print("\nAllowed Functions:")
        print("  - Trigonometric functions: sin(y), cos(y), tan(y), cot(y), sec(y), csc(y)")
        print("  - Inverse trigonometric functions: asin(y), acos(y), atan(y)")
        print("  - Hyperbolic functions: sinh(y), cosh(y), tanh(y)")
        print("  - Exponential and logarithmic functions: exp(y), log(y) [natural log], ln(y)")
        print("  - Power and root functions: sqrt(y), y**n, t**n")
        print("  - Absolute value: abs(y)")
        print("  - Mathematical constants: pi, e")
        print("\nAllowed Operators:")
        print("  - Addition: +")
        print("  - Subtraction: -")
        print("  - Multiplication: *")
        print("  - Division: /")
        print("  - Exponentiation: **")
        print("  - Parentheses: ( )")
        print("\nAllowed Variables:")
        print("  - t (independent variable)")
        print("  - y (dependent variable)")
        print("\nExamples of valid expressions:")
        print("  1. t*sin(y) + cos(t)")
        print("  2. y/t + log(y)")
        print("  3. exp(t) - y**2")
        print("  4. (t**2 + y**3)/(1 + y)")
        print("  5. sinh(t) * cosh(y) - tan(y)")
        print("  6. atan(y) + sqrt(t)")
        print("  7. y * exp(-t) + log(t + y)")
        print("  8. abs(y) / (1 + t**2)")
        print("  9. (y**2 + t**2)**0.5")
        print("  10. (exp(y) - y) / t")
        print("  11. e**(-t) * sin(y)")
        print("  12. pi * y + t**2")

        # 定义允许的函数和符号，移除 log10
        allowed_functions = {
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'cot': sp.cot,
            'sec': sp.sec,
            'csc': sp.csc,
            'asin': sp.asin,
            'acos': sp.acos,
            'atan': sp.atan,
            'sinh': sp.sinh,
            'cosh': sp.cosh,
            'tanh': sp.tanh,
            'exp': sp.exp,
            'log': sp.log,  # 自然对数
            'ln': sp.log,  # ln 作为自然对数的别名
            'sqrt': sp.sqrt,
            'abs': sp.Abs,
            'pi': sp.pi,
            'e': sp.E,
        }
        allowed_symbols = {'t', 'y', 'pi', 'e'}

        while True:
            expr = input("Please input your customized expression for dy/dt: ").strip()

            # 检查用户是否想要退出
            if expr.lower() in ['exit', 'quit']:
                print("Exiting function input.")
                return None

            try:
                # 使用 sympy 解析用户输入的公式，限制在 allowed_functions 中
                ode_expr = sp.sympify(expr, locals=allowed_functions)

                # 检查是否只包含允许的符号
                symbols_in_expr = ode_expr.free_symbols
                allowed_free_symbols = {sp.Symbol('t'), sp.Symbol('y'), sp.Symbol('pi'), sp.Symbol('e')}
                if not symbols_in_expr.issubset(allowed_free_symbols):
                    raise ValueError(
                        f"Only 't', 'y', 'pi', and 'e' variables/constants are allowed. Found symbols: {symbols_in_expr}")

                # 转换为 Python 函数以便数值计算
                ode_func = sp.lambdify((sp.Symbol('t'), sp.Symbol('y')), ode_expr, modules=["numpy"])

                # 测试函数是否可以正常计算
                test_val = ode_func(1, 1)
                if not isinstance(test_val, (int, float, np.floating, np.integer)):
                    raise ValueError("The expression does not evaluate to a numerical value.")

                return ode_func

            except sp.SympifyError:
                print(
                    "Error: The expression could not be parsed. Please check the syntax and use only allowed functions.")
            except ValueError as ve:
                print(f"Error: {ve}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            print("Please try entering the expression again or type 'exit' to cancel.")


# 求解常微分方程（ODE）的数值解
def solve_ode(f, t0, y0, t_end, h, known_w_values=None, init_method=None):
    """
    使用 Runge-Kutta-Fehlberg 方法 (RKF45) 初始化前三步，然后使用 Predictor-Corrector 方法继续求解。

    参数：
    f : function
        常微分方程的右侧函数 f(t, y)
    t0 : float
        初始时间
    y0 : float
        初始条件
    t_end : float
        结束时间
    h : float
        步长
    known_w_values : list of floats, optional
        已知的初始条件列表（W1, W2, W3）
    init_method : str, optional
        选择初始化方法，'Runge-Kutta-Fehlberg'
    返回：
    t_values : numpy.ndarray
        时间点数组
    y_values : numpy.ndarray
        解数组
    """

    # 将所有输入转换为 float 类型，以确保兼容
    t0 = float(t0)
    y0 = float(y0)
    h = float(h)
    t_end = float(t_end)

    # 计算步数
    n_steps = int((t_end - t0) / h) + 1

    # 创建时间点数组，确保包含 t_end
    t_values = np.linspace(t0, t_end, num=n_steps)
    y_values = np.zeros(n_steps)
    y_values[0] = y0

    # 初始化前三步
    if known_w_values is not None and len(known_w_values) >= 3:
        # 使用已知的 W 值进行初始化 (W1, W2, W3)
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

    # 使用 Predictor-Corrector 方法继续求解
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

        # 预测步骤（使用 Adams-Bashforth 4步预测）
        y_pred = y_n + (h / 24) * (
                55 * f(t_n, y_n) -
                59 * f(t_n_minus_1, y_n_minus_1) +
                37 * f(t_n_minus_2, y_n_minus_2) -
                9 * f(t_n_minus_3, y_n_minus_3)
        )
        print(f"Prediction step for t = {t_pred:<10.2f}: y_pred = {y_pred:<15.8f}")

        # 修正步骤（使用 Adams-Moulton 4步修正）
        def implicit_correction(y_next):
            return y_next - y_n - (h / 720) * (
                    251 * f(t_pred, y_next) +
                    646 * f(t_n, y_n) -
                    264 * f(t_n_minus_1, y_n_minus_1) +
                    106 * f(t_n_minus_2, y_n_minus_2) -
                    19 * f(t_n_minus_3, y_n_minus_3)
            )

        # 使用 fsolve 解决隐式方程
        y_corrected = fsolve(implicit_correction, y_pred)[0]
        print(f"Corrector step for t = {t_pred:<10.2f}: y_corrected = {y_corrected:<15.8f}")

        # 更新 y_values
        y_values[n + 1] = y_corrected

    return t_values, y_values


# 主程序
def main():
    # 获取用户选择的ODE函数
    f = get_ode_function()

    if f is None:
        return  # 如果函数无法解析，则退出

    # 输入初始条件和参数
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

    # 用户输入已知的 W 值
    _w = input("W values already known? (y/n): ").strip().lower()
    known_w_values = []

    if _w == 'y':
        try:
            # 如果用户选择有已知的 W 值，直接输入它们 (仅输入 W1, W2, W3)
            print("Please input three known W values (W1, W2, W3):")
            for i in range(1, 4):
                w = float(input(f"Please input the value of W{i}: "))
                known_w_values.append(w)
        except ValueError:
            print("Invalid input for W values. Please enter valid numbers.")
            return

    elif _w == 'n':
        # 如果没有已知的 W 值，直接使用 RKF 方法初始化
        init_method = 'Runge-Kutta-Fehlberg'
    else:
        print("Invalid choice for W values. Please enter 'y' or 'n'.")
        return

    # 获取最大的小数位数
    max_decimal_places = 8

    # 设置 Decimal 的精度
    getcontext().prec = max_decimal_places + 7  # 增加一些额外的精度，以便计算时不会丢失精度

    # 求解 ODE
    try:
        # 如果用户选择有已知的 W 值，init_method 设为 None
        if _w == 'y':
            # 确保已输入至少三个 W 值
            if len(known_w_values) < 3:
                print("At least three W values are required when 'y' is selected.")
                return
            t_values, y_values = solve_ode(f, t0, y0, t_end, h, known_w_values=known_w_values, init_method=None)
        else:
            t_values, y_values = solve_ode(f, t0, y0, t_end, h, known_w_values=None, init_method='Runge-Kutta-Fehlberg')
    except Exception as e:
        print(f"An error occurred while solving the ODE: {e}")
        return

    # 输出结果
    print("\nNumerical solution result:")
    print(f"{'t':<15} {'y(t)':<15}")
    for t, y in zip(t_values, y_values):
        # 使用 round 保留指定小数位数
        print(f"{t:<15.8f} {y:<15.8f}")


if __name__ == "__main__":
    main()
