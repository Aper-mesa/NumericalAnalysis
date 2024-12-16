from decimal import getcontext
import numpy as np
import sympy as sp
from scipy.optimize import fsolve

# 设置 decimal 的默认精度（例如设置精度为50位，之后会根据需要调整）
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
        print("You can use the following mathematical functions:")
        print("  - Trigonometric functions: sin(y), cos(y), tan(y), etc.")
        print("  - Exponential and logarithmic functions: exp(y), log(y) [natural log], ln(y)")
        print("  - Power functions: y**2, t**3, etc.")
        print("  - Other functions: sqrt(y), etc.")
        print("Use 't' and 'y' as variables. For division, use '/' (e.g., y/t).")
        print("Examples:")
        print("  - t*sin(y) + cos(t)")
        print("  - y/t + log(y)")
        print("  - exp(t) - y**2")

        while True:
            expr = input("Please input your customized expression for dy/dt: ").strip()

            # 检查用户是否想要退出
            if expr.lower() in ['exit', 'quit']:
                print("Exiting function input.")
                return None

            # 使用 sympy 来解析用户输入的表达式
            t, y = sp.symbols('t y')

            # 定义允许的函数和符号
            allowed_functions = {
                'sin': sp.sin,
                'cos': sp.cos,
                'tan': sp.tan,
                'exp': sp.exp,
                'log': sp.log,  # 自然对数
                'sqrt': sp.sqrt,
                'Abs': sp.Abs,
                'abs': sp.Abs,
                'ln': sp.log,
            }
            allowed_symbols = {'t', 'y'}

            try:
                # 使用 sympy 解析用户输入的公式
                ode_expr = sp.sympify(expr, locals=allowed_functions)

                # 检查是否只包含允许的符号
                symbols_in_expr = ode_expr.free_symbols
                if not symbols_in_expr.issubset({t, y}):
                    raise ValueError(f"Only 't' and 'y' variables are allowed. Found symbols: {symbols_in_expr}")

                # 转换为 Python 函数以便数值计算
                ode_func = sp.lambdify((t, y), ode_expr, modules=["numpy"])

                # 测试函数是否可以正常计算
                test_val = ode_func(1, 1)
                if not isinstance(test_val, (int, float, np.floating, np.integer)):
                    raise ValueError("The expression does not evaluate to a numerical value.")

                return ode_func

            except sp.SympifyError:
                print("Error: The expression could not be parsed. Please check the syntax.")
            except ValueError as ve:
                print(f"Error: {ve}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            print("Please try entering the expression again or type 'exit' to cancel.")

    else:
        print("Invalid selection. By default use dy/dt = -2y.")

        def f(t, y):
            return -2 * y  # 默认选择 dy/dt = -2y

        return f


# 以下是原始代码的其他部分，未作修改
# 求解常微分方程（ODE）的数值解
def solve_ode(f, t0, y0, t_end, h, known_w_values=None, init_method=None):
    """
    使用龙格-库塔-费尔伯格方法（Runge-Kutta-Fehlberg method）求解常微分方程。
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
        已知的初始条件列表
    init_method : str, optional
        选择初始化方法，'Runge-Kutta'
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

    # 创建时间点数组，确保包含 t_end
    t_values = np.linspace(t0, t_end, num=int((t_end - t0) / h) + 1)
    n_steps = len(t_values)

    # 将 y_values 数组初始化为 float 类型的数组
    y_values = np.array([0.0] * n_steps)

    # 如果有已知的 w 值，使用它们来初始化 y 值
    if known_w_values is not None and len(known_w_values) > 0:
        # 从已知的 w 值开始
        for i in range(len(known_w_values)):
            y_values[i] = known_w_values[i]
    else:
        # 否则，根据初始化方法进行初始化
        y_values[0] = y0

        if init_method == 'Runge-Kutta':
            # 使用龙格-库塔-费尔伯格法（Runge-Kutta-Fehlberg Method）来计算初值
            for i in range(1, 4):
                t = t_values[i - 1]
                y = y_values[i - 1]

                # 计算 RKF 系数
                k1 = h * f(t, y)
                k2 = h * f(t + h / 4, y + (1 / 4) * k1)
                k3 = h * f(t + 3 * h / 8, y + (3 / 32) * k1 + (9 / 32) * k2)
                k4 = h * f(t + 12 * h / 13, y + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3)
                k5 = h * f(t + h, y + (439 / 216) * k1 - (8 / 21) * k2 + (3680 / 513) * k3 - (845 / 4104) * k4)
                k6 = h * f(t + h / 2,
                           y - (8 / 27) * k1 + (2 / 9) * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5)

                # 计算预测值和修正值
                y_rk4 = y + (k1 + 4 * k2 + k3) / 6
                y_rk5 = y + (k1 + 4 * k2 + k3 + k4 + k5) / 6

                # 将修正后的值赋给 y_values
                y_values[i] = y_rk4

        else:
            raise ValueError("Invalid initialization method. Choose 'Runge-Kutta'.")

    # 预测-修正步骤（结合 Adams-Bashforth 和 Adams-Moulton）
    for n in range(3, n_steps - 1):
        # 预测步骤（使用 Adams-Bashforth 方法）
        y_pred = y_values[n] + (h / 24) * (
                55 * f(t_values[n], y_values[n]) -
                59 * f(t_values[n - 1], y_values[n - 1]) +
                37 * f(t_values[n - 2], y_values[n - 2]) -
                9 * f(t_values[n - 3], y_values[n - 3])
        )

        # 打印预测步骤的 y_pred 值
        print(f"Prediction step for t = {t_values[n + 1]:<10.2f}: y_pred = {y_pred:<10.8f}")

        # 修正步骤（使用 Adams-Moulton 方法）
        def implicit_correction(y_next):
            return y_next - y_values[n] - (h / 720) * (
                    251 * f(t_values[n] + h, y_next) +
                    646 * f(t_values[n], y_values[n]) -
                    264 * f(t_values[n - 1], y_values[n - 1]) +
                    106 * f(t_values[n - 2], y_values[n - 2]) -
                    19 * f(t_values[n - 3], y_values[n - 3])
            )

        # 使用 fsolve 解决隐式方程
        y_values[n + 1] = fsolve(implicit_correction, y_pred)[0]

        # 打印修正后的 y_values[n + 1] 值
        print(f"Corrector step for t = {t_values[n + 1]:<10.2f}: y_corrected = {y_values[n + 1]:<10.8f}")

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
    except ValueError:
        print("Invalid input for numerical parameters. Please enter valid numbers.")
        return

    # 用户输入已知的 w 值
    _w = input("W values already known? (y/n): ").strip().lower()
    known_w_values = []

    if _w == 'y':
        try:
            # 如果用户选择有已知的 W 值，直接输入它们
            known_w_values.append(y0)  # W0 = y0，作为初始值
            for i in range(3):
                w = float(input(f"Please input the value of w{i + 1}: "))
                known_w_values.append(w)
            # 由于已知W值，跳过初始化方法的选择
            init_method = None
        except ValueError:
            print("Invalid input for W values. Please enter valid numbers.")
            return

    elif _w == 'n':
        # 如果没有已知的 W 值，直接使用 RKF 方法
        init_method = 'Runge-Kutta'
    else:
        print("Invalid choice for W values. Please enter 'y' or 'n'.")
        return

    # 获取最大的小数位数
    max_decimal_places = 8

    # 设置 Decimal 的精度
    getcontext().prec = max_decimal_places + 7  # 增加一些额外的精度，以便计算时不会丢失精度

    # 求解 ODE
    try:
        t_values, y_values = solve_ode(f, t0, y0, t_end, h, known_w_values, init_method)
    except Exception as e:
        print(f"An error occurred while solving the ODE: {e}")
        return

    # 输出结果
    print("\nNumerical solution result:")
    print(f"{'t':<10} {'y(t)':<10}")
    for t, y in zip(t_values, y_values):
        # 使用 round 保留指定小数位数
        print(f"{t:<10.2f} {round(y, max_decimal_places):<10.8f}")


if __name__ == "__main__":
    main()
