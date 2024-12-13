import numpy as np
from scipy.optimize import fsolve
from decimal import Decimal, getcontext
import sympy as sp

# 设置 decimal 的默认精度（例如设置精度为50位，之后会根据需要调整）
getcontext().prec = 50

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

    choice = input("Please input your option (1/2/3/4/5): ")

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
        print("You can use mathematical functions like sin, cos, exp, log, etc.")
        print("Please input your formula in terms of t and y (e.g., t*sin(y) + cos(t))")
        expr = input("Please input your customized expression for dy/dt: ")

        # 使用 sympy 来解析用户输入的表达式
        t, y = sp.symbols('t y')

        try:
            # 使用 sympy 解析用户输入的公式
            ode_expr = sp.sympify(expr)
            # 对公式进行微分，得到 dy/dt
            ode_dydt = sp.diff(ode_expr, t)

            # 转换为 Python 函数以便数值计算
            ode_func = sp.lambdify((t, y), ode_dydt, 'numpy')

            return ode_func

        except Exception as e:
            print(f"Error in parsing the expression: {e}")
            return None

    else:
        print("Invalid selection. By default use dy/dt = -2y.")
        def f(t, y):
            return -2 * y  # 默认选择 dy/dt = -2y
        return f


# 求解常微分方程（ODE）的数值解
def solve_ode(f, t0, y0, t_end, h, known_w_values=None, init_method=None):
    """
    使用多步法（预测-修正法）求解常微分方程。
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
        选择初始化方法，'Euler' 或 'Runge-Kutta'
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

    # 创建时间点数组
    t_values = np.arange(t0, t_end, h)
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

        if init_method == 'Euler':
            # 使用欧拉法（Euler Method）来计算初值
            for i in range(1, 4):
                y_values[i] = y_values[i - 1] + h * f(t_values[i - 1], y_values[i - 1])

        elif init_method == 'Runge-Kutta':
            # 使用四阶龙格-库塔法（4th-order Runge-Kutta Method）来计算初值
            for i in range(1, 4):
                t = t_values[i - 1]
                y = y_values[i - 1]
                k1 = h * f(t, y)
                k2 = h * f(t + h / 2, y + k1 / 2)
                k3 = h * f(t + h / 2, y + k2 / 2)
                k4 = h * f(t + h, y + k3)
                y_values[i] = y + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        else:
            raise ValueError("Invalid initialization method. Choose 'Euler' or 'Runge-Kutta'.")

    # 预测-修正步骤（结合 Adams-Bashforth 和 Adams-Moulton）
    for n in range(3, n_steps - 1):
        # 预测步骤（使用 Adams-Bashforth 方法）
        y_pred = y_values[n] + (h / 24) * (
                55 * f(t_values[n], y_values[n]) -
                59 * f(t_values[n - 1], y_values[n - 1]) +
                37 * f(t_values[n - 2], y_values[n - 2]) -
                9 * f(t_values[n - 3], y_values[n - 3])
        )

        # 修正步骤（使用 Adams-Moulton 方法）
        def implicit_correction(y_next):
            return y_next - y_values[n] - (h / 720) * (
                    251 * f(t_values[n], y_next) +
                    646 * f(t_values[n - 1], y_values[n - 1]) -
                    264 * f(t_values[n - 2], y_values[n - 2]) +
                    106 * f(t_values[n - 3], y_values[n - 3]) -
                    19 * f(t_values[n - 4], y_values[n - 4])
            )

        # 使用 fsolve 解决隐式方程
        y_values[n + 1] = fsolve(implicit_correction, y_pred)[0]

    return t_values, y_values


# 获取最大的小数位数
def get_max_decimal_places(known_w_values):
    """
    获取已知w值的最大小数位数
    """
    if known_w_values is not None and len(known_w_values) > 0:
        max_decimal_places = max(len(str(val).split('.')[-1]) for val in known_w_values if '.' in str(val))
        return max_decimal_places
    return 8  # 默认值为 8 位小数


# 主程序
def main():
    # 获取用户选择的ODE函数
    f = get_ode_function()

    if f is None:
        return  # 如果函数无法解析，则退出

    # 输入初始条件和参数
    t0 = float(input("Please input initial time t0: "))
    y0 = float(input("Please input initial condition y0: "))
    t_end = float(input("Please input end time t_end: "))
    h = float(input("Please input step length h: "))

    # 用户输入已知的 w 值
    _w = input("W values already known? (y/n): ").strip().lower()
    known_w_values = []

    if _w == 'y':
        # 如果用户选择有已知的 W 值，直接输入它们
        known_w_values.append(y0)  # W0 = y0，作为初始值
        for i in range(3):
            w = float(input(f"Please input the value of w{i + 1}: "))
            known_w_values.append(w)

        # 由于已知W值，跳过初始化方法的选择
        init_method = None

    elif _w == 'n':
        # 如果没有已知的 W 值，要求用户选择初始化方法
        print("Please choose the initialization method:")
        print("1. Euler Method")
        print("2. Runge-Kutta (4th-order) Method")
        init_choice = input("Please input your choice (1 or 2): ")

        if init_choice == '1':
            init_method = 'Euler'
        elif init_choice == '2':
            init_method = 'Runge-Kutta'
        else:
            print("Invalid choice. Using Euler Method by default.")
            init_method = 'Euler'
    else:
        print("Invalid choice.")

    # 获取最大的小数位数
    max_decimal_places = get_max_decimal_places(known_w_values)

    # 设置 Decimal 的精度
    getcontext().prec = max_decimal_places + 5  # 增加一些额外的精度，以便计算时不会丢失精度

    # 求解 ODE
    t_values, y_values = solve_ode(f, t0, y0, t_end, h, known_w_values, init_method)

    # 输出结果
    print("\nNumerical solution result:")
    print(f"{'t':<10} {'y(t)':<10}")
    for t, y in zip(t_values, y_values):
        # 使用 round 保留指定小数位数
        print(f"{t:<10.2f} {round(y, max_decimal_places):<10.6f}")


if __name__ == "__main__":
    main()
