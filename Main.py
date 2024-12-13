import numpy as np
from scipy.optimize import fsolve
from decimal import Decimal, getcontext

# 设置 decimal 的默认精度（例如设置精度为50位，之后会根据需要调整）
getcontext().prec = 50


# 用户自定义 ODE 函数
def get_ode_function():
    """
    提示用户输入一个 ODE 函数，并返回该函数。
    示例输入：dy/dt = -2y
    """
    print("choose a formula (function input format is f(t, y))")
    print("1. dy/dt = -2y")
    print("2. dy/dt = t * sqrt(y)")
    print("3. dy/dt = y * (1 - y)  (e.g. population models)")
    print("4. dy/dt = y - t^2 + 1 (equations in questions)")
    print("5. customized function")

    choice = input("please input your option (1/2/3/4/5): ")

    if choice == '1':
        def f(t, y):
            return Decimal(-2) * y  # 示例：dy/dt = -2y

        return f

    elif choice == '2':
        def f(t, y):
            return Decimal(t) * Decimal(y).sqrt()  # 示例：dy/dt = t * sqrt(y)

        return f

    elif choice == '3':
        def f(t, y):
            return Decimal(y) * (Decimal(1) - Decimal(y))  # 示例：dy/dt = y * (1 - y) (用于人口模型)

        return f

    elif choice == '4':
        # 这是问题中的微分方程: y' = y - t^2 + 1
        def f(t, y):
            return Decimal(y) - Decimal(t) ** 2 + Decimal(1)

        return f

    elif choice == '5':
        print("please input your customized function (e.g. t*y - y**2 + 3*t)：")
        expr = input("please input the expression: ")

        # 用 eval 来转换用户输入的表达式为一个 lambda 函数
        def f(t, y):
            return eval(expr, {"t": Decimal(t), "y": Decimal(y)})

        return f

    else:
        print("invalid selection. By default use dy/dt = -2y。")

        def f(t, y):
            return Decimal(-2) * y

        return f


# 求解常微分方程（ODE）的数值解
def solve_ode(f, t0, y0, t_end, h, known_w_values=None, method='Adams-Bash forth'):
    """
    使用4阶 Adams-Bash forth 或 Adams-Moulton 方法求解常微分方程。

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
    method : str, optional
        选择算法，'Adams-Bash forth' 或 'Adams-Moulton'

    返回：
    t_values : numpy.ndarray
        时间点数组
    y_values : numpy.ndarray
        解数组
    """
    # 将所有输入转换为 Decimal 类型，以确保精度
    t0 = Decimal(t0)
    y0 = Decimal(y0)
    h = Decimal(h)
    t_end = Decimal(t_end)

    # 创建时间点数组
    t_values = np.arange(float(t0), float(t_end), float(h))
    n_steps = len(t_values)

    # 将 y_values 数组初始化为 Decimal 类型的数组
    y_values = np.array([Decimal(0)] * n_steps)

    if known_w_values is not None and len(known_w_values) > 0:
        # 如果有已知的w值，从已知w值开始
        for i in range(len(known_w_values)):
            y_values[i] = Decimal(known_w_values[i])
    else:
        # 否则，从初始条件开始计算
        y_values[0] = y0

    # 初始条件（使用 Euler 方法或者 Runge-Kutta 方法计算初始值）
    if len(known_w_values) < 4:  # 如果已知的w值少于4个，需要计算额外的步骤
        for i in range(len(known_w_values), 4):
            y_values[i] = y_values[i - 1] + h * f(t_values[i - 1], y_values[i - 1])

    # 根据选择的算法执行
    if method == 'Adams-Bash forth':
        for n in range(3, n_steps - 1):
            y_values[n + 1] = y_values[n] + (h / Decimal(24)) * (
                    Decimal(55) * f(t_values[n], y_values[n]) -
                    Decimal(59) * f(t_values[n - 1], y_values[n - 1]) +
                    Decimal(37) * f(t_values[n - 2], y_values[n - 2]) -
                    Decimal(9) * f(t_values[n - 3], y_values[n - 3])
            )

    elif method == 'Adams-Moulton':
        for n in range(3, n_steps - 1):
            # 预测步，使用 Adams-Bash forth 方法
            y_pred = y_values[n] + (h / Decimal(24)) * (
                    Decimal(55) * f(t_values[n], y_values[n]) -
                    Decimal(59) * f(t_values[n - 1], y_values[n - 1]) +
                    Decimal(37) * f(t_values[n - 2], y_values[n - 2]) -
                    Decimal(9) * f(t_values[n - 3], y_values[n - 3])
            )

            # 修正步，使用 Adams-Moulton 方法
            def implicit_correction(y_next):
                return y_next - y_values[n] - (h / Decimal(720)) * (
                        Decimal(251) * f(t_values[n], y_next) +
                        Decimal(646) * f(t_values[n - 1], y_values[n - 1]) -
                        Decimal(264) * f(t_values[n - 2], y_values[n - 2]) +
                        Decimal(106) * f(t_values[n - 3], y_values[n - 3]) -
                        Decimal(19) * f(t_values[n - 4], y_values[n - 4])
                )

            # 使用 f-solve 解决隐式方程
            y_values[n + 1] = fsolve(implicit_correction, y_pred)[0]

    else:
        raise ValueError("Invalid method. Choose either 'Adams-Bashforth' or 'Adams-Moulton'.")

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


# 用户选择算法：1 或 2
def choose_method():
    print("please choose algorithm：")
    print("1. Adams-Bash forth (Explicit)")
    print("2. Adams-Moulton (Implicit)")

    choice = input("please input option (1 or 2): ")

    if choice == '1':
        return 'Adams-Bash forth'
    elif choice == '2':
        return 'Adams-Moulton'
    else:
        print("invalid option. By default use Adams-Bash forth method")
        return 'Adams-Bash forth'


# 主程序
def main():
    # 获取用户选择的ODE函数
    f = get_ode_function()

    # 输入初始条件和参数
    t0 = float(input("please input initial time t0: "))
    y0 = float(input("please input initial condition y0: "))
    t_end = float(input("please input end time t_end: "))
    h = float(input("please input step length h: "))

    # 用户输入已知的 w 值
    _w = input("W values already known? (y/n): ").strip().lower()
    known_w_values = []

    if _w == 'y':
        num_known_w = int(input("input the number of known W values: "))
        for i in range(num_known_w):
            w = float(input(f"please input the value of w{i + 1}: "))
            known_w_values.append(w)

    # 获取最大的小数位数
    max_decimal_places = get_max_decimal_places(known_w_values)

    # 设置 Decimal 的精度
    getcontext().prec = max_decimal_places + 5  # 增加一些额外的精度，以便计算时不会丢失精度

    # 选择方法
    method = choose_method()

    # 求解 ODE
    t_values, y_values = solve_ode(f, t0, y0, t_end, h, known_w_values, method)

    # 输出结果
    print("\nnumerical solution result：")
    print(f"{'t':<10} {'y(t)':<10}")
    for t, y in zip(t_values, y_values):
        # 使用 round 保留指定小数位数
        print(f"{t:<10.2f} {round(y, max_decimal_places):<10.6f}")


if __name__ == "__main__":
    main()
