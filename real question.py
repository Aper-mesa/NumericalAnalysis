import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

# 提供的更新数据，年份、人口和增长率
years = np.array([2020, 2019, 2018, 2017, 2016, 2015, 2014, 2013, 2012, 2011,
                  2010, 2009, 2008, 2007, 2006, 2005, 2004, 2003, 2002, 2001, 2000])
populations = np.array([1396387127, 1383112050, 1369003306, 1354195680, 1338636340, 1322866505, 1307246509,
                        1291132063, 1274487215, 1257621191, 1240613620, 1223640160, 1206734806, 1189691809,
                        1172373788, 1154638713, 1136264583, 1117415123, 1098313039, 1078970907, 1059633675])
growth_rates = np.array([0.0096, 0.0103, 0.0109, 0.0116, 0.0119, 0.0119, 0.0125, 0.0131, 0.0134, 0.0137,
                         0.0139, 0.0140, 0.0143, 0.0148, 0.0154, 0.0162, 0.0169, 0.0174, 0.0179, 0.0182, 0.0184])

# 计算增长率r的均值作为拟合的参考增长率
average_growth_rate = np.mean(growth_rates)
print(f"Average growth rate: {average_growth_rate:.6f}")

# 假设最大承载能力K为2020年人口的2倍
K = 2 * populations[0]

# 定义Logistic增长ODE
def logistic_model(t, y, r, K):
    dydt = r * y * (1 - y / K)
    return dydt

# 使用Predictor-Corrector方法求解ODE
def solve_ode(f, t0, y0, t_end, h, r, K):
    # 时间范围
    t_values = np.arange(t0, t_end, h)
    y_values = np.zeros(len(t_values))
    y_values[0] = y0

    # 初始化前三步使用 Runge-Kutta-Fehlberg 方法
    for i in range(1, 4):
        t_current = t_values[i - 1]
        y_current = y_values[i - 1]
        k1 = h * f(t_current, y_current, r, K)
        k2 = h * f(t_current + h / 4, y_current + (1 / 4) * k1, r, K)
        k3 = h * f(t_current + 3 * h / 8, y_current + (3 / 32) * k1 + (9 / 32) * k2, r, K)
        k4 = h * f(t_current + 12 * h / 13, y_current + (1932 / 2197) * k1 - (7200 / 2197) * k2 + (7296 / 2197) * k3, r, K)
        k5 = h * f(t_current + h, y_current + (439 / 216) * k1 - (8 / 21) * k2 + (3680 / 513) * k3 - (845 / 4104) * k4,
                   r, K)
        k6 = h * f(t_current + h / 2,
                   y_current - (8 / 27) * k1 + (2 / 9) * k2 - (3544 / 2565) * k3 + (1859 / 4104) * k4 - (11 / 40) * k5,
                   r, K)
        y_values[i] = y_current + (16 * k1 / 135 + 6656 * k3 / 12825 + 28561 * k4 / 56430 - 9 * k5 / 50 + 2 * k6 / 55)

    # 使用 Predictor-Corrector 方法继续求解
    for n in range(3, len(t_values) - 1):
        t_n = t_values[n]
        y_n = y_values[n]
        t_pred = t_n + h
        y_pred = y_n + (h / 24) * (
                    55 * f(t_n, y_n, r, K) - 59 * f(t_n - h, y_values[n - 1], r, K) + 37 * f(t_n - 2 * h, y_values[n - 2],
                                                                                       r, K) - 9 * f(t_n - 3 * h,
                                                                                                  y_values[n - 3], r, K))

        # 修正步骤
        def implicit_correction(y_next):
            return y_next - y_n - (h / 720) * (
                        251 * f(t_pred, y_next, r, K) + 646 * f(t_n, y_n, r, K) - 264 * f(t_n - h, y_values[n - 1],
                                                                                    r, K) + 106 * f(t_n - 2 * h,
                                                                                                 y_values[n - 2],
                                                                                                 r, K) - 19 * f(
                    t_n - 3 * h, y_values[n - 3], r, K))

        # 使用 fsolve 修正预测
        y_corrected = fsolve(implicit_correction, y_pred)[0]
        y_values[n + 1] = y_corrected

    return t_values, y_values

# 使用2020年人口作为初始条件，预测2020到2025年
t0 = 2020  # 从2020年开始预测
y0 = populations[0]  # 2020年的人口
t_end = 2024  # 预测到2025年
h = 0.01  # 时间步长，0.01年（每年100步）

# 使用之前定义的ODE求解函数
t_values, y_values = solve_ode(logistic_model, t0, y0, t_end, h, average_growth_rate, K)

# 绘制结果
plt.plot(t_values, y_values, label="Predicted Population (2020-2025)", color='blue')
plt.scatter(years, populations, color='red', label="Observed Data", zorder=5)
plt.title("Logistic Growth Population Prediction (2020-2024)")
plt.xlabel("Year")
plt.ylabel("Population")
plt.legend()
plt.grid(True)
plt.show()

# 打印预测结果
predicted_years = [2021, 2022, 2023, 2024]
for year in predicted_years:
    # 寻找接近目标年份的 t_values
    idx = np.argmin(np.abs(t_values - year))  # 找到最接近的年份索引
    print(f"Predicted population in {year}: {y_values[idx]:,.0f}")
