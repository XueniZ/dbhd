import matplotlib.pyplot as plt

# 假设你有两个变量x和y，每个变量包含一组数据点
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 3, 5, 7, 9]

# 绘制两条曲线
plt.plot(x, y1, label='Variable 1')
plt.plot(x, y2, label='Variable 2')

# 添加图例
plt.legend()

# 添加标题和轴标签
plt.title('Multiple Variables in One Graph')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# 显示图形
plt.show()
