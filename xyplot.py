import matplotlib.pyplot as plt
import matplotlib.pyplot as plt


def xyplot(data):
    x = list(data.keys())
    y = list(data.values())

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b')

    plt.title('Data Visualization')
    plt.xlabel('X values')
    plt.ylabel('Y values')

    plt.grid(True)
    plt.show()
