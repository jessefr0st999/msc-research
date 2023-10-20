import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1)

n = 200
t_vec = np.linspace(0, 50, n)
processes = []
params = [
    (3, 0, 1, 0.5),
    (3, 0, 0.25, 0.5),
    (3, 0, 4, 0.5),
    (3, 0, 1, 0.125),
    (3, 0, 1, 2),
    (3, 0, 0.25, 0.125),
    (3, 0, 4, 2),
]
for x0, beta, k, v in params:
    process = np.zeros(n)
    process[0] = x0
    for i in range(n):
        if i == 0:
            continue
        dt = t_vec[i] - t_vec[i - 1]
        process[i] = process[i - 1] + k * (beta - process[i - 1]) * dt + \
            np.sqrt(v) * np.random.normal(0, np.sqrt(dt))
    processes.append(process)

processes_to_plot = [
    [1, 0, 2],
    [3, 0, 4],
    # [5, 0, 6],
]
colours = ['blue', 'red', 'green', 'magenta', 'orange', 'gold', 'grey']
# figure, axes = plt.subplots(1, len(processes_to_plot))
figure, axes = plt.subplots(len(processes_to_plot), 1)
axes = iter(axes.flatten())
for param_sets in processes_to_plot:
    axis = next(axes)
    for p in param_sets:
        x0, beta, k, v = params[p]
        axis.plot(t_vec, processes[p], color=colours[p],
            label=f'k = {k}, v = {v}')
        axis.set_ylim([-2.5, 3.5])
    axis.legend()
plt.show()