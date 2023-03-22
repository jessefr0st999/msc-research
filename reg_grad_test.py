import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

x_vec = np.linspace(0, 10, 50)
y1_vec = np.sin(x_vec)
y2_vec = np.sin(x_vec * 1.1)
y3_vec = np.cos(x_vec * 1.1)
y4_vec = 2 * np.cos(x_vec * 1.1)
y5_vec = 2 * np.sin(x_vec * 1.1)
y6_vec = 1.5 * np.sin(x_vec * 1.1)
y7_vec = 0.5 * np.sin(x_vec * 1.1)

def link_strengths(s1, s2):
    return np.min(np.abs([
        LinearRegression().fit(np.column_stack([s1]), s2).coef_[0],
        LinearRegression().fit(np.column_stack([s2]), s1).coef_[0],
    ])), np.abs(np.corrcoef(s1, s2)[0, 1]), \
        np.abs(spearmanr(s1, s2).statistic)

reg_grad_1_2, pearson_1_2, spearman_1_2 = link_strengths(y1_vec, y2_vec)
reg_grad_1_3, pearson_1_3, spearman_1_3 = link_strengths(y1_vec, y3_vec)
reg_grad_1_4, pearson_1_4, spearman_1_4 = link_strengths(y1_vec, y4_vec)
reg_grad_1_5, pearson_1_5, spearman_1_5 = link_strengths(y1_vec, y5_vec)
reg_grad_1_6, pearson_1_6, spearman_1_6 = link_strengths(y1_vec, y6_vec)
reg_grad_1_7, pearson_1_7, spearman_1_7 = link_strengths(y1_vec, y7_vec)

figure, axes = plt.subplots(2, 3)
axes[0, 0].plot(x_vec, y1_vec, 'ko')
axes[0, 0].plot(x_vec, y2_vec, 'ro')
axes[0, 0].set_title(f'reg grad {round(reg_grad_1_2, 4)}, '
    f'pearson {round(pearson_1_2, 4)}, spearman {round(spearman_1_2, 4)}')

axes[0, 1].plot(x_vec, y1_vec, 'ko')
axes[0, 1].plot(x_vec, y3_vec, 'bo')
axes[0, 1].set_title(f'reg grad {round(reg_grad_1_3, 4)}, '
    f'pearson {round(pearson_1_3, 4)}, spearman {round(spearman_1_3, 4)}')

axes[0, 2].plot(x_vec, y1_vec, 'ko')
axes[0, 2].plot(x_vec, y4_vec, 'co')
axes[0, 2].set_title(f'reg grad {round(reg_grad_1_4, 4)}, '
    f'pearson {round(pearson_1_4, 4)}, spearman {round(spearman_1_4, 4)}')

axes[1, 0].plot(x_vec, y1_vec, 'ko')
axes[1, 0].plot(x_vec, y5_vec, 'go')
axes[1, 0].set_title(f'reg grad {round(reg_grad_1_5, 4)}, '
    f'pearson {round(pearson_1_5, 4)}, spearman {round(spearman_1_5, 4)}')

axes[1, 1].plot(x_vec, y1_vec, 'ko')
axes[1, 1].plot(x_vec, y6_vec, 'mo')
axes[1, 1].set_title(f'reg grad {round(reg_grad_1_6, 4)}, '
    f'pearson {round(pearson_1_6, 4)}, spearman {round(spearman_1_6, 4)}')

axes[1, 2].plot(x_vec, y1_vec, 'ko')
axes[1, 2].plot(x_vec, y7_vec, 'yo')
axes[1, 2].set_title(f'reg grad {round(reg_grad_1_7, 4)}, '
    f'pearson {round(pearson_1_7, 4)}, spearman {round(spearman_1_7, 4)}')

axes[0, 0].set_ylim([-2.5, 2.5])
axes[1, 0].set_ylim([-2.5, 2.5])
axes[0, 1].set_ylim([-2.5, 2.5])
axes[1, 1].set_ylim([-2.5, 2.5])
axes[0, 2].set_ylim([-2.5, 2.5])
axes[1, 2].set_ylim([-2.5, 2.5])
plt.show()