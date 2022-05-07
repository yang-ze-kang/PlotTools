# 混淆矩阵
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_confusion_matrix(matrix, label_info=None,
                          xlabel='预测标签', ylabel='真实标签', title='混淆矩阵',
                          font_size_num=20, font_size_xticks=13, font_size_yticks=13,
                          save='a.png', show=True):
    """绘制混淆矩阵"""
    assert matrix is not None
    assert label_info is not None
    num_classes = len(label_info)
    labels_name = label_info.keys()
    plt.imshow(matrix, cmap=plt.cm.Blues)
    plt.xticks(range(num_classes), labels_name,
               rotation=0, fontsize=font_size_xticks)
    plt.yticks(range(num_classes), labels_name, fontsize=font_size_yticks)
    plt.colorbar()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    thresh = matrix.max() / 2
    for x in range(num_classes):
        for y in range(num_classes):
            info = int(matrix[y, x])
            plt.text(x, y, info, fontsize=font_size_num,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")
    plt.tight_layout()
    if save:
        plt.savefig(save)
    if show:
        plt.show()


if __name__ == '__main__':
    matrix = np.array([
        [40, 0, 0, 0, 1],
        [2, 13, 3, 1, 0],
        [5, 5, 4, 0, 2],
        [5, 1, 0, 4, 2],
        [1, 1, 0, 1, 6]
    ])
    plot_confusion_matrix(matrix,
                          label_info={
                              '神经源性瘤': 0,
                              '骨髓瘤': 1,
                              '骨巨细胞瘤': 2,
                              '脊索瘤': 3,
                              'langerhans': 4
                          }
                          )
