import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def draw_muti_class_bar(x, y, hue,
                        ylim=None, decimal_num=2,
                        font_scale_theme=1.4, font_size_bar_val=12,
                        title='', axi_label=['', ''],
                        legend_title='', legend_out=True,
                        show_y_value=False,
                        tight_layout=False,
                        save=''):
    assert x is not None
    assert y is not None
    assert hue is not None
    a, b = len(x), len(hue)
    assert y.shape[0] == b and y.shape[1] == a
    x = x * b
    hue = [h for h in hue for _ in range(a)]
    y = y.flatten()
    df = pd.DataFrame(data={
        "类别": hue,
        "x值": x,
        "y值": y

    })

    def show_values_on_bars(axs):
        def _show_on_single_plot(ax):
            for p in ax.patches:
                _x = p.get_x() + p.get_width() / 2
                _y = p.get_y() + p.get_height()
                value = (f"%.{decimal_num}f" % p.get_height())
                ax.text(_x, _y, value, ha="center", fontsize=font_size_bar_val)
        if isinstance(axs, np.ndarray):
            for idx, ax in np.ndenumerate(axs):
                _show_on_single_plot(ax)
        else:
            _show_on_single_plot(axs)
    plt.rcParams['axes.unicode_minus'] = False
    sns.set(font_scale=font_scale_theme)
    sns.set_style('whitegrid', {'font.sans-serif': ['simhei', 'FangSong']})
    g = sns.catplot(
        data=df, kind="bar",
        x="x值", y="y值", hue="类别",
        ci="sd", palette="dark", alpha=.6, height=6,
        legend_out=legend_out
    )
    plt.title(title)
    g.despine(left=True)
    if ylim != None:
        g.set(ylim=ylim)
    g.set_axis_labels(axi_label[0], axi_label[1])
    g.legend.set_title(legend_title)
    if show_y_value:
        show_values_on_bars(g.ax)
    if tight_layout:
        plt.tight_layout()
    if save != '':
        plt.savefig(save)
    plt.show()


if __name__ == '__main__':
    x = ['神经源性瘤', '骨髓瘤', '骨巨细胞瘤', '脊索瘤', 'langerhans']
    hue = ['轴位', '矢状位']
    y = np.array([[73.58, 61.90, 30.00, 60.00, 75.00],
                  [71.74, 42.86, 54.55, 62.50, 10.00]])

    draw_muti_class_bar(x=x,
                        y=y,
                        hue=hue,
                        title='轴位和矢状位各个亚型的精确率',
                        axi_label=["肿瘤亚型", "精确率(%)"],
                        legend_title="模态",
                        show_y_value=True)
