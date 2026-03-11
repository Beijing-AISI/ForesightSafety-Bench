from random import shuffle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns
import colorcet as cc


def plot_settings():
    sns.reset_defaults()

    # plt.rcParams['font.family'] = 'Arial'  #
    sns.reset_defaults()
    sns.set_theme(context='paper', style='ticks')

    plt.rcParams.update({
        # 'font.family': 'Times New Roman',
        'font.size': 9,
        'axes.labelsize': 8,
        'axes.titlesize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
        'axes.linewidth': 0.5,
        'grid.linewidth': 0.5,
        'lines.linewidth': 0.5,
        'legend.frameon': False,
        'savefig.dpi': 300,
        'figure.dpi': 150,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5
    })


def create_radar_chart(df, savepath):
    subject = df["subject"].unique().tolist()
    # shuffle(subject)
    df = df.groupby(["model_name", "subject"])["asr"].mean().reset_index()

    # Calculate overall mean for each model
    model_means = df.groupby("model_name")["asr"].mean().reset_index()
    model_means.columns = ["model_name", "model_mean_asr"]

    # Merge back with original data
    df = df.merge(model_means, on="model_name")

    # Sort by the mean value (descending for highest first)
    df = df.sort_values("model_mean_asr", ascending=False)

    print(df.head())

    # 创建雷达图
    fig = plt.figure(figsize=(15, 6))
    ax = fig.add_subplot(111, polar=True)

    # 调色盘
    model_names = df["model_name"].unique().tolist()
    palette = sns.color_palette(cc.glasbey, n_colors=len(model_names))

    # 设置角度和绘图方向
    N = len(subject)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图

    # 绘制网格和标签
    plt.xticks(angles[:-1], subject)  # , size=10

    # 自适应设置y轴刻度
    max_value = 1.25 * df["asr"].max()
    # 确保最大值有一点余量
    max_value = min(100, max_value * 1.1)  # 不超过100%

    # 根据最大值选择合适的刻度间隔
    if max_value <= 20:
        yticks = np.arange(0, max_value + 5, 5)
    elif max_value <= 50:
        yticks = np.arange(0, max_value + 10, 10)
    elif max_value <= 100:
        yticks = np.arange(0, max_value + 20, 20)
    else:
        yticks = np.arange(0, max_value + 25, 25)

    ylabels = [f"{int(y)}%" for y in yticks]

    plt.yticks(yticks, ylabels, color="grey", size=10)
    plt.ylim(0, max_value)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subject, fontsize=12, ha='center', fontweight='bold')
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        label.set_y(label.get_position()[1] - 0.1)  # 向外或向内移动

    for i, model_name in enumerate(model_names):
        df_model = df[df["model_name"] == model_name]
        # 绘制数据
        values = df_model["asr"].tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color=palette[i], label=model_name, alpha=0.6)

    # 添加图例和标题
    plt.legend(loc='upper left', bbox_to_anchor=(1.2, 1), prop={'weight': 'bold', 'size': 12})

    # 美化网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


def create_bar_chart(df, savepath, mode="avg"):
    df['not_has_defense'] = (df['defense_method'].isna() | (df['defense_method'] == 'NoneDefender'))

    assert mode in ["avg", "min"]
    if mode == "avg":
        # 计算每个模型和防御组合的平均值
        group_avg = df.groupby(['model_name', 'not_has_defense'])["asr"].mean().reset_index()
    elif mode == "min":
        # 提取防御效果最好的结果
        group_avg = df.groupby(['model_name', 'not_has_defense', 'defense_method'])["asr"].mean().reset_index()
        group_avg = group_avg.groupby(['model_name', 'not_has_defense'])["asr"].min().reset_index()

    def get_sort_value(model):
        with_defense = group_avg[(group_avg['model_name'] == model) & (group_avg['not_has_defense'])]["asr"]
        if not with_defense.empty:
            return with_defense.values[0]
        return group_avg[(group_avg['model_name'] == model) & (group_avg['not_has_defense'])]["asr"].values[0]

    # 获取不重复的模型名称列表
    unique_models = df['model_name'].unique().tolist()
    # 根据模型的值（优先选择有防御的）进行排序
    model_order = sorted(unique_models, key=get_sort_value)

    # plt.figure(figsize=(10, 4))

    # 使用catplot创建水平条形图
    g = sns.catplot(
        data=group_avg,
        x="model_name",
        y="asr",
        hue="not_has_defense",
        palette=["#4575b4", "#d73027", ],  # 红色表示无防御，蓝色表示有防御
        kind="bar",
        # estimator=np.mean,
        errorbar=None,
        order=model_order,
        hue_order=[False, True],
        legend_out=False,
        height=3,
        aspect=4 / 3
    )

    # 提取轴对象
    ax = g.axes[0, 0]

    # 将Y轴移到右侧
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")

    y_max = ax.get_ylim()[1]
    # 根据数据范围设置合适的刻度间隔
    y_ticks_step = 20 if y_max > 80 else 10  # 如果最大值超过80，则使用20为间隔，否则使用10
    ax.set_yticks(np.arange(0, y_max + y_ticks_step, y_ticks_step))
    ax.set_yticklabels([f'{int(y)}%' for y in np.arange(0, y_max + y_ticks_step, y_ticks_step)])

    # 为每个柱子添加数值标签
    for i, p in enumerate(ax.patches):
        height = p.get_height()

        if height == 0.:
            continue

        ax.text(
            p.get_x() + p.get_width() / 2,  # 水平位置（居中于柱子）
            height + 1.0,  # 垂直位置（略高于柱子）
            f'{height:.1f}%',  # 显示ASR值，保留一位小数
            ha='center',  # 水平对齐方式
            va='bottom',  # 垂直对齐方式
            # rotation=90,
            fontsize=7,  # 字体大小
            color='black'  # 文字颜色
        )

    # 修改图例显示为色块而不是横线
    handles, labels = ax.get_legend_handles_labels()
    g.legend.remove()  # 移除原有图例
    ax.legend(handles, ['w/ Defense', 'w/o Defense'],
              loc='upper right', frameon=False, fancybox=True, shadow=False,
              ncol=1, bbox_to_anchor=(0.40, 0.95))

    # ax.set_title(f"ASR (%) with / without Defense", loc='left')
    ax.set_ylabel('')
    ax.set_xlabel('')
    # plt.xticks(rotation=90, ha='center')  # 旋转x轴标签以提高可读性

    # 设置轴线宽度
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # 移除左侧边框线
    ax.spines['left'].set_visible(False)
    # 添加右侧边框线
    ax.spines['right'].set_visible(True)

    sns.despine(fig=g.fig, left=True, right=False)  # 修改despine参数，保留右侧而不是左侧

    plt.savefig(savepath, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()


def create_radar_chart_2(df, savepath, mode="avg"):
    df['not_has_defense'] = (df['defense_method'].isna() | (df['defense_method'] == 'NoneDefender'))

    assert mode in ["avg", "min"]
    if mode == "avg":
        # 计算每个模型和防御组合的平均值
        jailbreak_avg = df.groupby(['subject', 'not_has_defense'])["asr"].mean().reset_index()
    elif mode == "min":
        # 提取防御效果最好的结果
        jailbreak_avg = df.groupby(['subject', 'not_has_defense', 'defense_method'])["asr"].mean().reset_index()
        jailbreak_avg = jailbreak_avg.groupby(['subject', 'not_has_defense'])["asr"].min().reset_index()

    # 确保ASR以百分比形式表示 (0-100)
    if jailbreak_avg["asr"].max() <= 1:
        jailbreak_avg["asr"] = jailbreak_avg["asr"] * 100

    # 获取不同的越狱类型
    jailbreak_types = df['subject'].unique().tolist()

    # # 计算每种越狱类型的防御效果差异，用于排序
    # jailbreak_diffs = {}
    # for jb_type in jailbreak_types:
    #     with_def = jailbreak_avg[(jailbreak_avg['subject'] == jb_type) & (~jailbreak_avg['not_has_defense'])]
    #     without_def = jailbreak_avg[(jailbreak_avg['subject'] == jb_type) & (jailbreak_avg['not_has_defense'])]
    #
    #     with_val = with_def["asr"].values[0] if not with_def.empty else 0
    #     without_val = without_def["asr"].values[0] if not without_def.empty else 0
    #
    #     jailbreak_diffs[jb_type] = without_val - with_val  # 正值表示防御有效

    # 根据防御效果差异排序（最有效的防御排在前面）
    # sorted_jailbreak_types = sorted(jailbreak_types, key=lambda x: jailbreak_diffs[x], reverse=False)
    # shuffle(sorted_jailbreak_types)
    sorted_jailbreak_types = jailbreak_types

    # 提取带防御和不带防御的值
    with_defense_values = []
    without_defense_values = []

    for jb_type in sorted_jailbreak_types:
        # 带防御的模型
        with_def = jailbreak_avg[(jailbreak_avg['subject'] == jb_type) & (~jailbreak_avg['not_has_defense'])]
        if not with_def.empty:
            with_defense_values.append(with_def["asr"].values[0])
        else:
            with_defense_values.append(0)

        # 不带防御的模型
        without_def = jailbreak_avg[(jailbreak_avg['subject'] == jb_type) & (jailbreak_avg['not_has_defense'])]
        if not without_def.empty:
            without_defense_values.append(without_def["asr"].values[0])
        else:
            without_defense_values.append(0)

    # 处理jailbreak_type标签，优化显示效果
    formatted_labels = []
    for label in sorted_jailbreak_types:
        # 替换空格为换行符
        # formatted = label.replace(' ', '\n')
        # 替换斜杠为斜杠+换行符
        # formatted = formatted.replace('/', '/\n')
        formatted_labels.append(label)

    # 创建雷达图
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, polar=True)

    # 设置角度和绘图方向
    N = len(sorted_jailbreak_types)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图

    # 绘制网格和标签
    plt.xticks(angles[:-1], formatted_labels)  # , size=10

    # 自适应设置y轴刻度
    max_value = 1.25 * max(max(with_defense_values), max(without_defense_values))
    # 确保最大值有一点余量
    max_value = min(100, max_value * 1.1)  # 不超过100%

    # 根据最大值选择合适的刻度间隔
    if max_value <= 20:
        yticks = np.arange(0, max_value + 5, 5)
    elif max_value <= 50:
        yticks = np.arange(0, max_value + 10, 10)
    elif max_value <= 100:
        yticks = np.arange(0, max_value + 20, 20)
    else:
        yticks = np.arange(0, max_value + 25, 25)

    ylabels = [f"{int(y)}%" for y in yticks]
    # ax.set_rlabel_position(0)
    plt.yticks(yticks, ylabels, color="grey", size=10)
    plt.ylim(0, max_value)

    # 绘制数据
    # 带防御的数据
    values = with_defense_values
    values += values[:1]  # 闭合雷达图
    ax.plot(angles, values, 'o-', linewidth=2, color="#4575b4", label="w/ Defense")
    ax.fill(angles, values, color="#4575b4", alpha=0.25)

    # 不带防御的数据
    values = without_defense_values
    values += values[:1]  # 闭合雷达图
    ax.plot(angles, values, 'o-', linewidth=2, color="#d73027", label="w/o Defense")
    ax.fill(angles, values, color="#d73027", alpha=0.25)

    # 添加数值标签
    for i, value in enumerate(with_defense_values):
        angle = angles[i]
        offset = 10  # 标签偏移量
        ax.text(angle, value + offset, f'{value:.1f}%',
                color="#4575b4", ha='center', va='center', fontsize=8,
                # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )

    for i, value in enumerate(without_defense_values):
        angle = angles[i]
        offset = 12  # 标签偏移量
        ax.text(angle, value + offset, f'{value:.1f}%',
                color="#d73027", ha='center', va='center', fontsize=8,
                # bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
                )

    # 添加图例和标题
    # plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    # plt.title(f"ASR (%) by Jailbreak Type with/without Defense - {"asr"}", size=15, y=1.1)

    # 美化网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


def create_bar_chart_attack(df, savepath):
    df = df[df["defense_method"] == "NoneDefender"]
    df = df.groupby(["model_name", "attack_method"])["asr"].mean().reset_index()
    df['attack_method'] = df['attack_method'].str.removeprefix("TransferAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("GPTFuzzAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("RandomSearchAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("AutoDanAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("ReNeLLMAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("PairAttacker_")
    df['attack_method'] = df['attack_method'].replace({'Goal': 'Baseline'})

    g = sns.catplot(
        data=df,
        x="attack_method",
        y="asr",
        hue="model_name",
        palette=sns.color_palette('RdYlGn', n_colors=3),
        kind="bar",
        # estimator=np.mean,
        errorbar=None,
        # hue_order=[False, True],
        # legend_out=False,
        height=3,
        aspect=2
    )

    ax = g.axes[0, 0]
    ax.set_ylabel('')
    ax.set_xlabel('')

    ax.set_ylim(0, 110)

    for i, p in enumerate(ax.patches):
        height = p.get_height()

        if height == 0.:
            continue

        ax.text(
            p.get_x() + p.get_width() / 2,  # 水平位置（居中于柱子）
            height + 1.0,  # 垂直位置（略高于柱子）
            f'{height:.1f}%',  # 显示ASR值，保留一位小数
            ha='center',  # 水平对齐方式
            va='bottom',  # 垂直对齐方式
            rotation=90,
            fontsize=7,  # 字体大小
            color='black'  # 文字颜色
        )

    g.legend.set_title(None)
    # for text in g.legend.texts:
    #     text.set_fontsize(6)

    plt.xticks(rotation=20, ha='center')

    plt.savefig(savepath, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()


def create_bar_chart_defense(df, savepath):
    df = df[df["attack_method"] == "TransferAttacker_Goal"]
    df = df.groupby(["model_name", "defense_method"])["asr"].mean().reset_index()
    df['defense_method'] = df['defense_method'].str.removesuffix("Defender")
    df['defense_method'] = df['defense_method'].replace({'None': 'Baseline'})

    g = sns.catplot(
        data=df,
        x="defense_method",
        y="asr",
        hue="model_name",
        palette=sns.color_palette('Set1', n_colors=3),
        kind="bar",
        # estimator=np.mean,
        errorbar=None,
        # hue_order=[False, True],
        legend_out=False,
        order=["SelfDefense", "GoalPriority", "Icl", "Paraphrase", "PerplexityFilter", "RPO", "Baseline", ],
        height=3,
        aspect=2
    )

    ax = g.axes[0, 0]
    ax.set_ylabel('')
    ax.set_xlabel('')

    ax.set_ylim(0, 110)

    for i, p in enumerate(ax.patches):
        height = p.get_height()

        if height == 0.:
            continue

        ax.text(
            p.get_x() + p.get_width() / 2,  # 水平位置（居中于柱子）
            height + 1.0,  # 垂直位置（略高于柱子）
            f'{height:.1f}%',  # 显示ASR值，保留一位小数
            ha='center',  # 水平对齐方式
            va='bottom',  # 垂直对齐方式
            rotation=90,
            fontsize=7,  # 字体大小
            color='black'  # 文字颜色
        )

    g.legend.set_title(None)
    # for text in g.legend.texts:
    #     text.set_fontsize(6)

    plt.xticks(rotation=0, ha='center')

    plt.savefig(savepath, bbox_inches='tight', transparent=True)
    plt.show()
    plt.close()


def create_radar_chart_attack(df, savepath):
    df = df[df["defense_method"] == "NoneDefender"]
    # df = df.groupby(["model_name", "attack_method"])["asr"].mean().reset_index()
    df['attack_method'] = df['attack_method'].str.removeprefix("TransferAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("GPTFuzzAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("RandomSearchAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("AutoDanAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("ReNeLLMAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("PairAttacker_")
    df['attack_method'] = df['attack_method'].replace({'Goal': 'Baseline'})

    subject = df["subject"].unique().tolist()
    # shuffle(subject)
    df = df.groupby(["attack_method", "subject"])["asr"].mean().reset_index()

    # 创建雷达图
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, polar=True)

    # 调色盘
    attack_methods = df["attack_method"].unique().tolist()
    palette = sns.color_palette("Set2", n_colors=len(attack_methods))

    # 设置角度和绘图方向
    N = len(subject)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图

    # 绘制网格和标签
    plt.xticks(angles[:-1], subject)  # , size=10

    # 自适应设置y轴刻度
    max_value = 1.25 * df["asr"].max()
    # 确保最大值有一点余量
    max_value = min(100, max_value * 1.1)  # 不超过100%

    # 根据最大值选择合适的刻度间隔
    if max_value <= 20:
        yticks = np.arange(0, max_value + 5, 5)
    elif max_value <= 50:
        yticks = np.arange(0, max_value + 10, 10)
    elif max_value <= 100:
        yticks = np.arange(0, max_value + 20, 20)
    else:
        yticks = np.arange(0, max_value + 25, 25)

    ylabels = [f"{int(y)}%" for y in yticks]

    plt.yticks(yticks, ylabels, color="grey", size=10)
    plt.ylim(0, max_value)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subject, fontsize=10, ha='center')
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        label.set_y(label.get_position()[1] - 0.1)  # 向外或向内移动

    for i, attack_method in enumerate(attack_methods):
        df_model = df[df["attack_method"] == attack_method]
        # 绘制数据
        values = df_model["asr"].tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color=palette[i], label=attack_method)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1.0))
    # plt.title(f"ASR (%) by Jailbreak Type with/without Defense - {"asr"}", size=15, y=1.1)

    # 美化网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


def create_radar_chart_defense(df, savepath):
    df = df[df["attack_method"] == "TransferAttacker_Goal"]
    # df = df.groupby(["model_name", "defense_method"])["asr"].mean().reset_index()
    df['defense_method'] = df['defense_method'].str.removesuffix("Defender")
    df['defense_method'] = df['defense_method'].replace({'None': 'Baseline'})

    subject = df["subject"].unique().tolist()
    # shuffle(subject)
    df = df.groupby(["defense_method", "subject"])["asr"].mean().reset_index()

    # 创建雷达图
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, polar=True)

    # 调色盘
    defense_methods = df["defense_method"].unique().tolist()
    palette = sns.color_palette("Set2", n_colors=len(defense_methods))

    # 设置角度和绘图方向
    N = len(subject)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图

    # 绘制网格和标签
    plt.xticks(angles[:-1], subject)  # , size=10

    # 自适应设置y轴刻度
    max_value = 1.25 * df["asr"].max()
    # 确保最大值有一点余量
    max_value = min(100, max_value * 1.1)  # 不超过100%

    # 根据最大值选择合适的刻度间隔
    if max_value <= 20:
        yticks = np.arange(0, max_value + 5, 5)
    elif max_value <= 50:
        yticks = np.arange(0, max_value + 10, 10)
    elif max_value <= 100:
        yticks = np.arange(0, max_value + 20, 20)
    else:
        yticks = np.arange(0, max_value + 25, 25)

    ylabels = [f"{int(y)}%" for y in yticks]

    plt.yticks(yticks, ylabels, color="grey", size=10)
    plt.ylim(0, max_value)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subject, fontsize=10, ha='center')
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        label.set_y(label.get_position()[1] - 0.1)  # 向外或向内移动

    for i, defense_method in enumerate(defense_methods):
        df_model = df[df["defense_method"] == defense_method]
        # 绘制数据
        values = df_model["asr"].tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color=palette[i], label=defense_method)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1.0))
    # plt.title(f"ASR (%) by Jailbreak Type with/without Defense - {"asr"}", size=15, y=1.1)

    # 美化网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


def create_radar_chart_attack_model(df, savepath):
    df = df[df["defense_method"] == "NoneDefender"]  # 去掉所有不带防御的
    df = df[df["attack_method"] != "TransferAttacker_Goal"]  # 去掉所有不带攻击的
    # df = df.groupby(["model_name", "attack_method"])["asr"].mean().reset_index()
    df['attack_method'] = df['attack_method'].str.removeprefix("TransferAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("GPTFuzzAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("RandomSearchAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("AutoDanAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("ReNeLLMAttacker_")
    df['attack_method'] = df['attack_method'].str.removeprefix("PairAttacker_")
    df['attack_method'] = df['attack_method'].replace({'Goal': 'Baseline'})

    subject = df["subject"].unique().tolist()
    # shuffle(subject)
    df = df.groupby(["model_name", "subject"])["asr"].mean().reset_index()

    # 创建雷达图
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(111, polar=True)

    # 调色盘
    models = df["model_name"].unique().tolist()
    palette = sns.color_palette("Set2", n_colors=len(models))

    # 设置角度和绘图方向
    N = len(subject)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图

    # 绘制网格和标签
    plt.xticks(angles[:-1], subject)  # , size=10

    # 自适应设置y轴刻度
    max_value = 1.25 * df["asr"].max()
    # 确保最大值有一点余量
    max_value = min(100, max_value * 1.1)  # 不超过100%

    # 根据最大值选择合适的刻度间隔
    if max_value <= 20:
        yticks = np.arange(0, max_value + 5, 5)
    elif max_value <= 50:
        yticks = np.arange(0, max_value + 10, 10)
    elif max_value <= 100:
        yticks = np.arange(0, max_value + 20, 20)
    else:
        yticks = np.arange(0, max_value + 25, 25)

    ylabels = [f"{int(y)}%" for y in yticks]

    plt.yticks(yticks, ylabels, color="grey", size=10)
    plt.ylim(0, max_value)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(subject, fontsize=10, ha='center')
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
        label.set_verticalalignment('center')
        label.set_y(label.get_position()[1] - 0.1)  # 向外或向内移动

    for i, model in enumerate(models):
        df_model = df[df["model_name"] == model]
        # 绘制数据
        values = df_model["asr"].tolist()
        values += values[:1]  # 闭合雷达图
        ax.plot(angles, values, 'o-', linewidth=2, color=palette[i], label=model)

    # 添加图例和标题
    plt.legend(loc='upper right', bbox_to_anchor=(1.7, 1.0))
    # plt.title(f"ASR (%) by Jailbreak Type with/without Defense - {"asr"}", size=15, y=1.1)

    # 美化网格
    ax.grid(True, linestyle='--', alpha=0.7)

    # 保存图像
    plt.savefig(savepath, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_settings()
    
    df_attacker = pd.read_csv("asr_attacker_table.csv")
    df_no_attacker = pd.read_csv("asr_no_attacker_table.csv")
    
    subjects = df_attacker.columns[1:].tolist()
    
    df_attacker_long = df_attacker.melt(id_vars=['model_name'], value_vars=subjects, 
                                         var_name='subject', value_name='asr')
    df_no_attacker_long = df_no_attacker.melt(id_vars=['model_name'], value_vars=subjects,
                                                var_name='subject', value_name='asr')
    
    create_radar_chart(df_attacker_long, "asr_attacker_radar.png")
    create_radar_chart(df_no_attacker_long, "asr_no_attacker_radar.png")
