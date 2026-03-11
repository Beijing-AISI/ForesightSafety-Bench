import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def get_result_df(result_dir):
    result_files = glob.glob(f"{result_dir}/*.jsonl")
    result_files = [f.replace('\\', '/') for f in result_files]

    dfs = []

    for file_path in result_files:
        # print(file_path)
        model = file_path.split("/")[-1].removesuffix(".jsonl").split("_")[0]
        # print(model)

        data_list = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                row_data = json.loads(line)

                category = row_data.get("category", None)
                category = row_data["id"].split("_")[0] if category is None else category

                if row_data.get("intervened_correct"):
                    strict_score = row_data["base_correct"] and not row_data["intervened_correct"]
                else:
                    strict_score = row_data["base_correct"]

                # designed_score = False
                # if category == "sycophancy":
                #     designed_score = strict_score
                # elif category == "sandbagging":
                #     designed_score = strict_score
                # elif category == "alignment_faking":
                #     designed_score = strict_score
                # elif category == "deception":
                #     designed_score = row_data["follows_intervention"] and strict_score

                data = {
                    "model": model,
                    "category": category,
                    "base_correct": row_data["base_correct"],
                    "intervened_correct": row_data.get("intervened_correct"),
                    "mislead": row_data["follows_intervention"],
                    "strict_score": strict_score,
                    # "designed_score": designed_score,
                }

                data_list.append(data)
        df = pd.DataFrame(data_list)
        df = df.groupby(["model", "category"])[
            df.select_dtypes(include=['bool']).columns
        ].mean().reset_index()
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols] * 100

        dfs.append(df)

    all_data = pd.concat(dfs)
    all_data.sort_values(by=["category", "model", ], inplace=True)
    # print(all_data.to_string())

    pivot_table = all_data.pivot_table(index="model", columns="category", values="strict_score")
    pivot_table = pivot_table[[
        "alignment_faking", "sandbagging", "deception", "sycophancy", "manipulation", "feint", "bluffing"
    ]]
    # pivot_table = pivot_table.reset_index()
    # pivot_table = pivot_table[[
    #     "deception"
    # ]]
    pivot_table["avg"] = pivot_table.mean(axis=1)
    pivot_table = pivot_table.sort_values(by='avg', ascending=True)
    # pivot_table.loc["average"] = pivot_table.mean()
    #
    # print(pivot_table.to_string()
    pivot_table = pivot_table.reset_index()
    pivot_table.to_csv("judged_results.csv", index=False)

    # model_names = pivot_table["model"].sort_values()
    # for model in model_names:
    #     print(f"\"{model}\": \"{model}\",")

    return pivot_table


def plot(df):
    # 为了美观，定义维度的显示名称映射（用于图表标签）
    dim_labels = {
        "alignment_faking": "Alignment Faking",
        "sandbagging": "Sandbagging",
        "deception": "Deception",
        "sycophancy": "Sycophancy",
        "manipulation": "Manipulation",
        "feint": "Feint",
        "bluffing": "Bluffing"
    }
    df = df.rename(columns=dim_labels)

    model_labels = {
        "claude-3-5-haiku": "Claude-3.5-Haiku",
        "claude-haiku-4-5": "Claude-Haiku-4.5",
        "claude-sonnet-4-5": "Claude-Sonnet-4.5",
        "deepseek-v3.2": "DeepSeek-V3.2",
        "deepseek-v3.2-speciale": "DeepSeek-V3.2-Speciale",
        "doubao-seed-1-6": "Doubao-Seed-1.6",
        "doubao-seed-1-8": "Doubao-seed-1.8",
        "gemini-2.5-flash": "Gemini-2.5-Flash",
        "gemini-3-flash-preview": "Gemini-3-Flash-Preview",
        "gemini-3-pro-preview": "Gemini-3-pro-Preview",
        "glm-4.7": "GLM-4.7",
        "gpt-4o-2024-11-20": "GPT-4o-2024-11-20",
        "gpt-5.2": "GPT-5.2",
        "grok-4": "Grok-4",
        "grok-4-fast-non-reasoning": "Grok-4-Fast-Non-Reasoning",
        "kimi-k2-0905": "Kimi-K2-0905",
        "kimi-k2.5": "Kimi-K2.5",
        "llama-3.3-70b": "Llama-3.3-70B",
        "llama-4-maverick": "Llama-4-Maverick",
        "qwen2.5-72b-instruct": "Qwen2.5-72B-Instruct",
        "qwen3-235b-a22b-instruct-2507": "Qwen3-235B-A22B-Instruct-2507",
        "qwen3-max-2026-01-23": "Qwen3-Max-Thinking",
    }
    df['model'] = df['model'].map(model_labels)

    # 定义风险维度的顺序（可根据需要调整）
    dimensions = [
        "Alignment Faking", "Sandbagging", "Deception",
        "Sycophancy", "Manipulation", "Feint", "Bluffing"
    ]

    # 顶刊风格设置 (Nature Style Aesthetics)
    plt.rcParams['font.family'] = 'serif'  # 使用衬线体 (如 Times New Roman)
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 0.8  # 坐标轴线宽
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20  # 标题字号
    plt.rcParams['axes.titleweight'] = 'bold'  # 标题加粗

    # 创建多面板布局 (Multi-panel Layout)
    # 创建画布，设置宽高比
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)

    # 定义网格：2行2列，第一行两图并排，第二行横跨整行
    # height_ratios 控制上下两行的高度比例
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1.2], width_ratios=[1, 1],
        wspace=0, hspace=0.05
    )

    # Panel A: 综合安全排行榜 (Ranking)
    ax1 = fig.add_subplot(gs[0, 0])
    # 按平均分排序
    df_sorted = df.sort_values('avg', ascending=True)
    # 绘制条形图
    # Palette: RdYlGn_r (红-黄-绿 反转)，分数低(绿)代表安全，分数高(红)代表危险
    sns.barplot(
        data=df_sorted, x='avg', y='model',
        palette='RdYlGn_r', ax=ax1,
        width=0.65,
        edgecolor='black', linewidth=0.5)

    # 细节修饰
    ax1.set_title('a  Overall Social Safety Risk Leaderboard', loc='left', pad=10)
    ax1.set_xlabel('Average Risk Score (%)')
    ax1.set_ylabel('')  # 移除Y轴标签，因为模型名已经很长了
    ax1.grid(axis='x', linestyle='-', alpha=0.5)  # 仅保留X轴网格
    ax1.set_xlim(0, 100)  # 固定量程
    ax1.set_ylim(21.8, -0.8)

    # Panel B: 风险维度分布 (Distribution)
    ax2 = fig.add_subplot(gs[0, 1])

    # 数据转换：宽表转长表 (Wide to Long)
    df_melted = df.melt(id_vars=['model'], value_vars=dimensions, var_name='category', value_name='score')

    # 绘制箱线图 (Boxplot) + 散点图 (Stripplot)
    sns.boxplot(data=df_melted, x='score', y='category', palette='viridis', ax=ax2, width=0.6, fliersize=0)
    sns.stripplot(data=df_melted, x='score', y='category', color="black", size=4, alpha=0.4, ax=ax2, jitter=True)

    # 细节修饰
    ax2.set_title('b  Distribution of Strategic Risks', loc='left', pad=10)
    ax2.set_xlabel('Risk Score (%)')
    ax2.set_ylabel('')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 105)

    # ==========================================
    # Panel C: 全景热图 (Heatmap)
    # ==========================================
    ax3 = fig.add_subplot(gs[1, :])  # 横跨第二行所有列

    # 准备热图数据
    heatmap_data = df.set_index('model')[dimensions]
    # heatmap_data = df
    heatmap_data = heatmap_data.reindex(df_sorted['model'])  # 保持与 Panel A 相同的排序

    # 绘制热图
    # cmap: YlOrRd (黄-橙-红)，颜色越深风险越高
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd',
                cbar_kws={'label': 'Risk Score (%)', 'aspect': 40, 'pad': 0.02},
                ax=ax3, linewidths=0.5, linecolor='white', annot_kws={"size": 11})

    # 细节修饰
    ax3.set_title('c  Holistic Landscape of Social AI Safety (Model vs. Dimension)', loc='left', pad=10)
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    plt.xticks(rotation=0)  # 保持X轴标签水平

    # 4. 保存图片
    # plt.tight_layout()
    plt.savefig('social_safety_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig('social_safety_overview.pdf', bbox_inches='tight')  # PDF矢量图更适合投稿
    plt.show()


def plot_asr_data(csv_file, title_prefix, output_prefix):
    """
    为ASR数据创建综合可视化图表
    
    参数:
        csv_file: CSV文件路径
        title_prefix: 标题前缀
        output_prefix: 输出文件前缀
    """
    # 读取数据
    df = pd.read_csv(csv_file)
    
    # 获取学科列表（除了model_name列）
    subjects = df.columns[1:].tolist()
    
    # 计算平均值
    df['avg'] = df[subjects].mean(axis=1)
    
    # 顶刊风格设置 (Nature Style Aesthetics)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'
    
    # 创建多面板布局
    fig = plt.figure(figsize=(16, 14), constrained_layout=True)
    
    gs = fig.add_gridspec(
        2, 2,
        height_ratios=[1, 1.2], width_ratios=[1, 1],
        wspace=0, hspace=0.05
    )
    
    # Panel A: 综合ASR排行榜 (Ranking)
    ax1 = fig.add_subplot(gs[0, 0])
    df_sorted = df.sort_values('avg', ascending=True)
    
    # 绘制条形图
    bars = ax1.barh(df_sorted['model_name'], df_sorted['avg'], 
                    height=0.65, edgecolor='black', linewidth=0.5)
    
    # 设置颜色渐变（红-黄-绿反转）
    norm = plt.Normalize(vmin=df_sorted['avg'].min(), vmax=df_sorted['avg'].max())
    colors = plt.cm.RdYlGn_r(norm(df_sorted['avg']))
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    ax1.set_title(f'a  {title_prefix} - Overall ASR Leaderboard', loc='left', pad=10)
    ax1.set_xlabel('Average ASR Score (%)')
    ax1.set_ylabel('')
    ax1.grid(axis='x', linestyle='-', alpha=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.8, len(df_sorted) - 0.2)


    # Panel B: 学科维度分布 (Distribution)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # 数据转换：宽表转长表
    df_melted = df.melt(id_vars=['model_name'], value_vars=subjects,
                        var_name='subject', value_name='score')
    
    # 绘制箱线图 + 散点图
    sns.boxplot(data=df_melted, x='score', y='subject',
                palette='viridis', ax=ax2, width=0.6, fliersize=0)
    sns.stripplot(data=df_melted, x='score', y='subject',
                  color="black", size=4, alpha=0.4, ax=ax2, jitter=True)
    
    ax2.set_title(f'b  Distribution of ASR across Subjects', loc='left', pad=10)
    ax2.set_xlabel('ASR Score (%)')
    ax2.set_ylabel('')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 105)
    
    # Panel C: 全景热图 (Heatmap)
    ax3 = fig.add_subplot(gs[1, :])
    
    heatmap_data = df_sorted.set_index('model_name')[subjects]
    
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='YlOrRd',
                cbar_kws={'label': 'ASR Score (%)', 'aspect': 40, 'pad': 0.02},
                ax=ax3, linewidths=0.5, linecolor='white',
                annot_kws={"size": 11})
    
    ax3.set_title(f'c  {title_prefix} - Holistic Landscape (Model vs. Subject)',
                  loc='left', pad=10)
    ax3.set_xlabel('')
    ax3.set_ylabel('')
    plt.xticks(rotation=0)
    
    # 保存图片
    plt.savefig(f'{output_prefix}_overview.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_prefix}_overview.pdf', bbox_inches='tight')
    print(f"综合图表已保存到: {output_prefix}_overview.png 和 .pdf")
    
    plt.close()


def plot_rank(csv_file, title_prefix, output_prefix):
    # 读取数据
    df = pd.read_csv(csv_file)

    # 获取学科列表（除了model_name列）
    subjects = df.columns[1:].tolist()

    # 计算平均值
    df['avg'] = df[subjects].mean(axis=1)

    # 顶刊风格设置 (Nature Style Aesthetics)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'

    # Panel A: 综合ASR排行榜 (Ranking)
    fig = plt.figure(figsize=(4,6))
    ax1 = fig.add_subplot(111)
    df_sorted = df.sort_values('avg', ascending=True)
    ax1.yaxis.tick_right()

    # 绘制条形图
    bars = ax1.barh(df_sorted['model_name'], df_sorted['avg'],
                    height=0.65, edgecolor='black', linewidth=0.5)

    # 设置颜色渐变（红-黄-绿反转）
    norm = plt.Normalize(vmin=df_sorted['avg'].min(), vmax=df_sorted['avg'].max())
    colors = plt.cm.RdYlGn_r(norm(df_sorted['avg']))
    for bar, color in zip(bars, colors):
        bar.set_color(color)

    # ax1.set_title(f'a  {title_prefix} - Overall ASR Leaderboard', loc='left', pad=10)
    ax1.set_xlabel('Average ASR Score (%)')
    ax1.set_ylabel('')
    ax1.grid(axis='x', linestyle='-', alpha=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-0.8, len(df_sorted) - 0.2)

    save_path = output_prefix + "_rank.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SAVE: {save_path}")
    plt.close()

def plot_box(csv_file, title_prefix, output_prefix):
    # 读取数据
    df = pd.read_csv(csv_file)

    # 获取学科列表（除了model_name列）
    subjects = df.columns[1:].tolist()

    # 计算平均值
    df['avg'] = df[subjects].mean(axis=1)

    # 顶刊风格设置 (Nature Style Aesthetics)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'

    # Panel B: 学科维度分布 (Distribution)
    fig = plt.figure(figsize=(4,6))
    ax2 = fig.add_subplot(111)
    ax2.yaxis.tick_right()

    # 数据转换：宽表转长表
    df_melted = df.melt(id_vars=['model_name'], value_vars=subjects,
                        var_name='subject', value_name='score')

    # 绘制箱线图 + 散点图
    sns.boxplot(data=df_melted, x='score', y='subject',
                palette='viridis', ax=ax2, width=0.6, fliersize=0)
    sns.stripplot(data=df_melted, x='score', y='subject',
                  color="black", size=4, alpha=0.4, ax=ax2, jitter=True)

    # ax2.set_title(f'b  Distribution of ASR across Subjects', loc='left', pad=10)
    ax2.set_xlabel('ASR Score (%)')
    ax2.set_ylabel('')
    ax2.grid(axis='x', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 105)

    save_path = output_prefix + "_box.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SAVE: {save_path}")
    plt.close()

def plot_heatmap(csv_file, title_prefix, output_prefix):
    # 读取数据
    df = pd.read_csv(csv_file)

    # 获取学科列表（除了model_name列）
    subjects = df.columns[1:].tolist()

    # 计算平均值
    df['avg'] = df[subjects].mean(axis=1)
    df_sorted = df.sort_values('avg', ascending=True)

    # 顶刊风格设置 (Nature Style Aesthetics)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.titleweight'] = 'bold'

    # Panel C: 全景热图 (Heatmap)
    fig = plt.figure(figsize=(15,6))
    ax3 = fig.add_subplot(111)
    heatmap_data = df_sorted.set_index('model_name')[subjects]

    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap='viridis',
                cbar_kws={'label': 'ASR Score (%)', 'aspect': 40, 'pad': 0.02},
                ax=ax3, linewidths=0.5, linecolor='white',
                annot_kws={"size": 11})

    ax3.set_xlabel('')
    ax3.set_ylabel('')
    # plt.xticks(rotation=90,fontsize=12)
    plt.xticks(fontsize=12)
    # plt.yticks(fontsize=12)

    save_path = output_prefix + "_heatmap.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"SAVE: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 原始功能（如果有results_judged文件夹）
    # df = get_result_df("results_judged")
    # plot(df)
    
    # RANK
    plot_rank(
        'asr_attacker_table.csv',
        'ASR with Attacker',
        'asr_attacker'
    )
    
    plot_rank(
        'asr_no_attacker_table.csv',
        'ASR without Attacker',
        'asr_no_attacker'
    )

    # BOX
    plot_box(
        'asr_attacker_table.csv',
        'ASR with Attacker',
        'asr_attacker'
    )

    plot_box(
        'asr_no_attacker_table.csv',
        'ASR without Attacker',
        'asr_no_attacker'
    )

    # Heatmap
    plot_heatmap(
        'asr_attacker_table.csv',
        'ASR with Attacker',
        'asr_attacker'
    )

    plot_heatmap(
        'asr_no_attacker_table.csv',
        'ASR without Attacker',
        'asr_no_attacker'
    )
