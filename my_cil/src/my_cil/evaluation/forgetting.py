def calculate_forgetting_rate(stages_stats):
    """
    计算每个阶段的平均遗忘率（正值表示遗忘，负值表示性能提升）。

    参数:
        stages_stats (list of dict): 每个阶段的统计信息，格式为：
            [
                {0: {"n_correct": 90, "n_samples": 100}},  # 阶段0
                {0: {"n_correct": 85, "n_samples": 100},   # 阶段1
                 1: {"n_correct": 190, "n_samples": 200}},
                # 后续阶段...
            ]

    返回:
        list: 每个阶段的平均遗忘率（从阶段1开始）。
    """
    forgetting_rates = []

    # 遍历阶段1到最后一个阶段
    for t in range(1, len(stages_stats)):
        current_stats = stages_stats[t]  # 当前阶段统计
        prev_stats = stages_stats[t - 1]  # 上一阶段统计

        # 提取旧类列表（上一阶段存在的所有类别）
        old_classes = list(prev_stats.keys())
        N = len(old_classes)
        print(N)
        if N == 0:
            forgetting_rates.append(0.0)
            continue

        sum_forget = 0.0
        for cls in old_classes:
            # 上一阶段的类别准确率
            prev_correct = prev_stats[cls]["n_correct"]
            prev_total = prev_stats[cls]["n_samples"]
            prev_acc = prev_correct / prev_total if prev_total > 0 else 0.0

            # 当前阶段的类别准确率（若未测试该类，准确率视为0）
            if cls in current_stats:
                current_correct = current_stats[cls]["n_correct"]
                current_total = current_stats[cls]["n_samples"]
                current_acc = current_correct / current_total if current_total > 0 else 0.0
            else:
                current_acc = 0.0

            # 遗忘量 = 上一阶段准确率 - 当前阶段准确率
            forget = prev_acc - current_acc
            sum_forget += forget

        # 平均遗忘率（正值表示整体发生遗忘）
        avg_forget = sum_forget / N
        forgetting_rates.append(avg_forget)

    return forgetting_rates


def convert_to_stage_stats(class_correct_list, class_total_list):
    """
    将多个阶段的class_correct和class_total转换为stage_stats格式。

    参数:
        class_correct_list (list of dict): 每个元素是字典，表示某阶段的各类别正确数，例如 [{0: 90}, {0: 85, 1: 190}, ...]
        class_total_list (list of dict): 每个元素是字典，表示某阶段的各类别总样本数，例如 [{0: 100}, {0: 100, 1: 200}, ...]

    返回:
        list of dict: stage_stats格式，例如：
            [
                {0: {"n_correct": 90, "n_samples": 100}},
                {0: {"n_correct": 85, "n_samples": 100}, 1: {"n_correct": 190, "n_samples": 200}},
                ...
            ]
    """
    stage_stats = []
    # 遍历每个阶段
    for correct_dict, total_dict in zip(class_correct_list, class_total_list):
        stage = {}
        # 遍历当前阶段的所有类别
        for cls in total_dict:
            # 获取当前类别的正确数和总样本数
            n_correct = correct_dict.get(cls, 0)  # 若未记录正确数，默认0
            n_samples = total_dict[cls]
            stage[cls] = {"n_correct": n_correct, "n_samples": n_samples}
        stage_stats.append(stage)
    return stage_stats

if __name__ == "__main__":
    # 阶段0
    class_correct_0={0: 1239, 1: 1616, 2: 559}
    class_total_0={0: 1530, 1: 1645, 2: 575}

    # 阶段1
    class_correct_1={0: 527, 1: 226, 2: 315, 3: 26471, 4: 3663}
    class_total_1={0: 1530, 1: 1645, 2: 575, 3: 26857, 4: 3851}

    # 阶段2
    class_correct_2={0: 138, 1: 142, 2: 416, 3: 15237, 4: 2373, 5: 27223, 6: 11821}
    class_total_2={0: 1530, 1: 1645, 2: 575, 3: 26857, 4: 3851, 5: 30329, 6: 13507}



    class_correct_list = [class_correct_0, class_correct_1, class_correct_2]
    class_total_list = [class_total_0, class_total_1, class_total_2]

    # 转换为stage_stats格式
    stage_stats = convert_to_stage_stats(class_correct_list, class_total_list)

    print(stage_stats)

    forgetting_rates = calculate_forgetting_rate(stage_stats)
    print("各阶段平均遗忘率:", [round(fr, 4) for fr in forgetting_rates])
