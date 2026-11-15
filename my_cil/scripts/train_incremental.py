import torch
from torch.autograd import Variable
import numpy as np
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from my_cil.data import HsiLiDARData

from my_cil.training import MyCilTrainNet


from my_cil.evaluation import calculate_forgetting_rate, convert_to_stage_stats

# 参数设置
CUDA_VISIBLE_DEVICES=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.set_device(0)

import random
random.seed(11)
# 加载数据

num_classes = 3
# 运行

if __name__ == '__main__':
    MyModel = MyCilTrainNet(num_group_tokens=3, group_feature_size=128, feature_size=128, num_classes=0)


    test_acc = []
    K = 30
    class_correct_list = []
    class_total_list = []

    train_set_base = HsiLiDARData(root_train_data='./data_houston/train_patches.npy',
                                  root_train_label='./data_houston/train_labels.npy',
                                  root_test_data='./data_houston/test_patches.npy',
                                  root_test_label='./data_houston/test_labels.npy',
                                  train=True,
                                  classes=range(7),)
    train_loader_base = torch.utils.data.DataLoader(train_set_base, batch_size=16, shuffle=True, num_workers=4)

    test_set_base = HsiLiDARData(root_train_data='./data_houston/train_patches.npy',
                                 root_train_label='./data_houston/train_labels.npy',
                                 root_test_data='./data_houston/test_patches.npy',
                                 root_test_label='./data_houston/test_labels.npy',
                                 train=False,
                                 classes=range(7),
                                 )
    test_loader_base = torch.utils.data.DataLoader(test_set_base, batch_size=16, shuffle=False, num_workers=4)

    MyModel.update_representation(train_set_base, 30)
    MyModel.eval()
    A = 6
    print("m: ", A)

    MyModel.reduce_exemplar_set(A)

    for y in range(MyModel.n_known, MyModel.n_classes):
        print("Constructing exemplar set for class-%d..." % y)
        images = train_set_base.get_image_class(y)
        MyModel.construct_exemplar_set(images, A)
        print('Done')

    for y, P_y in enumerate(MyModel.exemplar_sets):
        print("Exemplar set for class-%d..." % y, P_y.shape)

    MyModel.n_known = MyModel.n_classes
    print("Number of known_classes: ", MyModel.n_known)

    total = 0.0
    correct = 0.0
    for index, images, labels in train_loader_base:
        images = Variable(images).to(device)
        images_1 = images[:, 0:144, :, :]
        images_2 = images[:, 144:, :, :]
        preds = MyModel.classify(images_1, images_2)
        preds = preds.tolist()

        total += labels.size(0)
        correct += (preds == labels.numpy()).sum()
    print('Train Accuracy : %.2f ,' % (100.0 * correct / total))

    total = 0.0
    correct = 0.0
    class_total = {}
    class_correct = {}
    all_preds = []
    all_labels = []
    for index, images, labels in test_loader_base:
        images = Variable(images).to(device)
        images_1 = images[:, 0:144, :, :]
        images_2 = images[:, 144:, :, :]
        preds = MyModel.classify(images_1, images_2)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        preds = preds.tolist()

        total += labels.size(0)
        correct += (preds == labels.numpy()).sum()

        # 统计每个类别的正确数和总数
        for label, pred in zip(labels, preds):
            cls = label.item()
            class_total[cls] = class_total.get(cls, 0) + 1
            if label == pred:
                class_correct[cls] = class_correct.get(cls, 0) + 1

    print('Test Accuracy : %.2f ,' % (100.0 * correct / total))
    # ----------------------- 计算指标 -----------------------
    # 1. 类平均准确率 (Class-Average Accuracy)
    class_acc = {}
    for cls in class_total:
        if class_total[cls] > 0:
            class_acc[cls] = class_correct.get(cls, 0) / class_total[cls]
    avg_accuracy = np.mean(list(class_acc.values())) if class_acc else 0.0

    # 2. Kappa系数
    confusion = confusion_matrix(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    print("Class-Average Accuracy:", avg_accuracy)
    print("confusion_matrix:", confusion)
    print("Kappa Coefficient:", kappa)
    print("class_correct:", class_correct)
    print("class_total:", class_total)

    class_total_list.append(class_total)
    class_correct_list.append(class_correct)

    torch.cuda.empty_cache()

    # ----------------------- 增量阶段 -----------------------

    for s in range(7, 15, num_classes):
        print("Iteration: ", s)
        print("Loading training examples for classes", range(s, s + num_classes))

        train_set = HsiLiDARData(root_train_data='./data_houston/train_patches.npy',
                                 root_train_label='./data_houston/train_labels.npy',
                                 root_test_data='./data_houston/test_patches.npy',
                                 root_test_label='./data_houston/test_labels.npy',
                                 train=True,
                                 classes=range(s, s + num_classes),
                                 )
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4)

        test_set_old = HsiLiDARData(root_train_data='./data_houston/train_patches.npy',
                                    root_train_label='./data_houston/train_labels.npy',
                                    root_test_data='./data_houston/test_patches.npy',
                                    root_test_label='./data_houston/test_labels.npy',
                                    train=False,
                                    classes=range(s),

                                    )
        test_set_all = HsiLiDARData(root_train_data='./data_houston/train_patches.npy',
                                    root_train_label='./data_houston/train_labels.npy',
                                    root_test_data='./data_houston/test_patches.npy',
                                    root_test_label='./data_houston/test_labels.npy',
                                    train=False,
                                    classes=range(s + num_classes),

                                    )
        test_loader_old = torch.utils.data.DataLoader(test_set_old, batch_size=16, shuffle=False, num_workers=4)
        test_loader_all = torch.utils.data.DataLoader(test_set_all, batch_size=16, shuffle=False, num_workers=4)

        epoch = 35
        MyModel.update_representation(train_set, epoch)
        MyModel.eval()
        m = K//s
        # m = 6

        print("m: ", m)

        MyModel.reduce_exemplar_set(m)

        for y in range(MyModel.n_known, MyModel.n_classes):
            print("Constructing exemplar set for class-%d..." % y)
            images = train_set.get_image_class(y)
            MyModel.construct_exemplar_set(images, m)
            print('Done')

        for y, P_y in enumerate(MyModel.exemplar_sets):
            print("Exemplar set for class-%d..." % y, P_y.shape)

        MyModel.n_known = MyModel.n_classes
        print("Number of known_classes: ", MyModel.n_known)

        total = 0.0
        correct = 0.0
        for index, images, labels in train_loader:
            images = Variable(images).to(device)
            images_1 = images[:, 0:144, :, :]
            images_2 = images[:, 144:, :, :]
            labels = labels.to(device)
            preds, _, _, _, _, _ = MyModel(images_1, images_2)
            _, preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            # preds = MyModel.classify(images_1, images_2)
            # preds = preds.tolist()
            # total += labels.size(0)
            # correct += (preds == labels.numpy()).sum()
        print('Train Accuracy : %.2f ,' % (100.0 * correct / total))

        total = 0.0
        correct = 0.0
        for index, images, labels in test_loader_old:
            images = Variable(images).to(device)
            images_1 = images[:, 0:144, :, :]
            images_2 = images[:, 144:, :, :]
            labels = labels.to(device)
            preds, _, _, _, _, _ = MyModel(images_1, images_2)
            _, preds = torch.max(preds.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            # preds = MyModel.classify(images_1, images_2)
            # preds = preds.tolist()
            # total += labels.size(0)
            # correct += (preds == labels.numpy()).sum()
        print('Test_old Accuracy : %.2f ,' % (100.0 * correct / total))

        total = 0.0
        correct = 0.0
        class_correct = {}
        class_total = {}
        all_preds = []
        all_labels = []
        for index, images, labels in test_loader_all:
            images = Variable(images).to(device)
            images_1 = images[:, 0:144, :, :]
            images_2 = images[:, 144:, :, :]
            labels = labels.to(device)
            preds, _, _, _, _, _ = MyModel(images_1, images_2)
            _, preds = torch.max(preds.data, 1)
            # 收集预测结果和真实标签
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            # preds = MyModel.classify(images_1, images_2)
            # preds = preds.tolist()
            # total += labels.size(0)
            # correct += (preds == labels.numpy()).sum()

            # 统计每个类别的正确数和总数
            for label, pred in zip(labels, preds):
                cls = label.item()
                class_total[cls] = class_total.get(cls, 0) + 1
                if label == pred:
                    class_correct[cls] = class_correct.get(cls, 0) + 1
        print('Test_all Accuracy : %.2f ,' % (100.0 * correct / total))

        # ----------------------- 计算指标 -----------------------
        # 1. 类平均准确率 (Class-Average Accuracy)
        class_acc = {}
        for cls in class_total:
            if class_total[cls] > 0:
                class_acc[cls] = class_correct.get(cls, 0) / class_total[cls]
        avg_accuracy = np.mean(list(class_acc.values())) if class_acc else 0.0

        # 2. Kappa系数
        confusion = confusion_matrix(all_labels, all_preds)
        kappa = cohen_kappa_score(all_labels, all_preds)

        print("Class-Average Accuracy:", avg_accuracy)
        print("Kappa Coefficient:", kappa)
        print("class_correct:", class_correct)
        print("class_total:", class_total)

        class_total_list.append(class_total)
        class_correct_list.append(class_correct)

        # 转换为stage_stats格式
        stage_stats = convert_to_stage_stats(class_correct_list, class_total_list)

        print(stage_stats)

        forgetting_rates = calculate_forgetting_rate(stage_stats)
        print("各阶段平均遗忘率:", [round(fr, 4) for fr in forgetting_rates])

        # if s == 13:
        #     total = 0.0
        #     correct = 0.0
        #     all_preds = []
        #     all_labels = []
        #     for index, images, labels in test_loader_all:
        #         images = Variable(images).cuda()
        #         images_1 = images[:, 0:144, :, :]
        #         images_2 = images[:, 144:, :, :]
        #         labels = labels.cuda()
        #         preds, _, _, _, _, _ = MyModel(images_1, images_2)
        #         _, preds = torch.max(preds.data, 1)
        #         # 收集预测结果和真实标签
        #         all_preds.extend(preds.cpu().numpy())
        #         all_labels.extend(labels.cpu().numpy())
        #         total += labels.size(0)
        #         correct += (preds == labels).sum().item()
        #     print('测试集准确率 : %.2f ,' % (100.0 * correct / total))
        #     np.save('all_preds.npy', all_preds)
