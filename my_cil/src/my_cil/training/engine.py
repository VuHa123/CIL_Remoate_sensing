"""
增量模型训练框架
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.functional import cosine_similarity
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from ..losses.club import CLUB
from ..losses.sinkhorn import SinkhornDistance
from ..models.prompt_vit import ViT_Prompt
from ..models.task_aware import TaskAwareLayer

from imblearn.over_sampling import RandomOverSampler

CUDA_VISIBLE_DEVICES=0
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ModalityMappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, rank=16):
        super(ModalityMappingNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank

        # 低秩分解: W ≈ B @ A
        # A: (input_dim, rank)
        # B: (rank, output_dim)
        self.down_proj = nn.Linear(input_dim, rank, bias=False)
        self.up_proj = nn.Linear(rank, output_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(output_dim))

        # 初始化
        nn.init.kaiming_uniform_(self.down_proj.weight, a=0, mode='fan_out')
        nn.init.zeros_(self.up_proj.weight)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.down_proj(x)  # (batch_size, rank)
        x = self.up_proj(x)  # (batch_size, output_dim)
        x = x + self.bias
        return x


def kaiming_normal_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='sigmoid')


def loss_modality_trans(x_model1, x_model2):
    estimator = CLUB(x_dim=x_model1.shape[2], y_dim=x_model2.shape[2], hidden_size=256).to(device)
    mul_loss = estimator.learning_loss(x_model1, x_model2)
    return mul_loss


def calculate_similarity(sample_vector, prompt_tokens):
    """
    计算样本特征向量与提示token的余弦相似度
    :param sample_vector: 样本特征向量, shape: (dim,)
    :param prompt_tokens: 提示token, shape: (num_tokens, dim)
    :return: 相似度, shape: (num_tokens,)
    """
    sample_vector = sample_vector.unsqueeze(0)  # shape: (1, dim)
    similarity = cosine_similarity(sample_vector, prompt_tokens, dim=1)  # shape: (num_tokens,)
    return similarity


def freeze_tokens(prompt_tokens, similarity, threshold=0.5):
    """
    根据相似度冻结或解冻token
    :param prompt_tokens: 提示token, shape: (num_tokens, dim)
    :param similarity: 相似度, shape: (num_tokens,)
    :param threshold: 相似度阈值
    """
    for i, sim in enumerate(similarity):
        if sim > threshold:
            prompt_tokens[i].requires_grad_(False)  # 冻结token
        else:
            prompt_tokens[i].requires_grad_(True)  # 保持可训练


class MyCilTrainNet(nn.Module):
    def __init__(self, num_group_tokens, group_feature_size, feature_size, num_classes,
                 hsi_channels=144, batch_size=16, init_lr=0.0001, base_lr=None,
                 fix_optimizer=False):
        """
        hsi_channels: number of HSI channels (144 raw, 40 after PCA as in the paper).
        base_lr: learning rate of the base session (default: init_lr).
        fix_optimizer: if True, rebuild Adam/scheduler at every session over the
            parameters that are actually trainable in that session (new fc, new
            projectors, task tokens), register task tokens as real parameters and
            compute OT class weights in dataset order. False = original behaviour.
        """
        super(MyCilTrainNet, self).__init__()
        self.hsi_channels = hsi_channels
        self.fix_optimizer = fix_optimizer

        # 增量类别管理
        self.n_classes = num_classes
        self.n_known = 0

        # 网络超参数'
        self.init_lr = init_lr
        self.base_lr = init_lr if base_lr is None else base_lr
        self.num_epochs = 50
        self.batch_size = batch_size

        # 模型训练参数
        self.momentum = 0.01
        self.weight_decay = 0.0001
        self.epsilon = 10

        # 增量任务阶段管理
        self.train_stage = 0
        self.stage_mapping_net_1 = nn.ModuleList()
        self.stage_mapping_net_2 = nn.ModuleList()

        # 样例集
        self.exemplar_sets = []
        self.m = []

        # 样本均值
        self.compute_means = True
        self.exemplar_means = []

        self.num_group_tokens = num_group_tokens
        self.group_feature_size = group_feature_size

        self.shared_group_tokens = nn.Parameter(torch.zeros(num_group_tokens, group_feature_size),
                                                requires_grad=True).to(device)
        # fix_optimizer: task tokens of each session, registered as parameters
        self.group_token_list = nn.ParameterList(
            [nn.Parameter(torch.zeros(num_group_tokens, group_feature_size))] if fix_optimizer else [])

        self.Prompt_Net_modality_1 = ViT_Prompt(image_size=11, patch_size=1, in_channels=hsi_channels, embed_dim=128,
                                                num_classes=15, depth=4, num_heads=4, drop_rate=0.1, pool_size=30,
                                                embedding_key='mean_max', prompt_init='uniform', top_k=3,
                                                prompt_length=5, batchwise_prompt=True, prompt_key_init='uniform',
                                                prompt_pool=True).to(device)

        self.Prompt_Net_modality_2 = ViT_Prompt(image_size=11, patch_size=1, in_channels=1, embed_dim=128,
                                                num_classes=15, depth=4, num_heads=4, drop_rate=0.1, pool_size=30,
                                                embedding_key='mean_max', prompt_init='uniform', top_k=3,
                                                prompt_length=5, batchwise_prompt=True, prompt_key_init='uniform',
                                                prompt_pool=True).to(device)

        # self.load_pretrain_weights(self.pretrained_path_1, self.pretrained_path_2)

        self.TaskAwareNet_1 = TaskAwareLayer(dim=128,
                                             depth=2,
                                             num_heads=4,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop=0.,
                                             attn_drop=0.,
                                             drop_path=0.,
                                             norm_layer=nn.LayerNorm,
                                             out_dim_grouping=128,
                                             num_heads_grouping=2,
                                             num_group_tokens=3,
                                             num_output_groups=3,
                                             hard_assignment=True).to(device)
        self.TaskAwareNet_2 = TaskAwareLayer(dim=128,
                                             depth=2,
                                             num_heads=4,
                                             mlp_ratio=4.,
                                             qkv_bias=True,
                                             qk_scale=None,
                                             drop=0.,
                                             attn_drop=0.,
                                             drop_path=0.,
                                             norm_layer=nn.LayerNorm,
                                             out_dim_grouping=128,
                                             num_heads_grouping=2,
                                             num_group_tokens=3,
                                             num_output_groups=3,
                                             hard_assignment=True).to(device)

        self.bn = nn.BatchNorm1d(feature_size)
        self.Tanh = nn.Tanh()
        self.fc = nn.Linear(feature_size, num_classes, bias=False)

        self.cls_loss = nn.CrossEntropyLoss()
        self.distill_loss = nn.MSELoss()
        self.cos_loss = nn.CosineSimilarity()
        self.aux_loss_fn = nn.BCEWithLogitsLoss()
        self.criterion = SinkhornDistance(eps=0.1, max_iter=200, reduction=None, dis='cos').to(device)

        self.optimizer = optim.Adam(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)

        # 动态学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.num_epochs)

        for param in self.TaskAwareNet_1.grouping.parameters():
            param.requires_grad = False
        for param in self.TaskAwareNet_2.grouping.parameters():
            param.requires_grad = False

        for param in self.Prompt_Net_modality_1.parameters():
            param.requires_grad = True

        for param in self.Prompt_Net_modality_2.parameters():
            param.requires_grad = True

        for param in self.Prompt_Net_modality_1.prompt.parameters():
            param.requires_grad = False
        for param in self.Prompt_Net_modality_2.prompt.parameters():
            param.requires_grad = False

    def group_tokens(self):
        if self.fix_optimizer:
            return torch.cat(list(self.group_token_list), dim=0)
        return self.shared_group_tokens

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, x_modality_1, x_modality_2):
        self.apply_stage_freezing()
        return self._forward(x_modality_1, x_modality_2)

    def apply_stage_freezing(self):
        if self.train_stage >= 2:
            for param in self.TaskAwareNet_1.grouping.parameters():
                param.requires_grad = False
            for param in self.TaskAwareNet_2.grouping.parameters():
                param.requires_grad = False

            for param in self.Prompt_Net_modality_1.parameters():
                param.requires_grad = False

            for param in self.Prompt_Net_modality_2.parameters():
                param.requires_grad = False

            for param in self.Prompt_Net_modality_1.prompt.parameters():
                param.requires_grad = True
            for param in self.Prompt_Net_modality_2.prompt.parameters():
                param.requires_grad = True

    def _forward(self, x_modality_1, x_modality_2):
        _, res_1 = self.Prompt_Net_modality_1.forward_prompt(x_modality_1)
        _, res_2 = self.Prompt_Net_modality_2.forward_prompt(x_modality_2)

        # 原始提示token
        prompt_1_in = res_1['prompt_in']
        prompt_2_in = res_2['prompt_in']

        sim_1 = res_1['reduce_sim']
        sim_2 = res_2['reduce_sim']

        sim_loss = -sim_1 - sim_2

        prompted_token_1_trans = prompt_1_in
        for b in self.stage_mapping_net_1:
            prompted_token_1_trans = b(prompted_token_1_trans)

        prompted_token_2_trans = prompt_2_in
        for b in self.stage_mapping_net_2:
            prompted_token_2_trans = b(prompted_token_2_trans)

        modality_mutual_info_loss = loss_modality_trans(prompt_1_in, prompted_token_2_trans) + loss_modality_trans(
            prompt_2_in, prompted_token_1_trans)

        prompted_1_final_in = torch.cat([prompt_1_in, prompted_token_2_trans], dim=1)
        prompted_2_final_in = torch.cat([prompt_2_in, prompted_token_1_trans], dim=1)

        prompted_embedding_1 = torch.cat([prompted_1_final_in, res_1['x_embed']], dim=1)
        prompted_embedding_2 = torch.cat([prompted_2_final_in, res_2['x_embed']], dim=1)

        res_medium_1 = self.Prompt_Net_modality_1.forward_features(prompted_embedding_1, res_1)
        res_medium_2 = self.Prompt_Net_modality_2.forward_features(prompted_embedding_2, res_2)

        res_1_head = self.Prompt_Net_modality_1.forward_head(res_medium_1)
        res_2_head = self.Prompt_Net_modality_2.forward_head(res_medium_2)

        features_medium_1 = res_1_head['wto_prompt_feature']
        features_medium_2 = res_2_head['wto_prompt_feature']

        prompt_out_1 = res_1_head['prompt_feature']
        prompt_out_2 = res_2_head['prompt_feature']

        # prompt_out_mean = torch.cat([prompt_out_1, prompt_out_2], dim=1)
        features_medium = torch.cat([features_medium_1, features_medium_2], dim=1)
        features_medium_mean = features_medium.mean(dim=1)

        _, x_1_feature = self.TaskAwareNet_1(features_medium_1, self.group_tokens(), prompt_out_1)
        _, x_2_feature = self.TaskAwareNet_2(features_medium_2, self.group_tokens(), prompt_out_2)

        x_feature = torch.cat([x_1_feature, x_2_feature], dim=1)
        x_feature_mean = x_feature.mean(dim=1)

        x = self.bn(x_feature_mean)
        x = self.Tanh(x)
        x = self.fc(x)
        return x, x_feature, x_feature_mean, modality_mutual_info_loss, sim_loss, features_medium_mean

    def incremental_classes_fc(self, n):
        in_features = self.fc.in_features
        out_features = self.fc.out_features
        weight = self.fc.weight.data

        self.fc = nn.Linear(in_features, out_features + n, bias=False)
        kaiming_normal_init(self.fc.weight)
        self.fc.weight.data[:out_features] = weight
        self.n_classes += n

        self.train_stage += 1

    def classify(self, image_1, image_2):
        # 利用余弦相似度算法进行分类
        batch_size = image_1.size(0)

        if self.compute_means:
            print('computing means of exemplars...')
            exemplar_means = []
            for P_y in self.exemplar_sets:
                features = []
                # 为每个P_y中的样本提取特征
                for ex in P_y:
                    ex = Variable(torch.FloatTensor(ex)).to(device)
                    ex_1 = ex[0:self.hsi_channels, :, :]
                    ex_2 = ex[self.hsi_channels:, :, :]
                    # unsqueeze(0) 的作用:是 (C, H, W) 的图像数据转换为 (1, C, H, W) 形式的批次数据
                    # 这样才能正确地喂给模型进行前向计算。
                    _, _, feature, _, _, _ = self.forward(ex_1.unsqueeze(0), ex_2.unsqueeze(0))
                    feature = feature.squeeze(0)
                    feature.data = feature.data / feature.data.norm()
                    features.append(feature)
                # features：（N,D1,D2,...）
                features = torch.stack(features, dim=0)
                # 对维度0计算平均值，也就是计算每一类的平均特征向量
                mu_y = features.mean(0).squeeze()
                mu_y.data = mu_y.data / mu_y.data.norm()
                exemplar_means.append(mu_y)
            self.exemplar_means = exemplar_means
            self.compute_means = False
            print('Done')

        exemplar_means = self.exemplar_means
        means = torch.stack(exemplar_means, dim=0)  # (N,D1,D2,...)
        means = torch.stack([means] * batch_size)  # (batch_size,N,D1,D2,...)
        means = means.transpose(1, 2)  # (batch_size,D1,D2,...,N)

        _, _, feature, _, _, _ = self.forward(image_1, image_2)  # (batch_size,D1,D2,...)
        for i in range(feature.size(0)):
            feature.data[i] = feature.data[i] / feature.data.norm()
        feature = feature.unsqueeze(2)  # (batch_size,D1,D2,...,1)
        feature = feature.expand_as(means)  # (batch_size,D1,D2,...,N)

        # 使用余弦相似度计算
        cos_similarity = F.cosine_similarity(feature, means, dim=1)  # (batch_size,N)
        _, preds = cos_similarity.max(1)

        return preds

    def construct_exemplar_set(self, images, m):
        features = []
        for img in images:
            x = Variable(torch.FloatTensor(img)).to(device)
            x = x.unsqueeze(0)
            x_1 = x[:, :self.hsi_channels, :, :]
            x_2 = x[:, self.hsi_channels:, :, :]
            _, _, feature, _, _, _ = self.forward(x_1, x_2)
            feature = feature.data.cpu().numpy()
            feature = feature / np.linalg.norm(feature)
            features.append(feature)

        # features:(batch_size,1,feature_size)
        features = np.array(features)
        print('features_shape: ', features.shape)
        class_mean = np.mean(features, axis=0)
        class_mean = class_mean / np.linalg.norm(class_mean)
        print('class_mean_shape: ', class_mean.shape)

        exemplar_set = []
        now_class_sum = np.zeros((1, 128))
        for k in range(m):
            phi = features
            # mu:(1,feature_size)
            mu = class_mean
            # x:(batch_size,1,feature_size)
            x = mu - (now_class_sum + phi) / (k + 1)
            # x:(batch_size)
            x = np.linalg.norm(np.squeeze(x), axis=1)
            i = np.argmin(x)
            now_class_sum += features[i]

            exemplar_set.append(images[i])

        self.exemplar_sets.append(np.array(exemplar_set))

    def reduce_exemplar_set(self, m):
        # 遍历存储所有类别 exemplar 集合的列表或数组
        for y, P_y in enumerate(self.exemplar_sets):
            # 对于每一个类别 y，取其 exemplar 集合 P_y 中的前 m 个元素
            # [:m] 是 Python 切片操作，用于获取从开始到第 m 个（不包括 m 本身）的所有元素
            self.exemplar_sets[y] = P_y[:m]
        print('reduce_exemplar_set Done.')

    def combine_dataset_with_exemplar(self, dataset):
        print('data_set_length_1: ', dataset.train_data.shape)
        for y, P_y in enumerate(self.exemplar_sets):
            exemplar_images = np.array(P_y)
            # 创建一个与 P_y （当前类别的代表性样本集）长度相同的标签列表
            exemplar_labels = np.array([y] * len(P_y))

            dataset.train_data = np.concatenate((dataset.train_data, exemplar_images))
            dataset.train_labels = np.concatenate((dataset.train_labels, exemplar_labels))
        return dataset

    # 计算每个类别的权重
    def calculate_class_weights(self, dataloader, beta=0.9999):
        # 提取所有样本的类别标签
        targets = []
        for _, _, target in dataloader:
            targets.extend(target.numpy())

        # 计算每个类别的样本数量
        unique_targets, img_num_list = np.unique(targets, return_counts=True)

        # 计算每个类别的有效样本数
        effective_num = 1.0 - np.power(beta, img_num_list)

        # 计算每个类别的权重
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(unique_targets)

        # 将权重转换为 PyTorch 张量
        weights = torch.FloatTensor(per_cls_weights).to(device)

        # 根据训练集的目标标签生成权重缓冲区
        weightsbuffer = torch.tensor([weights[cls_i] for cls_i in targets]).to(device)

        return weights, weightsbuffer

    def class_weights_from_labels(self, labels, beta=0.9999):
        labels = np.asarray(labels)
        unique_targets, img_num_list = np.unique(labels, return_counts=True)
        effective_num = 1.0 - np.power(beta, img_num_list)
        per_cls_weights = (1.0 - beta) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(unique_targets)
        lookup = dict(zip(unique_targets.tolist(), per_cls_weights.tolist()))
        weights = torch.FloatTensor(per_cls_weights).to(device)
        buffer = torch.tensor([lookup[int(c)] for c in labels], dtype=torch.float32, device=device)
        return weights, buffer

    # 获取平衡数据集并进行上采样
    def get_balanced_data(self, train_loader, num_classes, num_samples_per_class=10):
        balanced_data = []
        balanced_labels = []
        class_counts = [0] * num_classes
        for index, inputs, labels in train_loader:
            for input, label in zip(inputs, labels):
                if class_counts[label] < num_samples_per_class:
                    balanced_data.append(input.numpy())
                    balanced_labels.append(label.numpy())
                    class_counts[label] += 1
                if all(count == num_samples_per_class for count in class_counts):
                    break
            if all(count == num_samples_per_class for count in class_counts):
                break

        # 将数据转换为numpy数组
        balanced_data = np.array(balanced_data)
        balanced_labels = np.array(balanced_labels)

        # 记录原始形状 (假设原始形状为 (batch_size, height, width, channels))
        original_shape = balanced_data.shape[1:]  # (height, width, channels)

        # 展平数据
        n_samples = balanced_data.shape[0]
        flattened_data = balanced_data.reshape(n_samples, -1)

        # 上采样
        ros = RandomOverSampler(random_state=42)
        balanced_data_flattened, balanced_labels = ros.fit_resample(flattened_data, balanced_labels)

        # 恢复原始形状
        balanced_data_restored = balanced_data_flattened.reshape(-1, *original_shape)

        # 转换为Tensor
        balanced_data_tensor = torch.tensor(balanced_data_restored, dtype=torch.float32)
        balanced_labels_tensor = torch.tensor(balanced_labels, dtype=torch.long)

        # 创建TensorDataset
        dataset = TensorDataset(balanced_data_tensor, balanced_labels_tensor)

        # 创建DataLoader
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        return dataloader

    def softmax_normalize(self, weights, temperature=1.):
        return nn.functional.softmax(weights / temperature, dim=0)

    # 训练更新表征
    def update_representation(self, dataset, num_epochs):
        classes = list(set(dataset.train_labels))

        new_classes = [cls for cls in classes if cls > self.n_classes - 1]
        self.incremental_classes_fc(len(new_classes))
        self.to(device)
        print("new classes:", new_classes)
        print("new_stage:", self.train_stage)

        dataset = self.combine_dataset_with_exemplar(dataset)
        print('data_set_length_2: ', dataset.train_data.shape)

        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=8)

        optimizer = self.optimizer
        scheduler = self.scheduler

        # 增加新的轻量映射网络，冻结旧映射网络
        if self.train_stage >= 0:
            new_stage_1 = ModalityMappingNetwork(128, 128).to(device)
            new_stage_1.apply(kaiming_normal_init)
            new_stage_2 = ModalityMappingNetwork(128, 128).to(device)
            new_stage_2.apply(kaiming_normal_init)
            self.stage_mapping_net_1.append(new_stage_1)
            self.stage_mapping_net_2.append(new_stage_2)

        for stage in self.stage_mapping_net_1[:-1]:
            for param in stage.parameters():
                param.requires_grad = False

        for stage in self.stage_mapping_net_2[:-1]:
            for param in stage.parameters():
                param.requires_grad = False

        # 增量阶段增加新的task_tokens
        if self.train_stage >= 2:
            if self.fix_optimizer:
                for old_tokens in self.group_token_list:
                    old_tokens.requires_grad = False
                self.group_token_list.append(
                    nn.Parameter(torch.zeros(self.num_group_tokens, self.group_feature_size, device=device)))
            else:
                new_stage_token = nn.Parameter(torch.zeros(self.num_group_tokens, self.group_feature_size)).to(device)
                self.shared_group_tokens = torch.cat([self.shared_group_tokens.detach(), new_stage_token], dim=0)
        else:
            self.shared_group_tokens = self.shared_group_tokens

        if self.fix_optimizer:
            self.to(device)
            self.apply_stage_freezing()
            params = self.trainable_parameters()
            lr = self.base_lr if self.train_stage == 1 else self.init_lr
            self.optimizer = optim.Adam(params, lr=lr, weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs)
            optimizer, scheduler = self.optimizer, self.scheduler
            print("trainable params this session: %.3fM (lr=%g)" % (sum(p.numel() for p in params) / 1e6, lr))
            # OT class weights, indexed in dataset order (dataset index == `index` in batches)
            _, dataset_weights = self.class_weights_from_labels(dataset.train_labels)

        print("已添加新的task_token!")

        # 训练当前阶段数据
        with (tqdm(total=num_epochs) as pbar):
            print("train stage: ", self.train_stage)

            if self.train_stage >= 2:

                self.train()
                balanced_dataloader = self.get_balanced_data(loader, self.n_classes)
                balanced_data, balanced_labels = next(iter(balanced_dataloader))
                balanced_data = Variable(torch.FloatTensor(balanced_data)).to(device)
                balanced_data_hsi = balanced_data[:, 0:self.hsi_channels, :, :]
                balanced_data_lidar = balanced_data[:, self.hsi_channels:, :, :]
                balanced_labels = torch.tensor(balanced_labels, dtype=torch.long).to(device)

                _, _, feature_balanced, _, _, _ = self.forward(balanced_data_hsi, balanced_data_lidar)

                for epoch in range(num_epochs):

                    for i, batch in enumerate(loader):
                        if len(batch[0]) != self.batch_size:
                            continue
                        index, images, labels = tuple(t.to(device) for t in batch)
                        # images = Variable(torch.FloatTensor(images)).to(device)
                        images_1 = images[:, 0:self.hsi_channels, :, :]
                        images_2 = images[:, self.hsi_channels:, :, :]

                        labels = Variable(labels).to(device)
                        if self.fix_optimizer:
                            weights_buffer = dataset_weights
                        else:
                            _, weights_buffer = self.calculate_class_weights(loader)
                        weights = Variable(weights_buffer[index], requires_grad=True).to(device)

                        self.eval()
                        Attoptimizer = torch.optim.SGD([weights], lr=0.01, momentum=0.9, weight_decay=0.0001)

                        Attoptimizer.zero_grad()

                        _, _, feature_imbalanced, _, _, _ = self.forward(
                            images_1, images_2)

                        # 计算OT距离

                        feature_imbalanced = feature_imbalanced.detach()
                        feature_balanced = feature_balanced.detach()
                        ot_loss = self.criterion(feature_imbalanced, feature_balanced, weights.squeeze())

                        ot_loss.backward()
                        Attoptimizer.step()

                        self.train()
                        optimizer.zero_grad()
                        logits_imbalanced, _, feature_mean, modality_mutual_info_loss, sim_loss, feature_medium = self.forward(
                            images_1,
                            images_2)
                        logits_balanced, _, _, _, _, _ = self.forward(balanced_data_hsi, balanced_data_lidar)
                        cls_loss_imbalanced = self.cls_loss(logits_imbalanced, labels)
                        cls_loss_balanced = self.cls_loss(logits_balanced, balanced_labels)

                        distill_loss = self.distill_loss(feature_mean, feature_medium)

                        aux_labels = torch.zeros_like(labels, dtype=torch.float32).to(device)
                        aux_labels[labels >= self.n_classes - len(new_classes)] = 1.0

                        aux_loss = self.aux_loss_fn(logits_imbalanced[:, -len(new_classes):].mean(dim=1), aux_labels)

                        loss = torch.sum(
                            cls_loss_imbalanced * weights.data) + torch.mean(
                            cls_loss_balanced) + 0.01 * modality_mutual_info_loss + sim_loss + aux_loss + distill_loss

                        loss.backward()
                        optimizer.step()

                        if i % 10 == 0:
                            pbar.set_description(
                                'Epoch:{}/{} | Iter:{}/{} | cls_loss_imbalanced:{:.4f}| aux_loss:{:.4f}'.format(
                                    epoch + 1, num_epochs, i + 1, len(loader), cls_loss_imbalanced.item(),
                                    aux_loss.item()
                                ))

                    scheduler.step()
                    self.eval()
                    pbar.update(1)
            else:
                for epoch in range(num_epochs):

                    self.train()
                    for i, (index, images, labels) in enumerate(loader):
                        images = Variable(torch.FloatTensor(images)).to(device)
                        images_1 = images[:, 0:self.hsi_channels, :, :]
                        images_2 = images[:, self.hsi_channels:, :, :]

                        labels = Variable(labels).to(device)

                        optimizer.zero_grad()

                        logits, _, _, modality_mutual_info_loss, sim_loss, _ = self.forward(images_1,
                                                                                            images_2)

                        cls_loss = self.cls_loss(logits, labels)
                        loss = cls_loss + modality_mutual_info_loss + sim_loss

                        loss.backward()
                        optimizer.step()

                        if i % 10 == 0:
                            pbar.set_description(
                                'Epoch: {}/{} | Iter: {}/{} | cls_loss: {:.4f} | sim_loss: {:.4f} '.format(
                                    epoch + 1, num_epochs, i + 1, len(loader), cls_loss.item(), sim_loss.item()))
                    scheduler.step()
                    self.eval()
                    pbar.update(1)


# Descriptive public name; keep ``MyCilTrainNet`` for historical checkpoints.
ContinualLearner = MyCilTrainNet
