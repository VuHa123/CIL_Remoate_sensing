import torch
import torch.nn as nn


class ModalityMappingNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ModalityMappingNetwork, self).__init__()
        self.mapping = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        return self.mapping(x)


class IncrementalModalityMapping(nn.Module):
    def __init__(self, input_dim, output_dim, mapping_factory=ModalityMappingNetwork):
        super(IncrementalModalityMapping, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mapping_factory = mapping_factory
        self.stages = nn.ModuleList()
        self.current_stage = 0

        # 初始化第一个阶段的网络
        self.add_new_stage()

    def add_new_stage(self, modality_trans_net=None):
        """ 添加一个新的轻量级网络模块 """
        factory = modality_trans_net or self.mapping_factory
        new_stage = factory(self.input_dim, self.output_dim)
        self.stages.append(new_stage)
        self.current_stage += 1

    def freeze_previous_stages(self):
        """ 冻结所有之前的阶段 """
        for stage in self.stages[:-1]:
            for param in stage.parameters():
                param.requires_grad = False

    def forward(self, x):
        """ 前向传播，只使用当前阶段的网络 """
        return self.stages[self.current_stage - 1](x)

    def next_stage(self):
        """ 进入下一个增量阶段 """
        self.freeze_previous_stages()
        self.add_new_stage()
