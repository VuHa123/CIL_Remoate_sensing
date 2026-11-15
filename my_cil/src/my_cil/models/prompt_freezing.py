import torch
from torch.nn.functional import cosine_similarity

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

# 示例使用
if __name__ == "__main__":
    # 假设我们有一些样本特征向量和提示token
    sample_vector = torch.randn(10)  # 样本特征向量, shape: (10,)
    prompt_tokens = torch.randn(5, 10)  # 提示token, shape: (5, 10)

    # 计算相似度
    similarity = calculate_similarity(sample_vector, prompt_tokens)

    # 根据相似度冻结或解冻token
    freeze_tokens(prompt_tokens, similarity, threshold=0.5)

    # 打印结果
    print("Similarity:", similarity)
    print("Frozen tokens:", [not token.requires_grad for token in prompt_tokens])