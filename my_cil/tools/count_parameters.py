"""Print total and trainable parameter counts for the active model."""

from my_cil.training import MyCilTrainNet


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def print_model_params(model):
    total, trainable = count_parameters(model)
    print(f"Total params: {total:,} ({total / 1e6:.3f} M)")
    print(f"Trainable params: {trainable:,} ({trainable / 1e6:.3f} M)")
    print("\nPer-module breakdown:")
    for name, module in model.named_children():
        module_total, module_trainable = count_parameters(module)
        print(f"  {name}: total={module_total:,}, trainable={module_trainable:,}")


def main():
    model = MyCilTrainNet(3, 128, 128, 15).cpu()
    print_model_params(model)


if __name__ == "__main__":
    main()
