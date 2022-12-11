import torch


def init_module_weights(module):
    """Initialize the weights"""

    from src.model.modules import QuantizationModule

    # gumbel softmax requires special init
    if isinstance(module, QuantizationModule):
        module.weight_proj.weight.data.normal_(mean=0.0, std=1)
        module.weight_proj.bias.data.zero_()
        torch.nn.init.uniform_(module.codebooks)
    elif isinstance(module, torch.nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.5)
    elif isinstance(module, (torch.nn.LayerNorm, torch.nn.GroupNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, torch.nn.Conv1d):
        torch.nn.init.kaiming_normal_(module.weight.data)

    if (
        isinstance(module, (torch.nn.Linear, torch.nn.Conv1d))
        and module.bias is not None
    ):
        module.bias.data.zero_()
