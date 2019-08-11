import torch


class BatchFirstLockedDropout(torch.nn.Module):
    """A batch-first implementation of variational dropout."""

    def __init__(self, p=0.5):
        self.p = p
        super().__init__()

    def forward(self, x):
        """
        Args:
            x: Input to apply dropout to. Shape should be (batch_size, time, feature_size)
        """
        if not self.training:
            return x
        x = x.clone()
        mask = x.new_empty(x.size(0), 1, x.size(2),
                           requires_grad=False).bernoulli_(1 - self.p)
        mask = mask.div_(1 - self.p)
        mask = mask.expand_as(x)
        return x * mask


"""The following is copied from torchnlp:
https://github.com/PetrochukM/PyTorch-NLP/blob/e852daececefc5dd089d0aa0ce39d0efb07f4537/torchnlp/nn/weight_drop.py

It's copied because there seems to be a bug with the module that's fixed below. See here for the issue:
https://github.com/salesforce/awd-lstm-lm/issues/79#ref-commit-2ca28cf

Copying the credit provided in that module:
**Thank you** to Sales Force for their initial implementation of :class:`WeightDrop`. Here is
    their `License
    <https://github.com/salesforce/awd-lstm-lm/blob/master/LICENSE>`__.
"""


def _do_nothing(*args, **kwargs):
    return


def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + "_raw", torch.nn.Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + "_raw")
            w = torch.nn.Parameter(
                torch.nn.functional.dropout(raw_w,
                                            p=dropout,
                                            training=module.training))
            setattr(module, name_w, w)

        return original_module_forward(*args)

    setattr(module, "forward", forward)

    # Running on GPU seems to require flattening parameters, and this causes a bug where the module doesn't
    # have `_flat_weights`. This can be addressed by having the `flatten_parameters` function do nothing.
    # The result seems to be that the RNN parameters take up more memory on the GPU.
    setattr(module, "flatten_parameters", _do_nothing)


class WeightDropGRU(torch.nn.GRU):
    """
    Wrapper around :class:`torch.nn.GRU` that adds ``weight_dropout`` named argument.
    Args:
        weight_dropout (float): The probability a weight will be dropped.
    """

    def __init__(self, *args, weight_dropout=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight_hh_l' + str(i) for i in range(self.num_layers)]
        _weight_drop(self, weights, weight_dropout)
