import torch


class MultiDiscreteCrossEntropyLoss(torch.nn.Module):
    """A combination of multiple cross entropy losses."""

    def __init__(self, edges, kwargs_list=None):
        """
        Args:
            edges: The sequence of start:end indices.
            For example, if we're making a multi-discrete softmax loss with number of outputs each
            [3,4,6]
            then edges would be 
            [0,3,7,13]

            kwargs_list: The key word args for each softmax. 
            These will be forwarded to the corresponding softmax based on index.
            Therefore if kwargs_list is not None, it must be a list of length
            equal to the number of sub-losses.
        """
        super().__init__()
        self.edges = edges
        self.losses = []
        # Number of edges minus one is the number of sub-losses.
        for i in range(len(edges) - 1):
            kwargs = dict()
            if kwargs_list is not None:
                assert i < len(kwargs_list)
                kwargs = kwargs_list[i]
                assert kwargs.get("reduction", "mean") in ["mean", "sum"], "Reduction must be used."
            self.losses += [torch.nn.CrossEntropyLoss(**kwargs)]

    def forward(self, logits, targets):
        loss = 0
        for i, (start, end) in enumerate(zip(self.edges, self.edges[1:])):
            # Assume the individual losses are weighted equally.
            loss += self.losses[i](logits[:, start:end], targets[:, i])
        return loss
