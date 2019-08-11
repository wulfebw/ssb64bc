import numpy as np
import torch
import torchvision

import ssb64bc.models.torch_utils


class MultiframeMulticlassResnetActionModel(torchvision.models.ResNet):
    def __init__(self,
                 block=torchvision.models.resnet.BasicBlock,
                 layers=[1, 1, 1, 1],
                 n_frames=4,
                 n_channels=1,
                 **kwargs):
        super(MultiframeMulticlassResnetActionModel, self).__init__(block, layers, **kwargs)
        in_channels = n_frames * n_channels
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            return torch.nn.functional.softmax(logits, dim=1)


class MultiframeMultidiscreteResnetActionModel(torchvision.models.ResNet):
    def __init__(self,
                 n_classes_per_action,
                 block=torchvision.models.resnet.BasicBlock,
                 layers=[1, 1, 1, 1],
                 n_frames=4,
                 n_channels=1,
                 **kwargs):
        self.output_dim = sum(n_classes_per_action)
        self.softmax_edges = [0] + list(np.cumsum(n_classes_per_action).astype(int))
        kwargs["output_dim"] = self.output_dim
        super(MultiframeMulticlassResnetActionModel, self).__init__(block, layers, **kwargs)
        in_channels = n_frames * n_channels
        self.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    def predict(self, x):
        with torch.no_grad():
            logits = self.forward(x)
            probs = []
            for (start, end) in zip(self.softmax_edges, self.softmax_edges[1:]):
                probs += [torch.nn.functional.softmax(logits[:, start:end], dim=1)]
            probs = torch.cat((probs), dim=1)
            return probs


def resnet_feature_extractor(feature_dim,
                             n_frames=1,
                             n_channels=1,
                             block=torchvision.models.resnet.BasicBlock,
                             layers=[1, 1, 1, 1],
                             no_batch_norm=False):
    norm_layer = torch.nn.Module if no_batch_norm else None
    net = torchvision.models.ResNet(block, layers, norm_layer=norm_layer, num_classes=feature_dim)
    in_channels = n_frames * n_channels
    net.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return net


class RecurrentMultiframeMulticlassActionModel(torch.nn.Module):
    def __init__(self, output_dim, hidden_dim=128, dropout_prob=0.0, feature_extractor=None):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob
        self._build_model(feature_extractor)

    def _build_model(self, feature_extractor):
        if feature_extractor is None:
            feature_extractor = resnet_feature_extractor(feature_dim=self.hidden_dim)
        self.feature_extractor = feature_extractor
        # Variational dropout applied to the input and output of the rnn.
        self.rnn_dropout = torch_utils.BatchFirstLockedDropout(self.dropout_prob)
        # The initial hidden state of the rnn, which we attempt to learn.
        self.init_hidden = torch.nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        # This performs DropConnect on only the hidden-to-hidden weight matrices.
        self.rnn = torch_utils.WeightDropGRU(input_size=self.hidden_dim,
                                             hidden_size=self.hidden_dim,
                                             batch_first=True,
                                             weight_dropout=self.dropout_prob)
        self.fc = torch.nn.Linear(self.hidden_dim, self.output_dim)

    def init_hidden_states(self, batch_size, device=None):
        return self.init_hidden.repeat(1, batch_size, 1)

    def forward(self, x, hidden=None, return_hidden=False):
        """Performs a forward pass through the network.

        A major assumption of this network is that all the inputs are valid (i.e., the
        length of the different samples in the input all equal the maximum length).

        Args:
            x: Tensor of shape (batch_size, max_seq_len, image_shape...)
        """
        batch_size, max_seq_len = x.shape[:2]
        hidden = self.init_hidden_states(batch_size, x.device) if hidden is None else hidden

        # Flatten the sequence dimension and pass through the feature extractor.
        x = x.view(batch_size * max_seq_len, *x.shape[2:])
        x = self.feature_extractor(x)
        x = x.view(batch_size, max_seq_len, -1)

        # Pass through the rnn, applying dropout to the input and output.
        # Do not use batch norm after this point.
        x = self.rnn_dropout(x)
        x, hidden = self.rnn(x, hidden)
        x = self.rnn_dropout(x)

        # Flatten again, and apply another linear layer to output logits, then reshape to the sequence.
        x = x.contiguous()
        x = x.view(batch_size * max_seq_len, -1)
        x = self.fc(x)
        x = x.view(batch_size, max_seq_len, -1)

        return (x, hidden) if return_hidden else x

    def predict(self, x, hidden):
        with torch.no_grad():
            logits, hidden = self.forward(x, hidden=hidden, return_hidden=True)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            return probs, hidden

    def load_state_dict(self, d):
        # Due to the custom dropout, we have to delete a weight.
        del d["rnn.weight_hh_l0"]
        super().load_state_dict(d)
