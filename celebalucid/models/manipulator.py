import os
import numpy as np
from dotmap import DotMap
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace
from lucent.optvis import transform, param, render

from celebalucid.models.inceptionv1 import InceptionV1
from celebalucid.utils import load_layer_info


class ModelManipulator(InceptionV1):
    def __init__(self,
                 pt_url):
        super(ModelManipulator, self).__init__(n_features=40,
                                               pretrained=False,
                                               redirected_ReLU=True)
        self._load_weights_from_url(pt_url)
        self.layer_info = load_layer_info()
        self.weights = self._load_weights()
        self.activations = {}
        self._register_activation_fw_hooks()

        # Assign device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(device).eval()

    def lucid(self, layer_n_channel, size=224, thresholds=[512], progress=False):
        layer_n_channel = self._correct_layer_n_channel(layer_n_channel)
        transforms = transform.standard_transforms.copy()
        transforms.append(lambda x: x * 255 - 117)
        def param_f(): return param.image(224, fft=True, decorrelate=True)
        try:
            img = render.render_vis(self,
                                    layer_n_channel,
                                    show_image=False,
                                    preprocess=False,
                                    progress=progress,
                                    transforms=transforms,
                                    thresholds=thresholds,
                                    param_f=param_f)[0][0]
        except AssertionError:
            raise AssertionError(
                'Invalid layer {}. Retrieve the list of layers with `model.layer_info`.'.format(
                    layer_n_channel)
            )

        img = (img*255).astype(np.uint8)
        return img

    # ========================================================================
    # Private functions
    # ========================================================================

    def _correct_layer_n_channel(self, layer_n_channel):
        layer, channel = layer_n_channel.split(':')
        layer += '_pre_relu_conv' if layer != 'logits' else ''
        return ':'.join([layer, channel])

    def _register_activation_fw_hooks(self):
        for module in self.modules():
            module.register_forward_hook(self._save_activation)

    def _save_activation(self, module, m_in, m_out):
        self.activations[module] = m_out

    def _load_weights(self):
        extracted_weights = DotMap()
        for name, weights in self.named_parameters():
            layer_name, weight_format = name.split('.')
            weight_format = weight_format[0]
            layer_name = layer_name.replace('_pre_relu_conv', '')
            extracted_weights[layer_name][weight_format] = weights.detach(
            ).cpu().numpy()

        return extracted_weights

    def _load_weights_from_url(self, url):
        self.load_state_dict(torch.hub.load_state_dict_from_url(
            url, progress=True), strict=False)
