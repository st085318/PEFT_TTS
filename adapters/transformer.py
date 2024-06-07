from torch import nn as nn
from typing import List, Optional
from omegaconf import DictConfig

import torch
import torch.nn.functional as F

from nemo.collections.asr.parts.utils import adapter_utils
from nemo.collections.tts.modules.transformer import MultiHeadAttn, TransformerLayer, FFTransformerDecoder, FFTransformerEncoder, mask_from_lens
from nemo.core.classes import NeuralModule, typecheck
from adapters import adapter_mixins


class MultiHeadAttnMultispeakerAdapter(MultiHeadAttn, adapter_mixins.AdapterModuleMixinMultispeaker):
    def forward(self, inp, attn_mask=None, conditioning=None):
        return self._forward(inp, attn_mask, conditioning)

    def _forward(self, inp, attn_mask=None, conditioning=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp, conditioning)

        n_head, d_head = self.n_head, self.d_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)

        if self.is_adapter_available() and self.connection_type == 'multiplication':
            vk = self.forward_enabled_adapters(torch.cat((head_v, head_k), dim=2))
            head_v, head_k = torch.chunk(vk, 2, dim=2) 


        if self.is_adapter_available() and self.connection_type == 'residual':
            adapter_correction = self.forward_enabled_adapters(inp)
            q_add, k_add = torch.chunk(adapter_correction, 2, dim=2)
            head_q = head_q + q_add
            head_k = head_k + k_add 

        head_q = head_q.view(inp.size(0), inp.size(1), n_head, d_head)
        head_k = head_k.view(inp.size(0), inp.size(1), n_head, d_head)
        head_v = head_v.view(inp.size(0), inp.size(1), n_head, d_head)

        q = head_q.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        k = head_k.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)
        v = head_v.permute(2, 0, 1, 3).reshape(-1, inp.size(1), d_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            attn_mask = attn_mask.repeat(n_head, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float('inf'))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropatt(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(n_head, inp.size(0), inp.size(1), d_head)
        attn_vec = attn_vec.permute(1, 2, 0, 3).contiguous().view(inp.size(0), inp.size(1), n_head * d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out, conditioning)

        return output
    
    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg

    def add_adapter(self, name: str, cfg: dict, adapter_connection):
        self.connection_type = adapter_connection
        cfg["n_head"] = self.n_head
        cfg["d_head"] = self.d_head

        super().add_adapter(name, cfg)

        

class TransformerLayerMultispeakerAdapter(TransformerLayer, adapter_mixins.AdapterModuleMixinMultispeaker):
    def __init__(self, n_head, d_model, d_head, d_inner, kernel_size, dropout, condition_types=[], **kwargs):
        super(TransformerLayerMultispeakerAdapter, self).__init__(
            n_head, 
            d_model, 
            d_head, 
            d_inner, 
            kernel_size, 
            dropout,
            condition_types=condition_types
        )

        self.dec_attn = MultiHeadAttnMultispeakerAdapter(
            n_head, 
            d_model, 
            d_head, 
            dropout, 
            condition_types=condition_types
        )
        self.d_model = d_model

    def add_adapter(self, name: str, cfg: dict, adapter_connection):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        if self.resolve_adapter_module_name_(name)[1].startswith('attention'):
            self.dec_attn.add_adapter(name, cfg, adapter_connection)
        else:
            super().add_adapter(name, cfg)
        
    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        if self.resolve_adapter_module_name_(name)[1].startswith('attention'):
            self.dec_attn.set_enabled_adapters(name=name, enabled=enabled)
        else:
            super().set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        names.update(super().get_enabled_adapters())
        names.update(self.dec_attn.get_enabled_adapters())
        names = sorted(list(names))
        return names


class FFTransformerDecoderMultispeaker(FFTransformerDecoder):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        condition_types=[],
        adapter_connection=None,
        **kwargs
    ):
        super(FFTransformerDecoderMultispeaker, self).__init__(
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            kernel_size,
            dropout,
            dropatt,
            dropemb,
            pre_lnorm,
            condition_types
        )

        self.adapter_connection = adapter_connection
        self.layers = nn.ModuleList()
        for _ in range(n_layer):
            self.layers.append(
                TransformerLayerMultispeakerAdapter(
                    n_head, 
                    d_model, 
                    d_head, 
                    d_inner, 
                    kernel_size, 
                    dropout,
                    condition_types=condition_types
                )
            )

    def forward(self, input, seq_lens, conditioning=None):
        return self._forward(input, mask_from_lens(seq_lens).unsqueeze(2), conditioning)

    def _forward(self, inp, mask, conditioning):
        pos_seq = torch.arange(inp.size(1), device=inp.device).to(inp.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask
        inp = inp + pos_emb
        inp = self.cond_input(inp, conditioning)
        out = self.drop(inp)

        for layer in self.layers:
            out = layer(out, mask=mask, conditioning=conditioning)

        # out = self.drop(out)
        return out, mask


class FFTransformerEncoderMultispeaker(FFTransformerDecoderMultispeaker):
    def __init__(
        self,
        n_layer,
        n_head,
        d_model,
        d_head,
        d_inner,
        kernel_size,
        dropout,
        dropatt,
        dropemb=0.0,
        pre_lnorm=False,
        n_embed=None,
        d_embed=None,
        padding_idx=0,
        condition_types=[],
        adapter_connection=None,
    ):
        super(FFTransformerEncoderMultispeaker, self).__init__(
            n_layer,
            n_head,
            d_model,
            d_head,
            d_inner,
            kernel_size,
            dropout,
            dropatt,
            dropemb,
            pre_lnorm,
            condition_types,
            adapter_connection,
        )

        self.padding_idx = padding_idx
        self.word_emb = nn.Embedding(n_embed, d_embed or d_model, padding_idx=self.padding_idx)

    @property
    def input_types(self):
        return {
            "input": NeuralType(('B', 'T'), TokenIndex()),
            "conditioning": NeuralType(('B', 'T', 'D'), EncodedRepresentation(), optional=True),
        }


    def forward(self, input, conditioning=0):
        return self._forward(self.word_emb(input), (input != self.padding_idx).unsqueeze(2), conditioning)  # (B, L, 1)


class FFTransformerDecoderMultispeakerAdapter(FFTransformerDecoderMultispeaker, adapter_mixins.AdapterModuleMixinMultispeaker):
    def add_adapter(self, name: str, cfg: dict):
        cfg = self._update_adapter_cfg_input_dim(cfg)
        for fft_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            fft_layer.add_adapter(name, cfg, self.adapter_connection)

    def is_adapter_available(self) -> bool:
        return any([FFT_layer.is_adapter_available() for FFT_layer in self.layers])

    def set_enabled_adapters(self, name: Optional[str] = None, enabled: bool = True):
        for FFT_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            FFT_layer.set_enabled_adapters(name=name, enabled=enabled)

    def get_enabled_adapters(self) -> List[str]:
        names = set([])
        for FFT_layer in self.layers:  # type: adapter_mixins.AdapterModuleMixin
            names.update(FFT_layer.get_enabled_adapters())

        names = sorted(list(names))
        return names

    def _update_adapter_cfg_input_dim(self, cfg: DictConfig):
        cfg = adapter_utils.update_adapter_cfg_input_dim(self, cfg, module_dim=self.d_model)
        return cfg

class FFTransformerEncoderMultispeakerAdapter(
    FFTransformerDecoderMultispeakerAdapter,
    FFTransformerEncoderMultispeaker,
    adapter_mixins.AdapterModuleMixinMultispeaker
):
    pass