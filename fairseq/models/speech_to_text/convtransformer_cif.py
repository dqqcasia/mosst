#!/usr/bin/env python3

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq import checkpoint_utils, utils
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import Embedding, Linear, TransformerDecoder
from fairseq.modules import LayerNorm, PositionalEmbedding, TransformerEncoderLayer
from torch import Tensor
from fairseq.models.speech_to_text.convtransformer import TransformerDecoderNoExtra

from fairseq.file_io import PathManager

PAD_IDX = 1
EOS_IDX = 2
eps = 1e-7
THRESHOLD = 0.999

logger = logging.getLogger(__name__)


def sequence_mask(lengths, maxlen=None, depth=None, dtype=torch.float, reverse=False):
    if maxlen is None:
        maxlen = lengths.max()
    mask = torch.ones((len(lengths), maxlen),
                      device=lengths.device,
                      dtype=torch.uint8).cumsum(dim=1) <= lengths.unsqueeze(0).t()
    if depth:
        mask = mask.unsqueeze(-1).repeat(1, 1, depth)

    mask = mask.type(dtype)

    if reverse:
        if dtype == torch.bool:
            mask = ~mask
        else:
            mask = 1 - mask

    return mask


@register_model("convtransformer_cif")
class ConvTransformerCIFModel(FairseqEncoderDecoderModel):
    """
    Transformer-based Speech translation model from ESPNet-ST
    https://arxiv.org/abs/2004.10234
    """
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--input-feat-per-channel",
            type=int,
            metavar="N",
            help="encoder input dimension per input channel",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            "--relu-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--decoder-output-dim",
            type=int,
            metavar="N",
            help="decoder output dimension (extra linear layer if different from decoder embed dim)",
        )
        parser.add_argument(
            "--share-decoder-input-output-embed",
            action="store_true",
            help="share decoder input and output embeddings",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-encoder-from",
            type=str,
            metavar="STR",
            help="model to take encoder weights from (for initialization)",
        )
        parser.add_argument(
            "--load-pretrained-decoder-from",
            type=str,
            metavar="STR",
            help="model to take decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--conv-out-channels",
            type=int,
            metavar="INT",
            help="the number of output channels of conv layer",
        )

    @classmethod
    def build_encoder(cls, args):
        encoder = ConvTransformerCIFEncoder(args)
        if getattr(args, "load_pretrained_encoder_from", None):
            if PathManager.exists(args.load_pretrained_encoder_from):
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=args.load_pretrained_encoder_from
                )
            else:
                logger.warning(("Pretrained encoder file not found: {}".format(args.load_pretrained_encoder_from)))
        return encoder

    @classmethod
    def build_decoder(cls, args, task, embed_tokens):
        decoder = TransformerDecoderNoExtra(args, task.target_dictionary, embed_tokens)
        if getattr(args, "load_pretrained_decoder_from", None):
            if PathManager.exists(args.load_pretrained_decoder_from):
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=args.load_pretrained_decoder_from
                )
            else:
                logger.warning(("Pretrained decoder file not found: {}".format(args.load_pretrained_decoder_from)))
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        decoder_embed_tokens = build_embedding(
            task.target_dictionary, args.decoder_embed_dim
        )
        encoder = cls.build_encoder(args)
        decoder = cls.build_decoder(args, task, decoder_embed_tokens)
        return cls(encoder, decoder)

    @staticmethod
    @torch.jit.unused
    def set_batch_first(lprobs):
        lprobs.batch_first = True

    def get_normalized_probs(
            self,
            net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
            log_probs: bool,
            sample: Optional[Dict[str, Tensor]] = None,
    ):
        # net_output['encoder_out'] is a (B, T, D) tensor
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs, sample)
        # if self.training:
        self.set_batch_first(lprobs)
        return lprobs

    def output_layout(self):
        return "BTD"

    """
    The forward method inherited from the base class has a **kwargs argument in
    its input, which is not supported in torchscript. This method overrites the forward
    method definition without **kwargs.
    """

    # def get_normalized_probs(self, net_output, log_probs):
    #     """Get normalized probabilities (or log probs) from a net's output."""
    #
    #     logits = net_output["logits"]
    #     if log_probs:
    #         res = utils.log_softmax(logits.float(), dim=-1)
    #     else:
    #         res = utils.softmax(logits.float(), dim=-1)
    #
    #     res.batch_first = True
    #
    #     return res

    def forward(self, src_tokens, src_lengths, prev_output_tokens, target_lengths=None):
        # print("src_lengths", src_lengths)
        # print("target_lengths", target_lengths)
        # target_lengths = src_lengths / 10
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, target_lengths=target_lengths)
        # hidden_encoded = encoder_out['encoder_out'][:, :, :-1]

        # encoder_out['encoder_padding_mask']为啥都是false？

        # if self.training:
        #     _alphas, num_output = self.resize(alphas, target_lengths)
        # else:
        #     _alphas, num_output = self.resize(alphas)

        # cif_outputs = self.cif(encoder_out, _alphas)

        # print("cif_outputs['encoder_out'][0].shape", cif_outputs['encoder_out'][0].shape)   # [12, 5, 512]
        # print("cif_outputs['encoder_padding_mask'][0].shape", cif_outputs['encoder_padding_mask'][0].shape) # [5, 12]
        # if cif_outputs['encoder_out'][0].shape[0] != cif_outputs['encoder_padding_mask'][0].shape[1]:
        #     import pdb;pdb.set_trace()

        # if self.training and cif_outputs['encoder_out'][0].abs().sum(-1).ne(0).sum() != target_lengths.sum():
        #     print('_alphas:\t', _alphas.sum(-1))
        #     print('alphas:\t', alphas.sum(-1))
        #     print('target:\t', target_lengths)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        # import pdb;pdb.set_trace()
        # return decoder_out # tuple ?
        return {'logits': decoder_out, 'len_logits': target_lengths,
                'alphas': encoder_out["alphas"], 'num_output': encoder_out["num_output"]}


class ConvTransformerCIFEncoder(FairseqEncoder):
    """Conv + Transformer encoder"""

    def __init__(self, args):
        """Construct an Encoder object."""
        super().__init__(None)

        self.dropout = args.dropout
        self.embed_scale = (
            1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
        )
        self.padding_idx = 1
        self.in_channels = 1
        self.input_dim = args.input_feat_per_channel
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, args.conv_out_channels, 3, stride=2, padding=3 // 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                args.conv_out_channels,
                args.conv_out_channels,
                3,
                stride=2,
                padding=3 // 2,
            ),
            torch.nn.ReLU(),
        )
        transformer_input_dim = self.infer_conv_output_dim(
            self.in_channels, self.input_dim, args.conv_out_channels
        )
        self.out = torch.nn.Linear(transformer_input_dim, args.encoder_embed_dim)
        self.embed_positions = PositionalEmbedding(
            args.max_source_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            learned=False,
        )

        self.transformer_layers = nn.ModuleList([])
        self.transformer_layers.extend(
            [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
        )
        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

        self.proj=self.Linear(args.encoder_embed_dim - 1, args.encoder_embed_dim)

    def pooling_ratio(self):
        return 4

    def infer_conv_output_dim(self, in_channels, input_dim, out_channels):
        sample_seq_len = 200
        sample_bsz = 10
        x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
        x = torch.nn.Conv2d(1, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
        x = x.transpose(1, 2)
        mb, seq = x.size()[:2]
        return x.contiguous().view(mb, seq, -1).size(-1)

    @torch.jit.unused
    def forward_non_torchscript(self, net_input: Dict[str, Tensor]):
        # encoder_input = {
        #     k: v for k, v in net_input.items() if k != "prev_output_tokens"
        # }
        encoder_input = {
            k: v for k, v in net_input.items() if k != "prev_output_tokens"
        }
        return self.forward(**encoder_input)

    def get_alphas(self, encoder_output):
        padding_mask = encoder_output['encoder_padding_mask'][0]

        if 'encoded' in encoder_output.keys():
            alphas = torch.transpose(encoder_output['encoded'][0], 0, 1)[:, :, -1]
        else:
            alphas = torch.transpose(encoder_output['encoder_out'][0], 0, 1)[:, :, -1]
        alphas = torch.sigmoid(alphas)
        alphas = alphas * (~padding_mask).float()

        return alphas

    def cif(self, encoder_output, alphas, threshold=THRESHOLD, log=False):
        if type(encoder_output) is Tensor:
            hidden = torch.transpose(encoder_output, 0, 1)
        elif 'encoded' in encoder_output.keys():
            hidden = torch.transpose(encoder_output['encoded'][0], 0, 1)[:, :, :-1]
        else:
            hidden = torch.transpose(encoder_output['encoder_out'][0], 0, 1)[:, :, :-1]

        device = hidden.device
        B, T, H = hidden.size()

        # loop varss
        integrate = torch.zeros([B], device=device)
        frame = torch.zeros([B, H], device=device)
        # intermediate vars along time
        list_fires = []
        list_frames = []

        for t in range(T):
            alpha = alphas[:, t]
            distribution_completion = torch.ones([B], device=device) - integrate

            integrate += alpha
            list_fires.append(integrate)

            fire_place = integrate >= threshold
            integrate = torch.where(fire_place,
                                    integrate - torch.ones([B], device=device),
                                    integrate)
            cur = torch.where(fire_place,
                              distribution_completion,
                              alpha)
            remainds = alpha - cur

            frame += cur[:, None] * hidden[:, t, :]
            list_frames.append(frame)
            frame = torch.where(fire_place[:, None].repeat(1, H),
                                remainds[:, None] * hidden[:, t, :],
                                frame)

            if log:
                print('t: {}\t{:.3f} -> {:.3f}|{:.3f} fire: {}'.format(
                    t, integrate[log], cur[log], remainds[log], fire_place[log]))

        fires = torch.stack(list_fires, 1)
        frames = torch.stack(list_frames, 1)
        list_ls = []
        len_labels = torch.round(alphas.sum(-1)).int()
        max_label_len = len_labels.max()
        for b in range(B):
            fire = fires[b, :]
            l = torch.index_select(frames[b, :, :], 0, torch.where(fire >= threshold)[0])
            pad_l = torch.zeros([max_label_len - l.size(0), H], device=device)
            list_ls.append(torch.cat([l, pad_l], 0))

            if log:
                print(b, l.size(0))

        if log:
            print('fire:\n', fires[log])
            print('fire place:\n', torch.where(fires[log] >= threshold))

        return torch.stack(list_ls, 0)

    def resize(self, alphas, target_lengths, noise=0.0, threshold=THRESHOLD):
        """
        alpha in thresh=1.0 | (0.0, +0.21)
        target_lengths: if None, apply round and resize, else apply scaling
        """
        device = alphas.device
        # sum
        _num = alphas.sum(-1)

        num = target_lengths.float()
        num = num + noise * torch.rand(alphas.size(0)).to(device)

        # scaling
        _alphas = alphas * (num / _num)[:, None].repeat(1, alphas.size(1))

        # rm attention value that exceeds threashold
        count = 0
        while len(torch.where(_alphas > threshold)[0]):
            count += 1
            if count > 10:
                break
            print('fixing alpha')
            xs, ys = torch.where(_alphas > threshold)
            for x, y in zip(xs, ys):
                if _alphas[x][y] >= threshold:
                    mask = _alphas[x].ne(0).float()
                    mean = 0.5 * _alphas[x].sum() / mask.sum()
                    _alphas[x] = _alphas[x] * 0.5 + mean * mask

        return _alphas, _num

    def Linear(self, in_features, out_features, bias=True):
        m = nn.Linear(in_features, out_features, bias)
        nn.init.xavier_uniform_(m.weight)
        if bias:
            nn.init.constant_(m.bias, 0.0)
        return m

    def forward(self, src_tokens, src_lengths, target_lengths=None):
        """Encode input sequence.
        :param torch.Tensor xs: input tensor
        :param torch.Tensor masks: input mask
        :return: position embedded tensor and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]:
        """
        # torch.set_printoptions(profile="full")
        # print("src_lengths", src_lengths)
        # print("target_lengths",target_lengths)
        # torch.set_printoptions(profile="default") # reset
        # import pdb;pdb.set_trace()
        bsz, max_seq_len, _ = src_tokens.size()
        x = (
            src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
                .transpose(1, 2)
                .contiguous()
        )
        x = self.conv(x)
        bsz, _, output_seq_len, _ = x.size()
        x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
        x = self.out(x)
        x = self.embed_scale * x

        subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
        input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
        input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(input_len_0.device)
        input_lengths = torch.min(
            input_len_0, input_len_1
        )

        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
        x += positions
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.transformer_layers:
            x = layer(x, encoder_padding_mask)

        # if not encoder_padding_mask.any():
        #     maybe_encoder_padding_mask = None
        # else:
        #     maybe_encoder_padding_mask = encoder_padding_mask
        maybe_encoder_padding_mask = encoder_padding_mask

        encoder_output = {
            "encoder_out": [x],
            "encoder_padding_mask": [maybe_encoder_padding_mask]
            if maybe_encoder_padding_mask is not None
            else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }

        alphas = self.get_alphas(encoder_output)

        if self.training:
            # gold_rate = self.set_gold_rate()
            decode_length = target_lengths
            # gold_ids = target_lengths
            noise = 0.0
        else:
            # gold_rate = 0.0
            decode_length = torch.round(alphas.sum(-1)).int()
            # gold_ids = None
            noise = 0.0
        _alphas, num_output = self.resize(alphas, decode_length, noise=noise)
        padding_mask = ~sequence_mask(decode_length).bool()
        # cif_outputs = self.cif(encoder_output, _alphas)
        cif_outputs = self.cif(encoder_output["encoder_out"][0][:, :, :-1], _alphas)
        cif_outputs = self.proj(cif_outputs)

        return {
            "encoder_out": [torch.transpose(cif_outputs, 0, 1)],
            "encoder_padding_mask": [padding_mask]
            if maybe_encoder_padding_mask is not None
            else [],
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
            "alphas": alphas,
            "num_output": num_output,
        }

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
        if len(encoder_out["encoder_padding_mask"]) == 0:
            new_encoder_padding_mask = []
        else:
            new_encoder_padding_mask = [
                (encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)
            ]
        if len(encoder_out["encoder_embedding"]) == 0:
            new_encoder_embedding = []
        else:
            new_encoder_embedding = [
                (encoder_out["encoder_embedding"][0]).index_select(0, new_order)
            ]
        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,
            "encoder_padding_mask": new_encoder_padding_mask,
            "encoder_embedding": new_encoder_embedding,
            "encoder_states": encoder_states,
            "src_tokens": [],
            "src_lengths": [],
        }


# class ConvTransformerEncoder(FairseqEncoder):
#     """Conv + Transformer encoder"""
#
#     def __init__(self, args):
#         """Construct an Encoder object."""
#         super().__init__(None)
#
#         self.dropout = args.dropout
#         self.embed_scale = (
#             1.0 if args.no_scale_embedding else math.sqrt(args.encoder_embed_dim)
#         )
#         self.padding_idx = 1
#         self.in_channels = 1
#         self.input_dim = args.input_feat_per_channel
#         self.conv = torch.nn.Sequential(
#             torch.nn.Conv2d(1, args.conv_out_channels, 3, stride=2, padding=3 // 2),
#             torch.nn.ReLU(),
#             torch.nn.Conv2d(
#                 args.conv_out_channels,
#                 args.conv_out_channels,
#                 3,
#                 stride=2,
#                 padding=3 // 2,
#             ),
#             torch.nn.ReLU(),
#         )
#         transformer_input_dim = self.infer_conv_output_dim(
#             self.in_channels, self.input_dim, args.conv_out_channels
#         )
#         self.out = torch.nn.Linear(transformer_input_dim, args.encoder_embed_dim)
#         self.embed_positions = PositionalEmbedding(
#             args.max_source_positions,
#             args.encoder_embed_dim,
#             self.padding_idx,
#             learned=False,
#         )
#
#         self.transformer_layers = nn.ModuleList([])
#         self.transformer_layers.extend(
#             [TransformerEncoderLayer(args) for i in range(args.encoder_layers)]
#         )
#         if args.encoder_normalize_before:
#             self.layer_norm = LayerNorm(args.encoder_embed_dim)
#         else:
#             self.layer_norm = None
#
#     def pooling_ratio(self):
#         return 4
#
#     def infer_conv_output_dim(self, in_channels, input_dim, out_channels):
#         sample_seq_len = 200
#         sample_bsz = 10
#         x = torch.randn(sample_bsz, in_channels, sample_seq_len, input_dim)
#         x = torch.nn.Conv2d(1, out_channels, 3, stride=2, padding=3 // 2)(x)
#         x = torch.nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=3 // 2)(x)
#         x = x.transpose(1, 2)
#         mb, seq = x.size()[:2]
#         return x.contiguous().view(mb, seq, -1).size(-1)
#
#     def forward(self, src_tokens, src_lengths):
#         """Encode input sequence.
#         :param torch.Tensor xs: input tensor
#         :param torch.Tensor masks: input mask
#         :return: position embedded tensor and mask
#         :rtype Tuple[torch.Tensor, torch.Tensor]:
#         """
#         bsz, max_seq_len, _ = src_tokens.size()
#         x = (
#             src_tokens.view(bsz, max_seq_len, self.in_channels, self.input_dim)
#             .transpose(1, 2)
#             .contiguous()
#         )
#         x = self.conv(x)
#         bsz, _, output_seq_len, _ = x.size()
#         x = x.transpose(1, 2).transpose(0, 1).contiguous().view(output_seq_len, bsz, -1)
#         x = self.out(x)
#         x = self.embed_scale * x
#
#         subsampling_factor = int(max_seq_len * 1.0 / output_seq_len + 0.5)
#         input_len_0 = (src_lengths.float() / subsampling_factor).ceil().long()
#         input_len_1 = x.size(0) * torch.ones([src_lengths.size(0)]).long().to(input_len_0.device)
#         input_lengths = torch.min(
#             input_len_0, input_len_1
#         )
#
#         encoder_padding_mask = lengths_to_padding_mask(input_lengths)
#
#         positions = self.embed_positions(encoder_padding_mask).transpose(0, 1)
#         x += positions
#         x = F.dropout(x, p=self.dropout, training=self.training)
#
#         for layer in self.transformer_layers:
#             x = layer(x, encoder_padding_mask)
#
#         # if not encoder_padding_mask.any():
#         #     maybe_encoder_padding_mask = None
#         # else:
#         #     maybe_encoder_padding_mask = encoder_padding_mask
#         maybe_encoder_padding_mask = encoder_padding_mask
#
#         return {
#             "encoder_out": [x],
#             "encoder_padding_mask": [maybe_encoder_padding_mask]
#             if maybe_encoder_padding_mask is not None
#             else [],
#             "encoder_embedding": [],
#             "encoder_states": [],
#             "src_tokens": [],
#             "src_lengths": [],
#         }
#
#     @torch.jit.export
#     def reorder_encoder_out(self, encoder_out: Dict[str, List[Tensor]], new_order):
#         """
#         Reorder encoder output according to *new_order*.
#
#         Args:
#             encoder_out: output from the ``forward()`` method
#             new_order (LongTensor): desired order
#
#         Returns:
#             *encoder_out* rearranged according to *new_order*
#         """
#         new_encoder_out = [encoder_out["encoder_out"][0].index_select(1, new_order)]
#         if len(encoder_out["encoder_padding_mask"]) == 0:
#             new_encoder_padding_mask = []
#         else:
#             new_encoder_padding_mask = [
#                 (encoder_out["encoder_padding_mask"][0]).index_select(0, new_order)
#             ]
#         if len(encoder_out["encoder_embedding"]) == 0:
#             new_encoder_embedding = []
#         else:
#             new_encoder_embedding = [
#                 (encoder_out["encoder_embedding"][0]).index_select(0, new_order)
#             ]
#         encoder_states = encoder_out["encoder_states"]
#         if len(encoder_states) > 0:
#             for idx, state in enumerate(encoder_states):
#                 encoder_states[idx] = state.index_select(1, new_order)
#
#         return {
#             "encoder_out": new_encoder_out,
#             "encoder_padding_mask": new_encoder_padding_mask,
#             "encoder_embedding": new_encoder_embedding,
#             "encoder_states": encoder_states,
#             "src_tokens": [],
#             "src_lengths": [],
#         }

#
# class TransformerDecoderNoExtra(TransformerDecoder):
#     def extract_features(
#         self,
#         prev_output_tokens,
#         encoder_out: Optional[Dict[str, List[Tensor]]],
#         incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
#         full_context_alignment: bool = False,
#         alignment_layer: Optional[int] = None,
#         alignment_heads: Optional[int] = None,
#     ):
#         # call scriptable method from parent class
#         x, _ = self.extract_features_scriptable(
#             prev_output_tokens,
#             encoder_out,
#             incremental_state,
#             full_context_alignment,
#             alignment_layer,
#             alignment_heads,
#         )
#         return x, None


@register_model_architecture(model_name="convtransformer_cif", arch_name="convtransformer_cif")
def base_architecture(args):
    args.input_feat_per_channel = getattr(args, "input_feat_per_channel", 80)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.max_source_positions = getattr(args, "max_source_positions", 3000)
    args.max_target_positions = getattr(args, "max_target_positions", 1024)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)
    args.conv_out_channels = getattr(args, "conv_out_channels", args.encoder_embed_dim)


@register_model_architecture(model_name="convtransformer_cif", arch_name="convtransformer_cif_espnet")
def convtransformer_cif_espnet(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)


@register_model_architecture("convtransformer_cif", "convtransformer_cif_espnet_add")
def convtransformer_espnet_cif(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.lambda_qua = getattr(args, "lambda_qua", 0.05)
    args.lambda_alpha = getattr(args, "lambda_alpha", 0.1)