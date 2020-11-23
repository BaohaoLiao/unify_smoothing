# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import namedtuple
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq import checkpoint_utils
from fairseq.models import (
    FairseqEncoder,
    FairseqIncrementalDecoder,
    FairseqEncoderDecoderModel,
    FairseqDA,
    register_model,
    register_model_architecture,
)
from fairseq.modules import (
    AdaptiveSoftmax,
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerDecoderLayer,
    TransformerEncoderLayer,
)
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder, TransformerModel
from fairseq.models.masked_lm import MaskedLMEncoder
import random

DEFAULT_MAX_SOURCE_POSITIONS = 1024
DEFAULT_MAX_TARGET_POSITIONS = 1024


@register_model('transformer_da')
class TransformerDAModel(FairseqDA):
    """
    Transformer model from `"Attention Is All You Need" (Vaswani, et al, 2017)
    <https://arxiv.org/abs/1706.03762>`_.

    Args:
        encoder (TransformerEncoder): the encoder
        decoder (TransformerDecoder): the decoder

    The Transformer model provides the following named architectures and
    command-line arguments:

    .. argparse::
        :ref: fairseq.models.transformer_parser
        :prog:
    """

    @classmethod
    def hub_models(cls):
        # fmt: off

        def moses_subword(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'subword_nmt',
            }

        def moses_fastbpe(path):
            return {
                'path': path,
                'tokenizer': 'moses',
                'bpe': 'fastbpe',
            }

        return {
            'transformer.wmt14.en-fr': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2'),
            'transformer.wmt16.en-de': 'https://dl.fbaipublicfiles.com/fairseq/models/wmt16.en-de.joined-dict.transformer.tar.bz2',
            'transformer.wmt18.en-de': moses_subword('https://dl.fbaipublicfiles.com/fairseq/models/wmt18.en-de.ensemble.tar.gz'),
            'transformer.wmt19.en-de': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.en-ru': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.ensemble.tar.gz'),
            'transformer.wmt19.de-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.ensemble.tar.gz'),
            'transformer.wmt19.ru-en': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.ensemble.tar.gz'),
            'transformer.wmt19.en-de.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-de.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.en-ru.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.en-ru.single_model.tar.gz'),
            'transformer.wmt19.de-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.de-en.joined-dict.single_model.tar.gz'),
            'transformer.wmt19.ru-en.single_model': moses_fastbpe('https://dl.fbaipublicfiles.com/fairseq/models/wmt19.ru-en.single_model.tar.gz'),
        }
        # fmt: on

    def __init__(self, args, nmtencoder, nmtdecoder, srcdamodel, tgtdamodel):
        super().__init__(nmtencoder, nmtdecoder, srcdamodel, tgtdamodel)
        self.args = args
        self.supports_align_args = True

        """
        if self.training and args.srcda:
            if args.srcda_choice == 'lm':
                checkpoint_utils.load_lm_state(srcdamodel, args.srcda_file)
            elif args.srcda_choice == 'bert':
                checkpoint_utils.load_bert_state(srcdamodel, args.srcda_file)
            elif args.srcda_choice == 'nmt':
                checkpoint_utils.load_nmt_state(srcdamodel, args.srcda_file)
        if self.training and args.tgtda:
            if args.tgtda_choice == 'lm':
                checkpoint_utils.load_lm_state(tgtdamodel, args.tgtda_file)
            elif args.tgtda_choice == 'bert':
                checkpoint_utils.load_bert_state(tgtdamodel, args.tgtda_file)
        """

        if srcdamodel is not None:
            for param in srcdamodel.parameters():
                param.requires_grad = False
        if tgtdamodel is not None:
            for param in tgtdamodel.parameters():
                param.requires_grad = False

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--attention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--activation-dropout', '--relu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--encoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--encoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--encoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--decoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--decoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--decoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--decoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--decoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--share-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--share-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--no-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--adaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--adaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--no-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--cross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--layer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--encoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--decoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--encoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--decoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--layernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--no-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        ## LM
        parser.add_argument('--lmactivation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--lmdropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--lmattention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--lmactivation-dropout', '--lmrelu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--lmdecoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--lmdecoder-output-dim', type=int, metavar='N',
                            help='decoder output dimension')
        parser.add_argument('--lmdecoder-input-dim', type=int, metavar='N',
                            help='decoder input dimension')
        parser.add_argument('--lmdecoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--lmdecoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--lmdecoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--lmdecoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--lmno-decoder-final-norm', action='store_true',
                            help='don\'t add an extra layernorm after the last decoder block')
        parser.add_argument('--lmadaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion')
        parser.add_argument('--lmadaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        parser.add_argument('--lmadaptive-softmax-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--lmno-token-positional-embeddings', action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--lmshare-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--lmcharacter-embeddings', action='store_true',
                            help='if set, uses character embedding convolutions to produce token embeddings')
        parser.add_argument('--lmcharacter-filters', type=str, metavar='LIST',
                            default='[(1, 64), (2, 128), (3, 192), (4, 256), (5, 256), (6, 256), (7, 256)]',
                            help='size of character embeddings')
        parser.add_argument('--lmcharacter-embedding-dim', default=4, type=int, metavar='N',
                            help='size of character embeddings')
        parser.add_argument('--lmchar-embedder-highway-layers', default=2, type=int, metavar='N',
                            help='number of highway layers for character token embeddder')
        parser.add_argument('--lmadaptive-input', action='store_true',
                            help='if set, uses adaptive input')
        parser.add_argument('--lmadaptive-input-factor', type=float, metavar='N',
                            help='adaptive input factor')
        parser.add_argument('--lmadaptive-input-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive input cutoff points.')
        parser.add_argument('--lmtie-adaptive-weights', action='store_true',
                            help='if set, ties the weights of adaptive softmax and adaptive input')
        parser.add_argument('--lmtie-adaptive-proj', action='store_true',
                            help='if set, ties the projection weights of adaptive softmax and adaptive input')
        parser.add_argument('--lmdecoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--lmdecoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--lmdecoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--lmlayernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--lmno-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')

        ## BERT
        parser.add_argument('--bertdropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--bertattention-dropout', type=float,
                            metavar='D', help='dropout probability for'
                            ' attention weights')
        parser.add_argument('--bertact-dropout', type=float,
                            metavar='D', help='dropout probability after'
                            ' activation in FFN')

        # Arguments related to hidden states and self-attention
        parser.add_argument('--bertencoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--bertencoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--bertencoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--bertbias-kv', action='store_true',
                            help='if set, adding a learnable bias kv')
        parser.add_argument('--bertzero-attn', action='store_true',
                            help='if set, pads attn with zero')

        # Arguments related to input and output embeddings
        parser.add_argument('--bertencoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--bertshare-encoder-input-output-embed',
                            action='store_true', help='share encoder input'
                            ' and output embeddings')
        parser.add_argument('--bertencoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--bertno-token-positional-embeddings',
                            action='store_true',
                            help='if set, disables positional embeddings'
                            ' (outside self attention)')
        parser.add_argument('--bertnum-segment', type=int, metavar='N',
                            help='num segment in the input')

        # Arguments related to sentence level prediction
        parser.add_argument('--bertsentence-class-num', type=int, metavar='N',
                            help='number of classes for sentence task')
        parser.add_argument('--bertsent-loss', action='store_true', help='if set,'
                            ' calculate sentence level predictions')

        # Arguments related to parameter initialization
        parser.add_argument('--bertapply-bert-init', action='store_true',
                            help='use custom param initialization for BERT')

        # misc params
        parser.add_argument('--bertactivation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--bertpooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='Which activation function to use for pooler layer.')
        parser.add_argument('--bertencoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--bertmax-positions', type=int, default=4098)

        ## NMT
        parser.add_argument('--nmtactivation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use')
        parser.add_argument('--nmtdropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--nmtattention-dropout', type=float, metavar='D',
                            help='dropout probability for attention weights')
        parser.add_argument('--nmtactivation-dropout', '--nmtrelu-dropout', type=float, metavar='D',
                            help='dropout probability after activation in FFN.')
        parser.add_argument('--nmtencoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--nmtencoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--nmtencoder-ffn-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension for FFN')
        parser.add_argument('--nmtencoder-layers', type=int, metavar='N',
                            help='num encoder layers')
        parser.add_argument('--nmtencoder-attention-heads', type=int, metavar='N',
                            help='num encoder attention heads')
        parser.add_argument('--nmtencoder-normalize-before', action='store_true',
                            help='apply layernorm before each encoder block')
        parser.add_argument('--nmtencoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the encoder')
        parser.add_argument('--nmtdecoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained decoder embedding')
        parser.add_argument('--nmtdecoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--nmtdecoder-ffn-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension for FFN')
        parser.add_argument('--nmtdecoder-layers', type=int, metavar='N',
                            help='num decoder layers')
        parser.add_argument('--nmtdecoder-attention-heads', type=int, metavar='N',
                            help='num decoder attention heads')
        parser.add_argument('--nmtdecoder-learned-pos', action='store_true',
                            help='use learned positional embeddings in the decoder')
        parser.add_argument('--nmtdecoder-normalize-before', action='store_true',
                            help='apply layernorm before each decoder block')
        parser.add_argument('--nmtshare-decoder-input-output-embed', action='store_true',
                            help='share decoder input and output embeddings')
        parser.add_argument('--nmtshare-all-embeddings', action='store_true',
                            help='share encoder, decoder and output embeddings'
                                 ' (requires shared dictionary and embed dim)')
        parser.add_argument('--nmtno-token-positional-embeddings', default=False, action='store_true',
                            help='if set, disables positional embeddings (outside self attention)')
        parser.add_argument('--nmtadaptive-softmax-cutoff', metavar='EXPR',
                            help='comma separated list of adaptive softmax cutoff points. '
                                 'Must be used with adaptive_loss criterion'),
        parser.add_argument('--nmtadaptive-softmax-dropout', type=float, metavar='D',
                            help='sets adaptive softmax dropout for the tail projections')
        # args for "Cross+Self-Attention for Transformer Models" (Peitz et al., 2019)
        parser.add_argument('--nmtno-cross-attention', default=False, action='store_true',
                            help='do not perform cross-attention')
        parser.add_argument('--nmtcross-self-attention', default=False, action='store_true',
                            help='perform cross+self-attention')
        parser.add_argument('--nmtlayer-wise-attention', default=False, action='store_true',
                            help='perform layer-wise attention (cross-attention or cross+self-attention)')
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument('--nmtencoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for encoder')
        parser.add_argument('--nmtdecoder-layerdrop', type=float, metavar='D', default=0,
                            help='LayerDrop probability for decoder')
        parser.add_argument('--nmtencoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--nmtdecoder-layers-to-keep', default=None,
                            help='which layers to *keep* when pruning as a comma-separated list')
        parser.add_argument('--nmtlayernorm-embedding', action='store_true',
                            help='add layernorm to embedding')
        parser.add_argument('--nmtno-scale-embedding', action='store_true',
                            help='if True, dont scale embeddings')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present in older models
        base_architecture(args)

        # NMT model
        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, 'max_source_positions', None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, 'max_target_positions', None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        def build_embedding(dictionary, embed_dim, path=None):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            emb = Embedding(num_embeddings, embed_dim, padding_idx)
            # if provided, load from preloaded dictionaries
            if path:
                embed_dict = utils.parse_embedding(path)
                utils.load_embedding(embed_dict, dictionary, emb)
            return emb

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError('--share-all-embeddings requires a joined dictionary')
            if args.encoder_embed_dim != args.decoder_embed_dim:
                raise ValueError(
                    '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
            if args.decoder_embed_path and (
                    args.decoder_embed_path != args.encoder_embed_path):
                raise ValueError('--share-all-embeddings not compatible with --decoder-embed-path')
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)


        # Data augmentation
        def prepare_da_args(args, prefix):
            tgtargs = daargs()
            for k, v in args.__dict__.items():
                if k.startswith(prefix):
                    tgtargs.__setattr__(k[len(prefix):], v)
            return tgtargs

        srcdamodel = None
        tgtdamodel = None
        if args.srcda:
            if args.srcda_choice == 'lm':
                if getattr(args, "lmmax_source_positions", None) is None:
                    args.lmmax_target_positions = DEFAULT_MAX_SOURCE_POSITIONS

                if args.lmadaptive_input:
                    src_embed_tokens = AdaptiveInput(
                        len(task.source_dictionary), task.source_dictionary.pad(), args.lmdecoder_input_dim,
                        args.lmadaptive_input_factor, args.lmdecoder_embed_dim,
                        options.eval_str_list(args.lmadaptive_input_cutoff, type=int),
                    )
                else:
                    src_embed_tokens = Embedding(len(task.source_dictionary), args.lmdecoder_input_dim,
                                             task.source_dictionary.pad())

                if args.lmtie_adaptive_weights:
                    assert args.lmadaptive_input
                    assert args.lmadaptive_input_factor == args.lmadaptive_softmax_factor
                    assert args.lmadaptive_softmax_cutoff == args.lmadaptive_input_cutoff, '{} != {}'.format(
                        args.lmadaptive_softmax_cutoff, args.lmadaptive_input_cutoff)
                    assert args.lmdecoder_input_dim == args.lmdecoder_output_dim

                newdaargs = prepare_da_args(args, 'lm')
                srcdamodel = TransformerDecoder(newdaargs, src_dict, src_embed_tokens, no_encoder_attn=True)
                #if os.path.isfile(args.srcda_file):
                #checkpoint_utils.load_lm_state(srcdamodel, args.srcda_file)
            elif args.srcda_choice == 'bert':
                newdaargs = prepare_da_args(args, 'bert')
                if not hasattr(newdaargs, 'max_positions'):
                    newdaargs.max_positions = DEFAULT_MAX_SOURCE_POSITIONS
                srcdamodel = MaskedLMEncoder(newdaargs, src_dict)
                #checkpoint_utils.load_bert_state(srcdamodel, args.srcda_file)
            elif args.srcda_choice == 'nmt':
                newdaargs = prepare_da_args(args, 'nmt')

                if getattr(newdaargs, "max_source_positions", None) is None:
                    newdaargs.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
                if getattr(newdaargs, "max_target_positions", None) is None:
                    newdaargs.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

                if args.nmtshare_all_embeddings:
                    if src_dict != tgt_dict:
                        raise ValueError('--share-all-embeddings requires a joined dictionary')
                    if args.nmtencoder_embed_dim != args.nmtdecoder_embed_dim:
                        raise ValueError(
                            '--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim')
                    srcda_encoder_embed_tokens = Embedding(len(tgt_dict), args.nmtencoder_embed_dim, tgt_dict.pad())
                    srcda_decoder_embed_tokens = srcda_encoder_embed_tokens
                    newdaargs.share_decoder_input_output_embed = True
                else:
                    srcda_encoder_embed_tokens = Embedding(len(tgt_dict), args.nmtencoder_embed_dim, tgt_dict.pad())
                    srcda_decoder_embed_tokens = Embedding(len(src_dict), args.nmtdecoder_embed_dim, src_dict.pad())

                srcda_encoder = TransformerEncoder(newdaargs, tgt_dict, srcda_encoder_embed_tokens)
                srcda_decoder = TransformerDecoder(newdaargs, src_dict, srcda_decoder_embed_tokens)
                srcdamodel = TransformerModel(newdaargs, srcda_encoder, srcda_decoder)
                #checkpoint_utils.load_nmt_state(srcdamodel, args.srcda_file)

        if args.tgtda:
            if args.tgtda_choice == 'lm':
                if getattr(args, "lmmax_source_positions", None) is None:
                    args.lmmax_target_positions = DEFAULT_MAX_SOURCE_POSITIONS

                if args.lmadaptive_input:
                    tgt_embed_tokens = AdaptiveInput(
                        len(task.target_dictionary), task.target_dictionary.pad(), args.lmdecoder_input_dim,
                        args.lmadaptive_input_factor, args.lmdecoder_embed_dim,
                        options.eval_str_list(args.lmadaptive_input_cutoff, type=int),
                    )
                else:
                    tgt_embed_tokens = Embedding(len(task.target_dictionary), args.lmdecoder_input_dim,
                                             task.target_dictionary.pad())

                if args.lmtie_adaptive_weights:
                    assert args.lmadaptive_input
                    assert args.lmadaptive_input_factor == args.lmadaptive_softmax_factor
                    assert args.lmadaptive_softmax_cutoff == args.lmadaptive_input_cutoff, '{} != {}'.format(
                        args.lmadaptive_softmax_cutoff, args.lmadaptive_input_cutoff)
                    assert args.lmdecoder_input_dim == args.lmdecoder_output_dim

                newdaargs = prepare_da_args(args, 'lm')
                tgtdamodel = TransformerDecoder(newdaargs, task.target_dictionary, tgt_embed_tokens, no_encoder_attn=True)
                #if os.path.isfile(args.tgtda_file):
                #checkpoint_utils.load_lm_state(tgtdamodel, args.tgtda_file)
            elif args.srcda_choice == 'bert':
                newdaargs = prepare_da_args(args, 'bert')
                if not hasattr(newdaargs, 'max_positions'):
                    newdaargs.max_positions = DEFAULT_MAX_SOURCE_POSITIONS
                tgtdamodel = MaskedLMEncoder(newdaargs, tgt_dict)
                #checkpoint_utils.load_bert_state(tgtdamodel, args.tgtda_file)

        return cls(args, encoder, decoder, srcdamodel, tgtdamodel)

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerEncoderModified(args, src_dict, embed_tokens)

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerDecoderModified(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, 'no_cross_attention', False),
        )

    def forward(self, src_tokens, src_lengths, src_tokens_da, tgt_tokens, tgt_lengths, 
                prev_output_tokens, prev_output_tokens_da, **kwargs):
        srcdaoutput = None
        if self.args.srcda_choice == 'lm':
            srcdaoutput, _ = self.srcdamodel(src_tokens_da)
        elif self.args.srcda_choice == 'bert':
            srcdaoutput, _ = self.srcdamodel(src_tokens)
        elif self.args.srcda_choice == 'nmt':
            srcdaoutput, _ = self.srcdamodel(tgt_tokens, tgt_lengths, src_tokens_da)

        tgtdaoutput = None
        if self.args.tgtda_choice == 'lm':
            tgtdaoutput, _ = self.tgtdamodel(prev_output_tokens_da)
        elif self.args.tgtda_choice == 'bert':
            tgtdaoutput, _ = self.tgtdamodel(prev_output_tokens)

        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, srcdaoutput=srcdaoutput, **kwargs)
        decoder_out = self.decoder(prev_output_tokens, encoder_out=encoder_out, tgtdaoutput=tgtdaoutput, **kwargs)
        return decoder_out


class daargs(object):
    def __init__(self):
        pass


EncoderOut = namedtuple('TransformerEncoderOut', [
    'encoder_out',  # T x B x C
    'encoder_padding_mask',  # B x T
    'encoder_embedding',  # B x T x C
    'encoder_states',  # List[T x B x C]
])


class TransformerEncoderModified(FairseqEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        self.dictionary = dictionary
        self.srcda = args.srcda
        self.srcda_choice = args.srcda_choice
        self.srcda_percentage = args.srcda_percentage
        self.srcda_smooth = args.srcda_smooth
        self.select_choice = args.select_choice


    def forward_embedding(self, src_tokens, srcdaoutput):
        # embed tokens and positions
        if self.srcda and self.training:
            # one hot vector
            x_onehot = torch.FloatTensor(src_tokens.size(0), src_tokens.size(1),
                           len(self.dictionary)).to(src_tokens.device)
            x_onehot.scatter_(2, src_tokens.unsqueeze(-1), 1.)

            # choose which tokens to augment
            if self.select_choice == 'uniform':
                random_select = x_onehot.new_empty(x_onehot.shape[:2]).uniform_(0, 1).unsqueeze(-1)
            onehot_select = random_select.ge(self.srcda_percentage)
            smooth_select = ~onehot_select

            # augment distribution
            if self.srcda_choice == 'uniform':
                smooth_dis = torch.ones(x_onehot.shape).to(x_onehot.device) * (1.0 / len(self.dictionary))
            elif self.srcda_choice == 'unigram':
                unigram = torch.FloatTensor(self.dictionary.count).to(x_onehot.device) / sum(self.dictionary.count)
                smooth_dis = torch.ones(src_tokens.size(0), src_tokens.size(1), 1).to(x_onehot.device) * unigram
            elif self.srcda_choice == 'lm' or 'bert' or 'nmt':
                smooth_dis = F.softmax(srcdaoutput, dim=-1)

            # smooth
            #eps_i = self.srcda_smooth / len(self.dictionary)
            smooth_tokens = (1. - self.srcda_smooth) * x_onehot + self.srcda_smooth * smooth_dis

            # final token representation
            x = x_onehot.masked_fill(smooth_select, 0.) + smooth_tokens.masked_fill(onehot_select, 0.)
            x = F.linear(x, self.embed_tokens.weight.t())
            x = embed = self.embed_scale * x
        else:
            x = embed = self.embed_scale * self.embed_tokens(src_tokens)

        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, src_lengths, srcdaoutput=None, 
                cls_input=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens, srcdaoutput)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                print('deleting {0}'.format(weights_key))
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict


class TransformerDecoderModified(FairseqIncrementalDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.decoder_layerdrop = args.decoder_layerdrop
        self.share_input_output_embed = args.share_decoder_input_output_embed

        input_embed_dim = embed_tokens.embedding_dim
        embed_dim = args.decoder_embed_dim
        self.output_embed_dim = args.decoder_output_dim

        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = args.max_target_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.project_in_dim = Linear(input_embed_dim, embed_dim, bias=False) if embed_dim != input_embed_dim else None

        self.embed_positions = PositionalEmbedding(
            args.max_target_positions, embed_dim, self.padding_idx,
            learned=args.decoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.cross_self_attention = getattr(args, 'cross_self_attention', False)
        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerDecoderLayer(args, no_encoder_attn)
            for _ in range(args.decoder_layers)
        ])

        self.adaptive_softmax = None

        self.project_out_dim = Linear(embed_dim, self.output_embed_dim, bias=False) \
            if embed_dim != self.output_embed_dim and not args.tie_adaptive_weights else None

        if args.adaptive_softmax_cutoff is not None:
            self.adaptive_softmax = AdaptiveSoftmax(
                len(dictionary),
                self.output_embed_dim,
                options.eval_str_list(args.adaptive_softmax_cutoff, type=int),
                dropout=args.adaptive_softmax_dropout,
                adaptive_inputs=embed_tokens if args.tie_adaptive_weights else None,
                factor=args.adaptive_softmax_factor,
                tie_proj=args.tie_adaptive_proj,
            )
        elif not self.share_input_output_embed:
            self.embed_out = nn.Parameter(torch.Tensor(len(dictionary), self.output_embed_dim))
            nn.init.normal_(self.embed_out, mean=0, std=self.output_embed_dim ** -0.5)

        if args.decoder_normalize_before and not getattr(args, 'no_decoder_final_norm', False):
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

        # data augmentation
        self.dictionary = dictionary
        self.tgtda = args.tgtda
        self.tgtda_choice = args.tgtda_choice
        self.tgtda_percentage = args.tgtda_percentage
        self.tgtda_smooth = args.tgtda_smooth
        self.select_choice = args.select_choice

    def forward(
        self,
        prev_output_tokens,
        tgtdaoutput=None,
        encoder_out=None,
        incremental_state=None,
        features_only=False,
        **extra_args
    ):
        """
        Args:
            prev_output_tokens (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_out (optional): output from the encoder, used for
                encoder-side attention
            incremental_state (dict): dictionary used for storing state during
                :ref:`Incremental decoding`
            features_only (bool, optional): only return features without
                applying output layer (default: False).

        Returns:
            tuple:
                - the decoder's output of shape `(batch, tgt_len, vocab)`
                - a dictionary with any model-specific outputs
        """
        x, extra = self.extract_features(
            prev_output_tokens,
            tgtdaoutput=tgtdaoutput,
            encoder_out=encoder_out,
            incremental_state=incremental_state,
            **extra_args
        )
        if not features_only:
            x = self.output_layer(x)
        return x, extra

    def extract_features(
        self,
        prev_output_tokens,
        tgtdaoutput=None,
        encoder_out=None,
        incremental_state=None,
        full_context_alignment=False,
        alignment_layer=None,
        alignment_heads=None,
        **unused,
    ):
        """
        Similar to *forward* but only return features.

        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = len(self.layers) - 1

        # embed positions
        positions = self.embed_positions(
            prev_output_tokens,
            incremental_state=incremental_state,
        ) if self.embed_positions is not None else None

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        if self.tgtda and self.training:
            # one hot vector
            x_onehot = torch.FloatTensor(prev_output_tokens.size(0), prev_output_tokens.size(1),
                           len(self.dictionary)).to(prev_output_tokens.device)
            x_onehot.scatter_(2, prev_output_tokens.unsqueeze(-1), 1.)

            # choose which tokens to augment
            if self.select_choice == 'uniform':
                random_select = x_onehot.new_empty(x_onehot.shape[:2]).uniform_(0, 1).unsqueeze(-1)
            onehot_select = random_select.ge(self.tgtda_percentage)
            smooth_select = ~onehot_select

            # augmentation method
            if self.tgtda_choice == 'uniform':
                smooth_dis = torch.ones(x_onehot.shape).to(x_onehot.device) * (1.0 / len(self.dictionary))
            elif self.tgtda_choice == 'unigram':
                unigram = torch.FloatTensor(self.dictionary.count).to(x_onehot.device) / sum(self.dictionary.count)
                smooth_dis = torch.ones(prev_output_tokens.size(0), prev_output_tokens.size(1), 1).to(x_onehot.device) * unigram
            elif self.tgtda_choice == 'lm' or 'bert':
                smooth_dis = F.softmax(tgtdaoutput, dim=-1)

            # smooth
            #eps_i = self.tgtda_smooth / len(self.dictionary)
            smooth_tokens = (1. - self.tgtda_smooth) * x_onehot + self.tgtda_smooth * smooth_dis

            # final token representation
            x = x_onehot.masked_fill(smooth_select, 0.) + smooth_tokens.masked_fill(onehot_select, 0.)

            x = F.linear(x, self.embed_tokens.weight.t())
            x = self.embed_scale * x
        else:
            x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        # decoder layers
        attn = None
        inner_states = [x]
        for idx, layer in enumerate(self.layers):
            encoder_state = None
            if encoder_out is not None:
                if self.layer_wise_attention:
                    encoder_state = encoder_out.encoder_states[idx]
                else:
                    encoder_state = encoder_out.encoder_out

            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None

            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.decoder_layerdrop):
                x, layer_attn = layer(
                    x,
                    encoder_state,
                    encoder_out.encoder_padding_mask if encoder_out is not None else None,
                    incremental_state,
                    self_attn_mask=self_attn_mask,
                    self_attn_padding_mask=self_attn_padding_mask,
                    need_attn=(idx == alignment_layer),
                    need_head_weights=(idx == alignment_layer),
                )
                inner_states.append(x)
                if layer_attn is not None and idx == alignment_layer:
                    attn = layer_attn.float()

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]

            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {'attn': attn, 'inner_states': inner_states}

    def output_layer(self, features, **kwargs):
        """Project features to the vocabulary size."""
        if self.adaptive_softmax is None:
            # project back to size of vocabulary
            if self.share_input_output_embed:
                return F.linear(features, self.embed_tokens.weight)
            else:
                return F.linear(features, self.embed_out)
        else:
            return features

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        if self.embed_positions is None:
            return self.max_target_positions
        return min(self.max_target_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if (
            not hasattr(self, '_future_mask')
            or self._future_mask is None
            or self._future_mask.device != tensor.device
            or self._future_mask.size(0) < dim
        ):
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)

        for i in range(len(self.layers)):
            # update layer norms
            layer_norm_map = {
                '0': 'self_attn_layer_norm',
                '1': 'encoder_attn_layer_norm',
                '2': 'final_layer_norm'
            }
            for old, new in layer_norm_map.items():
                for m in ('weight', 'bias'):
                    k = '{}.layers.{}.layer_norms.{}.{}'.format(name, i, old, m)
                    if k in state_dict:
                        state_dict['{}.layers.{}.{}.{}'.format(name, i, new, m)] = state_dict[k]
                        del state_dict[k]

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) <= 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])

        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


@register_model_architecture('transformer_da', 'transformer_da')
def base_architecture(args):
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 2048)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 8)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.encoder_learned_pos = getattr(args, 'encoder_learned_pos', False)
    args.decoder_embed_path = getattr(args, 'decoder_embed_path', None)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', args.encoder_ffn_embed_dim)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 8)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', False)
    args.decoder_learned_pos = getattr(args, 'decoder_learned_pos', False)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.)
    args.activation_fn = getattr(args, 'activation_fn', 'relu')
    args.dropout = getattr(args, 'dropout', 0.1)
    args.adaptive_softmax_cutoff = getattr(args, 'adaptive_softmax_cutoff', None)
    args.adaptive_softmax_dropout = getattr(args, 'adaptive_softmax_dropout', 0)
    args.share_decoder_input_output_embed = getattr(args, 'share_decoder_input_output_embed', False)
    args.share_all_embeddings = getattr(args, 'share_all_embeddings', False)
    args.no_token_positional_embeddings = getattr(args, 'no_token_positional_embeddings', False)
    args.adaptive_input = getattr(args, 'adaptive_input', False)
    args.no_cross_attention = getattr(args, 'no_cross_attention', False)
    args.cross_self_attention = getattr(args, 'cross_self_attention', False)
    args.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

    args.decoder_output_dim = getattr(args, 'decoder_output_dim', args.decoder_embed_dim)
    args.decoder_input_dim = getattr(args, 'decoder_input_dim', args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, 'no_scale_embedding', False)
    args.layernorm_embedding = getattr(args, 'layernorm_embedding', False)


@register_model_architecture('transformer_da', 'transformer_da_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)
    base_architecture(args)


@register_model_architecture('transformer_da', 'transformer_da_wmt_en_de')
def transformer_wmt_en_de(args):
    base_architecture(args)


# parameters used in the "Attention Is All You Need" paper (Vaswani et al., 2017)
@register_model_architecture('transformer_da', 'transformer_da_vaswani_wmt_en_de_big')
def transformer_vaswani_wmt_en_de_big(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 1024)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 4096)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 16)
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', False)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 1024)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 4096)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 16)
    args.dropout = getattr(args, 'dropout', 0.3)
    base_architecture(args)


@register_model_architecture('transformer_da', 'transformer_da_vaswani_wmt_en_fr_big')
def transformer_vaswani_wmt_en_fr_big(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


@register_model_architecture('transformer_da', 'transformer_da_wmt_en_de_big')
def transformer_wmt_en_de_big(args):
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)


# default parameters used in tensor2tensor implementation
@register_model_architecture('transformer_da', 'transformer_da_wmt_en_de_big_t2t')
def transformer_wmt_en_de_big_t2t(args):
    args.encoder_normalize_before = getattr(args, 'encoder_normalize_before', True)
    args.decoder_normalize_before = getattr(args, 'decoder_normalize_before', True)
    args.attention_dropout = getattr(args, 'attention_dropout', 0.1)
    args.activation_dropout = getattr(args, 'activation_dropout', 0.1)
    transformer_vaswani_wmt_en_de_big(args)

@register_model_architecture('transformer_da', 'transformer_da_lm')
def base_da_lm_atchitecture(args):
    #lm
    args.lmdropout = getattr(args, 'lmdropout', 0.1)
    args.lmattention_dropout = getattr(args, 'lmattention_dropout', 0.0)

    args.lmdecoder_embed_dim = getattr(args, 'lmdecoder_embed_dim', 512)
    args.lmdecoder_ffn_embed_dim = getattr(args, 'lmdecoder_ffn_embed_dim', 2048)
    args.lmdecoder_layers = getattr(args, 'lmdecoder_layers', 6)
    args.lmdecoder_attention_heads = getattr(args, 'lmdecoder_attention_heads', 8)
    args.lmadaptive_softmax_cutoff = getattr(args, 'lmadaptive_softmax_cutoff', None)
    args.lmadaptive_softmax_dropout = getattr(args, 'lmadaptive_softmax_dropout', 0)
    args.lmadaptive_softmax_factor = getattr(args, 'lmadaptive_softmax_factor', 4)
    args.lmdecoder_learned_pos = getattr(args, 'lmdecoder_learned_pos', False)
    args.lmactivation_fn = getattr(args, 'lmactivation_fn', 'relu')

    args.lmadd_bos_token = getattr(args, 'lmadd_bos_token', False)
    args.lmno_token_positional_embeddings = getattr(args, 'lmno_token_positional_embeddings', False)
    args.lmshare_decoder_input_output_embed = getattr(args, 'dashare_decoder_input_output_embed', False)
    args.lmcharacter_embeddings = getattr(args, 'lmcharacter_embeddings', False)

    args.lmdecoder_output_dim = getattr(args, 'lmdecoder_output_dim', args.lmdecoder_embed_dim)
    args.lmdecoder_input_dim = getattr(args, 'lmdecoder_input_dim', args.lmdecoder_embed_dim)

    # Model training is not stable without this
    args.lmdecoder_normalize_before = True
    args.lmno_decoder_final_norm = getattr(args, 'lmno_decoder_final_norm', False)

    args.lmadaptive_input = getattr(args, 'daadaptive_input', False)
    args.lmadaptive_input_factor = getattr(args, 'daadaptive_input_factor', 4)
    args.lmadaptive_input_cutoff = getattr(args, 'daadaptive_input_cutoff', None)

    args.lmtie_adaptive_weights = getattr(args, 'lmtie_adaptive_weights', False)
    args.lmtie_adaptive_proj = getattr(args, 'lmtie_adaptive_proj', False)

    args.lmno_scale_embedding = getattr(args, 'lmno_scale_embedding', False)
    args.lmlayernorm_embedding = getattr(args, 'lmlayernorm_embedding', False)

    #nmt
    base_architecture(args)

@register_model_architecture('transformer_da', 'transformer_da_lm_iwslt_de_en')
def transformer_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    args.lmdecoder_embed_dim = getattr(args, 'lmdecoder_embed_dim', 512)
    args.lmdecoder_ffn_embed_dim = getattr(args, 'lmdecoder_ffn_embed_dim', 1024)
    args.lmdecoder_layers = getattr(args, 'lmdecoder_layers', 6)
    args.lmdecoder_attention_heads = getattr(args, 'lmdecoder_attention_heads', 4)
    base_da_lm_atchitecture(args)

@register_model_architecture('transformer_da', 'transformer_da_bert')
def base_da_bert_architecture(args):
    #BERT
    args.bertdropout = getattr(args, 'bertdropout', 0.1)
    args.bertattention_dropout = getattr(args, 'bertattention_dropout', 0.1)
    args.bertact_dropout = getattr(args, 'bertact_dropout', 0.0)

    args.bertencoder_ffn_embed_dim = getattr(args, 'bertencoder_ffn_embed_dim', 4096)
    args.bertencoder_layers = getattr(args, 'bertencoder_layers', 6)
    args.bertencoder_attention_heads = getattr(args, 'bertencoder_attention_heads', 8)
    args.bertbias_kv = getattr(args, 'bertbias_kv', False)
    args.bertzero_attn = getattr(args, 'bertzero_attn', False)

    args.bertencoder_embed_dim = getattr(args, 'bertencoder_embed_dim', 1024)
    args.bertshare_encoder_input_output_embed = getattr(args, 'bertshare_encoder_input_output_embed', False)
    args.bertencoder_learned_pos = getattr(args, 'bertencoder_learned_pos', False)
    args.bertno_token_positional_embeddings = getattr(args, 'bertno_token_positional_embeddings', False)
    args.bertnum_segment = getattr(args, 'bertnum_segment', 2)

    args.bertsentence_class_num = getattr(args, 'bertsentence_class_num', 2)
    args.bertsent_loss = getattr(args, 'bertsent_loss', False)

    args.bertapply_bert_init = getattr(args, 'bertapply_bert_init', False)

    args.bertactivation_fn = getattr(args, 'bertactivation_fn', 'relu')
    args.bertpooler_activation_fn = getattr(args, 'bertpooler_activation_fn', 'tanh')
    args.bertencoder_normalize_before = getattr(args, 'bertencoder_normalize_before', False)

    #nmt
    base_architecture(args)

@register_model_architecture('transformer_da', 'transformer_da_bert_iwslt_de_en')
def transformer_da_bert_iwslt_de_en(args):
    # BERT
    args.bertencoder_embed_dim = getattr(args, 'bertencoder_embed_dim', 512)
    args.bertshare_encoder_input_output_embed = getattr(
        args, 'bertshare_encoder_input_output_embed', True)
    args.bertno_token_positional_embeddings = getattr(
        args, 'bertno_token_positional_embeddings', False)
    args.bertencoder_learned_pos = getattr(args, 'bertencoder_learned_pos', True)
    args.bertnum_segment = getattr(args, 'bertnum_segment', 1)

    args.bertencoder_layers = getattr(args, 'bertencoder_layers', 6)

    args.bertencoder_attention_heads = getattr(args, 'bertencoder_attention_heads', 4)
    args.bertencoder_ffn_embed_dim = getattr(args, 'bertencoder_ffn_embed_dim', 1024)
    args.bertbias_kv = getattr(args, 'bertbias_kv', False)
    args.bertzero_attn = getattr(args, 'bertzero_attn', False)

    args.bertsentence_class_num = getattr(args, 'bertsentence_class_num', 1)
    args.bertsent_loss = getattr(args, 'bertsent_loss', True)

    args.bertapply_bert_init = getattr(args, 'bertapply_bert_init', True)

    args.bertactivation_fn = getattr(args, 'bertactivation_fn', 'gelu')
    args.bertpooler_activation_fn = getattr(args, 'bertpooler_activation_fn', 'tanh')
    args.bertencoder_normalize_before = getattr(args, 'bertencoder_normalize_before', True)

    # NMT
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_ffn_embed_dim = getattr(args, 'encoder_ffn_embed_dim', 1024)
    args.encoder_attention_heads = getattr(args, 'encoder_attention_heads', 4)
    args.encoder_layers = getattr(args, 'encoder_layers', 6)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_ffn_embed_dim = getattr(args, 'decoder_ffn_embed_dim', 1024)
    args.decoder_attention_heads = getattr(args, 'decoder_attention_heads', 4)
    args.decoder_layers = getattr(args, 'decoder_layers', 6)

    base_da_bert_architecture(args)


@register_model_architecture("transformer_da", "transformer_da_nmt")
def base_da_nmt_architecture(args):
    args.nmtencoder_embed_path = getattr(args, "nmtencoder_embed_path", None)
    args.nmtencoder_embed_dim = getattr(args, "nmtencoder_embed_dim", 512)
    args.nmtencoder_ffn_embed_dim = getattr(args, "nmtencoder_ffn_embed_dim", 2048)
    args.nmtencoder_layers = getattr(args, "nmtencoder_layers", 6)
    args.nmtencoder_attention_heads = getattr(args, "nmtencoder_attention_heads", 8)
    args.nmtencoder_normalize_before = getattr(args, "nmtencoder_normalize_before", False)
    args.nmtencoder_learned_pos = getattr(args, "nmtencoder_learned_pos", False)
    args.nmtdecoder_embed_path = getattr(args, "nmtdecoder_embed_path", None)
    args.nmtdecoder_embed_dim = getattr(args, "nmtdecoder_embed_dim", args.nmtencoder_embed_dim)
    args.nmtdecoder_ffn_embed_dim = getattr(
        args, "nmtdecoder_ffn_embed_dim", args.nmtencoder_ffn_embed_dim
    )
    args.nmtdecoder_layers = getattr(args, "nmtdecoder_layers", 6)
    args.nmtdecoder_attention_heads = getattr(args, "nmtdecoder_attention_heads", 8)
    args.nmtdecoder_normalize_before = getattr(args, "nmtdecoder_normalize_before", False)
    args.nmtdecoder_learned_pos = getattr(args, "nmtdecoder_learned_pos", False)
    args.nmtattention_dropout = getattr(args, "nmtattention_dropout", 0.0)
    args.nmtactivation_dropout = getattr(args, "nmtactivation_dropout", 0.0)
    args.nmtactivation_fn = getattr(args, "nmtactivation_fn", "relu")
    args.nmtdropout = getattr(args, "nmtdropout", 0.1)
    args.nmtadaptive_softmax_cutoff = getattr(args, "nmtadaptive_softmax_cutoff", None)
    args.nmtadaptive_softmax_dropout = getattr(args, "nmtadaptive_softmax_dropout", 0)
    args.nmtshare_decoder_input_output_embed = getattr(
        args, "nmtshare_decoder_input_output_embed", False
    )
    args.nmtshare_all_embeddings = getattr(args, "nmtshare_all_embeddings", False)
    args.nmtno_token_positional_embeddings = getattr(
        args, "nmtno_token_positional_embeddings", False
    )
    args.nmtadaptive_input = getattr(args, "nmtadaptive_input", False)
    args.nmtno_cross_attention = getattr(args, "nmtno_cross_attention", False)
    args.nmtcross_self_attention = getattr(args, "nmtcross_self_attention", False)
    args.nmtlayer_wise_attention = getattr(args, "nmtlayer_wise_attention", False)

    args.nmtdecoder_output_dim = getattr(
        args, "nmtdecoder_output_dim", args.nmtdecoder_embed_dim
    )
    args.nmtdecoder_input_dim = getattr(args, "nmtdecoder_input_dim", args.nmtdecoder_embed_dim)

    args.nmtno_scale_embedding = getattr(args, "nmtno_scale_embedding", False)
    args.nmtlayernorm_embedding = getattr(args, "nmtlayernorm_embedding", False)
    base_architecture(args)

@register_model_architecture('transformer_da', 'transformer_da_nmt_iwslt_de_en')
def transformer_da_nmt_iwslt_de_en(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 512)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 1024)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 4)
    args.decoder_layers = getattr(args, "decoder_layers", 6)

    args.nmtshare_all_embeddings = getattr(args, "nmtshare_all_embeddings", True)
    args.nmtencoder_embed_dim = getattr(args, "nmtencoder_embed_dim", 512)
    args.nmtencoder_ffn_embed_dim = getattr(args, "nmtencoder_ffn_embed_dim", 1024)
    args.nmtencoder_attention_heads = getattr(args, "nmtencoder_attention_heads", 4)
    args.nmtencoder_layers = getattr(args, "nmtencoder_layers", 6)
    args.nmtdecoder_embed_dim = getattr(args, "nmtdecoder_embed_dim", 512)
    args.nmtdecoder_ffn_embed_dim = getattr(args, "nmtdecoder_ffn_embed_dim", 1024)
    args.nmtdecoder_attention_heads = getattr(args, "nmtdecoder_attention_heads", 4)
    args.nmtdecoder_layers = getattr(args, "nmtdecoder_layers", 6)
    base_da_nmt_architecture(args)

