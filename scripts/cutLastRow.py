import argparse

import torch 
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser(description='Command-line script for keys in checkpoint.')
    parser.add_argument('--path', type=str, help='checkpoint path')
    parser.add_argument('--out', type=str, help='output checkpoint path')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    checkpoint = torch.load(args.path, map_location=torch.device('cpu'))
    checkpoint['model']['encoder.sentence_encoder.embed_tokens.weight'] = checkpoint['model']['encoder.sentence_encoder.embed_tokens.weight'][:-1]
    checkpoint['model']['encoder.lm_output_learned_bias'] = checkpoint['model']['encoder.lm_output_learned_bias'][:-1]
    torch.save(checkpoint, args.out)

if __name__ == '__main__':
    main()
