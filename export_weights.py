#!/usr/bin/env python3
"""从 checkpoint 导出 .weights 文件（独立版，不依赖 quoridor_cpp）。"""
import sys, os, argparse
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "rl"))

from config import CONFIG
from model import QuoridorNet, export_weights

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="path to .pt checkpoint")
    parser.add_argument("--output", default=None, help="output .weights path")
    args = parser.parse_args()

    config = CONFIG
    net = QuoridorNet(config["input_channels"])
    net.eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    net.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint: epoch {ckpt.get('epoch', '?')}")

    out_path = args.output or os.path.join(config["export_dir"], config["export_name"])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    export_weights(net, out_path)
    print(f"Weights exported -> {out_path}")

if __name__ == "__main__":
    main()
