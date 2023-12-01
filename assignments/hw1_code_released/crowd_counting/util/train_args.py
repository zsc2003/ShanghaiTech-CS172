import os
import argparse
from pyhocon import ConfigFactory
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-B", type=int, default=1, help="batch size|only support 1 batch for different img resolution")
    parser.add_argument("--name", "-n", type=str, default='crowd_counting', help="experiment name")
    parser.add_argument("--logs_path", type=str, default="train/logs", help="logs output directory",)
    parser.add_argument("--checkpoints_path",type=str,default="train/checkpoints",help="checkpoints output directory",)
    parser.add_argument("--epochs",type=int,default=200,help="number of epochs to train for",)
    parser.add_argument("--extra_epochs",type=int,default=0,help="extra epochs for further use",)
    parser.add_argument("--start_epochs",type=int,default=0,help="which epoch to begin",)
    parser.add_argument("--history_epoch", type=int, default=0, help="resume from the history net")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--step_size", type=int, default=50, help="learning rate decay after how much step")
    parser.add_argument("--gamma", type=float, default=0.5, help="learning rate decay factor")
    parser.add_argument("--datadir", "-D", type=str, default=None, help="Dataset directory")
    parser.add_argument("--device", type=str, default='cuda', help='compute device')
    parser.add_argument("--resume", "-r", action="store_true", help="continue training")
    parser.add_argument("--save_interval", type=int, default=1, help="save the checkpoint after how much step")
    parser.add_argument("--test_interval", type=int, default=5, help="test the model after how much step")

    args = parser.parse_args()

    print("Experiment name:", args.name)
    print("Resume?", "yes" if args.resume else "no")
    print("* Dataset location:", args.datadir)
    return args
