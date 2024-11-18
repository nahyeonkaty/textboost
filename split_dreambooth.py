#!/usr/bin/env python3
import argparse
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='Split dreambooth photos')
    parser.add_argument('--dreambooth-dir', help='Path to the dreambooth directory')
    parser.add_argument('--split-file', default='data/dreambooth_n1.txt', help='Path to the split file')
    parser.add_argument('--output-dir', default='data', help='Output file')
    return parser.parse_args()


def main(args):
    with open(args.split_file, 'r') as f:
        train_files = f.readlines()

    prefix = os.path.basename(args.split_file).split('.')[0]
    train_dir = os.path.join(args.output_dir, prefix + '_train')
    val_dir = os.path.join(args.output_dir, prefix + '_val')

    for train_file in train_files:
        instance, filename = train_file.strip().split(',')
        all_files = os.listdir(os.path.join(args.dreambooth_dir, instance))
        os.makedirs(os.path.join(train_dir, instance), exist_ok=True)
        os.makedirs(os.path.join(val_dir, instance), exist_ok=True)
        for file in all_files:
            src = os.path.join(args.dreambooth_dir, instance, file)
            if file == filename:
                dst = os.path.join(train_dir, instance, file)
            else:
                dst = os.path.join(val_dir, instance, file)
            shutil.copy(src, dst)


if __name__ == "__main__":
    args = parse_args()
    main(args)
