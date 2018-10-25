import argparse


class DoodleParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Parser for Doodle Classification Challenge')
        self.build_parser()

    def build_parser(self):
        parser = self.parser
        parser.add_argument('--iter', type=int, default=800,
                            help='Total number of steps (batches of samples) to yield from generator '
                                 'before declaring one epoch finished and starting the next epoch.')
        parser.add_argument('--epoch', type=int, default=16,
                            help='Number of epochs to train the model.')
        parser.add_argument('--raw-image-size', type=int, default=256,
                            help='Size of raw training images, default is 256 x 256.')
        parser.add_argument('--image-size', type=int, default=72,
                            help='Size of training images, default is 64 x 64.')
        parser.add_argument('--batch-size', type=int, default=680,
                            help='Number of samples per evaluation step.')
        parser.add_argument('--lr', type=float, default=0.002,
                            help='Learning rate, default is 0.002')
        parser.add_argument('--random-seed', type=int, default=1987,
                            help='Number of samples per evaluation step.')
        parser.add_argument('--input-dir', type=str, default='./input/quickdraw-doodle-recognition/',
                            help='Input file directory')
        parser.add_argument('--dp-dir', type=str, default='./input/shuffle-csvs/',
                            help='DP file directory')
        parser.add_argument('--num-csv', type=int, default=100)
        parser.add_argument('--num-class', type=int, default=340)

    def parse(self):
        return self.parser.parse_args()
