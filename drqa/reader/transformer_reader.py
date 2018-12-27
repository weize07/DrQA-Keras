import argparse
import os

from drqa.reader import utils, vector, config, data
ROOT_DIR = '/Users/weize/Workspace/VENV-3.6/workspace/DrQA-Keras'

def prepare_data(args):
    train_exs = utils.load_data(args, args.train_file, skip_no_answer=True)
    return train_exs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'DrQA Document Reader',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    args = parser.parse_args()
    args.train_file = os.path.join(ROOT_DIR, 'data/datasets/SQuAD-v1.1-train-processed-corenlp.txt')
    args.uncased_question = True
    args.uncased_doc = True
    train_data = prepare_data(args)
    print(train_data[0])