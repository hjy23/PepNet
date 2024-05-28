import argparse
import datetime
import pickle
import sys
import time
import numpy as np
import os

import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchnet import meter
import torch
from sklearn.model_selection import train_test_split

from model_prefeature import AIMP
from preprocess import parse_fasta_predict, get_properties_features, get_onehot_features, get_pretrained_features_predict, \
    get_pretrained_features
from valid_metrices import CFM_eval_metrics, print_results, eval_metrics, th_eval_metrics
import warnings

warnings.filterwarnings("ignore", message="TypedStorage is deprecated.")
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--type", "-type", dest="type", type=str, default='AIP',
                        help="The type of training model is antimicrobial peptide or anti-inflammatory peptide, 'AMP' or 'AIP'.",
                        choices=['AMP', 'AIP'])
    parser.add_argument("--test_fasta", "-test_fasta", dest='test_fasta', type=str,
                        default='../datasets/AIP/test.txt',
                        help='The path of the test FASTA file.')
    parser.add_argument("--output_path", "-output_path", dest='output_path', type=str,
                        default='',
                        help='The output_path.')
    parser.add_argument("--drop", "-drop", dest='drop', type=float, default=0.5,
                        help='The probability of randomly dropping input units during each update in the training period after the concatenation of the results of the three stages.')
    parser.add_argument("--n_transformer", "-n_transformer", dest='n_transformer', type=int, default=1,
                        help='The probability of randomly dropping input units during each update in the training period after the concatenation of the results of the three stages.')
    parser.add_argument("--seed", "-seed", dest='seed', type=int, default=1999,
                        help='the seed of split training set into training and validation.')
    return parser.parse_args()


def checkargs(args):
    if args.type is None or args.test_fasta is None:
        print('ERROR: please input the necessary parameters!')
        raise ValueError

    if args.type not in ['AMP', 'AIP']:
        print(f'ERROR: type "{args.type}" is not supported by PepNet!')
        raise ValueError

    return


class Config():
    def __init__(self, args):

        self.type = args.type
        if self.type == 'AMP':
            self.th = 0.60
            self.batch_size = 256
            self.hidden = 256
            self.model_time = '2024_03_29_14_50_24_no_pre-feature_908'
        elif self.type == 'AIP':
            self.th = 0.12
            self.batch_size = 256
            self.hidden = 1024
            self.model_time = '2024_03_29_14_37_31_no_pre-feature_724'
        self.Dataset_dir = f'../datasets/{self.type}'
        self.test_fasta = args.test_fasta
        self.n_transformer = args.n_transformer
        self.drop = args.drop
        self.output_path = args.output_path
        self.checkpoints = f'{self.Dataset_dir}/checkpoints'
        self.model_path = f'{self.checkpoints}/{self.model_time}'
        self.submodel_path = self.model_path + '/model'

        self.theta = 40

    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))


"""
data processing part
"""


def data_pre(fasta_file, theta=40, shuffle_seed=11, mode='train'):
    # fasta_file_path = f'{opt.output_path}/{fasta_file}'
    fasta_file_path = fasta_file
    names, sequences = parse_fasta_predict(fasta_file_path, number=None)

    pre_feas = np.zeros([len(sequences), 1])
    oh_feas = get_onehot_features(sequences, theta=theta)
    prpt_feas = get_properties_features(sequences, theta=theta)
    features = np.concatenate([oh_feas, prpt_feas], axis=-1)

    return sequences, pre_feas, features


class myDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """

    def __init__(self, pre_feas, feas):
        self.pre_feas = pre_feas
        self.feas = feas

    def __getitem__(self, index):
        return self.pre_feas[index], self.feas[index]

    def __len__(self):
        return self.feas.size(0)


"""
test part
"""


def test(opt, device, model, test_data):
    model_path = '{}/model{}.pth'.format(opt.submodel_path, '_final')
    checkpoints = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    model.eval()

    test_dataloader = DataLoader(test_data, batch_size=2 * opt.batch_size, shuffle=False, pin_memory=True)
    test_probs = []

    with torch.no_grad():
        for ii, data in enumerate(test_dataloader):
            pre_feas, feas = data
            pre_feas, feas = pre_feas.to(device), feas.to(device)
            score = model(pre_feas, feas)
            test_probs += score.tolist()

    test_probs = np.array(test_probs)

    return test_probs


def main(opt, device, shuffle_seed=13, split_seed=7, val_ratio=0.2):
    print('Loading the data...')
    sequences_test, pre_feas_test, feature_test = data_pre(fasta_file=opt.test_fasta,
                                                           theta=opt.theta,
                                                           mode='test')
    print('Finish loading data!')

    test_data = [torch.Tensor(v) for v in [pre_feas_test, feature_test]]

    test_data = myDataset(test_data[0], test_data[1])

    print('Loading the model...')
    model = AIMP(pre_feas_dim=1024, feas_dim=34, hidden=opt.hidden, n_transformer=opt.n_transformer, dropout=opt.drop)

    # predict
    print('Predict...')

    test_probs = test(opt, device, model, test_data)
    pred_class = np.where(test_probs > opt.th, 1, 0)

    results = {'sequence': sequences_test, 'probability': test_probs, 'Binary': pred_class}

    result_df = pd.DataFrame(data=results)
    result_df.to_csv(f'{opt.output_path}/{opt.type}_prediction_result.csv', float_format='%.3f',
                     columns=["sequence", "probability", "Binary"])

    return


if __name__ == '__main__':
    arguments = parse_args()
    checkargs(arguments)
    opt = Config(arguments)
    opt.print_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(opt, device)
