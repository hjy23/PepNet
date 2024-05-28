import argparse
import datetime
import pickle
import sys
import time
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset
from torchnet import meter
import torch
from sklearn.model_selection import train_test_split

# from model import AIMP
from model_extract_representation import AIMP
from preprocess import parse_fasta, get_properties_features, get_onehot_features, get_pretrained_features
from valid_metrices import CFM_eval_metrics, print_results, eval_metrics, th_eval_metrics

import warnings

warnings.filterwarnings("ignore", message="TypedStorage is deprecated.")
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")


def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--type", "-type", dest="type", type=str, default='AIP',
                        help="The type of training model is antimicrobial peptide or anti-inflammatory peptide, 'AMP' or 'AIP'.",
                        choices=['AMP', 'AIP'])
    parser.add_argument("--train_fasta", "-train_fasta", dest='train_fasta', type=str,
                        default='train.txt',
                        help='The path of the train FASTA file.')
    parser.add_argument("--test_fasta", "-test_fasta", dest='test_fasta', type=str,
                        default='test.txt',
                        help='The path of the test FASTA file.')
    parser.add_argument("--drop", "-drop", dest='drop', type=float, default=0.5,
                        help='The probability of randomly dropping input units during each update in the training period after the concatenation of the results of the three stages.')
    parser.add_argument("--n_transformer", "-n_transformer", dest='n_transformer', type=int, default=1,
                        help='The probability of randomly dropping input units during each update in the training period after the concatenation of the results of the three stages.')
    parser.add_argument("--seed", "-seed", dest='seed', type=int, default=1999,
                        help='the seed of split training set into training and validation.')
    return parser.parse_args()


def checkargs(args):
    if args.type is None or args.train_fasta is None or args.test_fasta is None:
        print('ERROR: please input the necessary parameters!')
        raise ValueError

    if args.type not in ['AMP', 'AIP']:
        print(f'ERROR: type "{args.type}" is not supported by TriNet!')
        raise ValueError

    return


class Config():
    def __init__(self, args):

        self.type = args.type
        self.Dataset_dir = f'../datasets/{self.type}'
        self.train_fasta = args.train_fasta
        self.test_fasta = args.test_fasta
        self.n_transformer = args.n_transformer
        self.drop = args.drop
        self.feature_path = f'{self.Dataset_dir}/feature'
        self.checkpoints = f'{self.Dataset_dir}/checkpoints'
        if self.type == 'AMP':
            self.model_time = '2024_03_27_19_58_59_951'
            self.hidden = 256
            self.batch_size = 128
        else:
            self.model_time = '2024_03_28_15_24_17_806'
            self.batch_size = 256
            self.hidden = 1024
        self.train = False
        if self.model_time is not None:
            self.model_path = f'{self.checkpoints}/{self.model_time}'
        else:
            localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            self.model_path = f'{self.checkpoints}/{localtime}'
        self.submodel_path = self.model_path + '/model'
        self.sublog_path = self.model_path + '/log'

        self.theta = 40
        self.saved_model_num = 2

    def print_config(self):
        for name, value in vars(self).items():
            print('{} = {}'.format(name, value))


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, 'ab', buffering=0)

    def write(self, message):
        self.terminal.write(message)
        try:
            self.log.write(message.encode('utf-8'))
        except ValueError:
            pass

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

    def flush(self):
        pass


"""
data processing part
"""


def data_pre(fasta_file, feature_path, theta=40, shuffle_seed=11, mode='train'):
    if not os.path.exists(feature_path):
        os.makedirs(feature_path)
    fasta_file_path = f'{opt.Dataset_dir}/{fasta_file}'
    names, sequences, labels = parse_fasta(fasta_file_path, number=None)

    name = fasta_file.split('.')[0]
    pre_feas = get_pretrained_features(names, sequences, f'{feature_path}/{name}.h5', theta=theta)
    oh_feas = get_onehot_features(sequences, theta=theta)
    prpt_feas = get_properties_features(sequences, theta=theta)
    features = np.concatenate([oh_feas, prpt_feas], axis=-1)

    return pre_feas, features, labels


def shuffle_dataset(prpt, feature, seq_feas, labels, shuffle_seed):
    np.random.seed(shuffle_seed)
    pos_num = len(np.where(np.array(labels) == 1)[0])
    # shuffle index
    index1 = np.arange(pos_num)  # positive sample
    np.random.shuffle(index1)
    index2 = np.arange(pos_num, len(labels))  # negative sample
    np.random.shuffle(index2)
    index = np.append(index1, index2)
    prpt = prpt[index, :, :]
    labels = np.array(labels)
    labels = labels[index]
    return prpt, feature, labels


def split_dataset(pre_feas, features, labels, split_seed, val_ratio):
    # Take 20% of the trainging dataset to be validation ，X_train X_val y_train y_val
    train1, val1, train_label, val_label = train_test_split(
        pre_feas, labels, test_size=val_ratio, random_state=split_seed)
    train2, val2, _, _ = train_test_split(
        features, labels, test_size=val_ratio, random_state=split_seed)
    train_data = [train1, train2]
    val_data = [val1, val2]
    # train3, val3, _, _ = train_test_split(
    #     seq_feas_tr_eval, labels, test_size=val_ratio, random_state=split_seed)
    # train_data = [train1, train2, train3]
    # val_data = [val1, val2, val3]

    return train_data, train_label, val_data, val_label


class myDataset(Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """

    def __init__(self, pre_feas, feas, labels):
        self.pre_feas = pre_feas
        self.feas = feas
        # self.seq_feas = seq_feas
        self.labels = labels

    def __getitem__(self, index):
        return self.pre_feas[index], self.feas[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


def test(opt, device, model, test_data):
    model_path = '{}/model{}.pth'.format(opt.submodel_path, '_final')
    checkpoints = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoints['model_state_dict'])
    model.to(device)
    model.eval()

    test_dataloader = DataLoader(test_data, batch_size=2 * opt.batch_size, shuffle=False, pin_memory=True)
    test_probs = []
    test_targets = []
    transformer_out_list, feas_em_list, tcn_out_list, pre_feas_list, feas_raw_list = [], [], [], [], []

    with torch.no_grad():
        for ii, data in enumerate(test_dataloader):
            pre_feas, feas, target = data
            # pre_feas, feas, seq_feas, target = data
            pre_feas, feas, target = pre_feas.to(device), feas.to(device), target.to(device)
            # pre_feas, feas, seq_feas, target = pre_feas.to(device), feas.to(device), seq_feas.to(device), target.to(
            #     device)
            # score = model(pre_feas, feas, seq_feas)
            target = target.float()
            score, transformer_out, feas_em, tcn_out, pre_feas, feas_raw = model(pre_feas, feas)
            test_probs += score.tolist()
            test_targets += target.tolist()
            transformer_out_list.append(transformer_out.cpu().numpy())
            feas_em_list.append(feas_em.cpu().numpy())
            tcn_out_list.append(tcn_out.cpu().numpy())
            pre_feas_list.append(pre_feas.cpu().numpy())
            feas_raw_list.append(feas_raw.cpu().numpy())

    test_probs = np.array(test_probs)
    test_targets = np.array(test_targets)
    transformer_out_list = np.concatenate(transformer_out_list, axis=0)
    feas_em_list = np.concatenate(feas_em_list, axis=0)
    tcn_out_list = np.concatenate(tcn_out_list, axis=0)
    pre_feas_list = np.concatenate(pre_feas_list, axis=0)
    feas_raw_list = np.concatenate(feas_raw_list, axis=0)

    return test_probs, test_targets, transformer_out_list, feas_em_list, tcn_out_list, pre_feas_list, feas_raw_list


def main(opt, device, shuffle_seed=13, split_seed=7, val_ratio=0.2):
    feature_path = opt.feature_path
    print('Loading the data...')
    pre_feas_tr_eval, feature_tr_eval, label_tr_eval = data_pre(fasta_file=opt.train_fasta,
                                                                feature_path=feature_path,
                                                                theta=opt.theta,
                                                                mode='train')
    pre_feas_test, feature_test, label_test = data_pre(fasta_file=opt.test_fasta,
                                                       feature_path=feature_path,
                                                       theta=opt.theta,
                                                       mode='test')
    print('Finish loading data!')
    train_data, train_label, valid_data, valid_label = split_dataset(pre_feas_tr_eval, feature_tr_eval,
                                                                     label_tr_eval, split_seed=split_seed,
                                                                     val_ratio=val_ratio)
    train_data = [torch.Tensor(v) for v in train_data]
    valid_data = [torch.Tensor(v) for v in valid_data]
    test_data = [torch.Tensor(v) for v in [pre_feas_test, feature_test]]
    # test_data = [torch.Tensor(v) for v in [pre_feas_test, feature_test, seq_feas_test]]
    train_label = torch.Tensor(train_label).long()
    valid_label = torch.Tensor(valid_label).long()
    test_label = torch.Tensor(label_test).long()

    # train_data = myDataset(train_data[0], train_data[1], train_label)
    valid_data = myDataset(valid_data[0], valid_data[1], valid_label)
    test_data = myDataset(test_data[0], test_data[1], test_label)

    print('Loading the model...')
    # model = AIMP(pre_feas_dim=1024, feas_dim=34, hidden=opt.hidden, dropout=opt.drop)
    model = AIMP(pre_feas_dim=1024, feas_dim=34, hidden=opt.hidden, n_transformer=opt.n_transformer, dropout=opt.drop)

    # predict
    print('Test...')
    valid_probs, valid_labels, valid_transformer_out_list, valid_feas_em_list, valid_tcn_out_list, valid_pre_feas_list, valid_feas_raw_list = test(
        opt, device, model, valid_data)
    test_probs, test_labels, test_transformer_out_list, test_feas_em_list, test_tcn_out_list, test_pre_feas_list, test_feas_raw_list = test(
        opt, device, model, test_data)

    th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_, pred_class = eval_metrics(valid_probs, valid_labels)
    valid_matrices = th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_

    th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_, pred_class = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_,

    print_results(valid_matrices, test_matrices)
    results = {'test_probs': test_probs, 'test_labels': test_labels, 'transformer_out': test_transformer_out_list,
               'feas_em': test_feas_em_list, 'tcn_out': test_tcn_out_list, 'pre_feas': test_pre_feas_list,
               'feas_raw': test_feas_raw_list}

    with open(opt.sublog_path + '/results_test.pkl', 'wb') as f:
        pickle.dump(results, f)

    # with open(opt.sublog_path + '/results.pkl', 'rb') as f:
    #     results_ref = pickle.load(f)
    return


if __name__ == '__main__':
    arguments = parse_args()
    checkargs(arguments)
    opt = Config(arguments)
    opt.print_config()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    main(opt, device)
