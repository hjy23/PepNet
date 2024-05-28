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

from model import AIMP
# from model_prefeature import AIMP
# from model_adaptive_avg_pool import AIMP
# from model_tcn_ablation import AIMP
# from model_transformer_ablation import AIMP
from preprocess import parse_fasta, get_properties_features, get_onehot_features, get_pretrained_features
from valid_metrices import CFM_eval_metrics, print_results, eval_metrics, th_eval_metrics

import warnings

warnings.filterwarnings("ignore", message="TypedStorage is deprecated.")
warnings.filterwarnings("ignore", message="indexing with dtype torch.uint8 is now deprecated")


# python train.py --type AMP --train_fasta train.txt --test_fasta test.txt --hidden 256 --batch_size 128 --epoch 100
# python train.py --type AIP --train_fasta train.txt --test_fasta test.txt --hidden 1024 --batch_size 256 --epoch 3

def parse_args():
    parser = argparse.ArgumentParser(description="Launch a list of commands.")
    parser.add_argument("--type", "-type", dest="type", type=str, default='AIP',
                        help="The type of training model is antimicrobial peptide or anti-inflammatory peptide, 'AMP' or 'AIP'.",
                        choices=['AMP', 'AIP'])
    parser.add_argument("--train_fasta", "-train_fasta", dest='train_fasta', type=str,
                        default='train.txt',
                        # default='train_chenqixuan.txt',
                        # default='AIP_train.txt',
                        # default='train_o.txt',
                        help='The path of the train FASTA file.')
    parser.add_argument("--test_fasta", "-test_fasta", dest='test_fasta', type=str,
                        default='test.txt',
                        # default='test_chenqixuan.txt',
                        # default='AIP_test.txt',
                        # default='test_o.txt',
                        help='The path of the test FASTA file.')
    parser.add_argument("--hidden", "-hidden", dest='hidden', type=int, default=1024,
                        help='The number of hidden units or dimensions in a neural network layer.')
    parser.add_argument("--drop", "-drop", dest='drop', type=float, default=0.5,
                        help='The probability of randomly dropping input units during each update in the training period after the concatenation of the results of the three stages.')
    parser.add_argument("--n_transformer", "-n_transformer", dest='n_transformer', type=int, default=1,
                        help='The number of transformer layers parameter determines how many identical layers the transformer model will have. ')
    parser.add_argument("--lr", "-lr", dest='lr', type=float, default=0.0001,
                        help='The initial learning rate used to control the step size of parameter updates in each update.')
    parser.add_argument("--batch_size", "-batch_size", dest='batch_size', type=int, default=256,
                        help='The number of training examples utilized in one iteration during the training process of a machine learning model.')
    parser.add_argument("--seed", "-seed", dest='seed', type=int, default=1999,
                        help='the seed of split training set into training and validation.')
    parser.add_argument("--epoch", "-epoch", dest='epoch', type=int, default=100,
                        help='The number of times the entire training dataset is passed forward and backward through the neural network during the training process.')
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

        # self.ligand = 'P' + args.ligand if args.ligand != 'HEME' else 'PHEM'
        self.type = args.type
        self.Dataset_dir = f'../datasets/{self.type}'
        self.train_fasta = args.train_fasta
        self.test_fasta = args.test_fasta
        self.batch_size = args.batch_size
        self.hidden = args.hidden
        self.n_transformer = args.n_transformer
        self.drop = args.drop
        self.epoch = args.epoch
        self.lr = args.lr
        self.feature_path = f'{self.Dataset_dir}/feature'
        self.checkpoints = f'{self.Dataset_dir}/checkpoints'
        self.model_time = None
        self.train = True
        if self.model_time is not None:
            self.model_path = f'{self.checkpoints}/{self.model_time}'
        else:
            localtime = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
            self.model_path = f'{self.checkpoints}/{localtime}'
        self.submodel_path = self.model_path + '/model'
        self.sublog_path = self.model_path + '/log'
        if not os.path.exists(self.submodel_path): os.makedirs(self.submodel_path)
        if not os.path.exists(self.sublog_path): os.makedirs(self.sublog_path)
        os.system(f'cp ./train.py {self.model_path}')
        os.system(f'cp ./model.py {self.model_path}')

        self.max_metric = 'PRC'
        self.theta = 40
        self.saved_model_num = 3
        self.early_stop_epochs = 10

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
    # features = oh_feas  # without prpt
    # features = prpt_feas  # without oh

    return pre_feas, features, labels, names, sequences


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


def split_dataset(pre_feas, features, labels, names, seqs, split_seed, val_ratio):
    # Take 20% of the trainging dataset to be validation ，X_train X_val y_train y_val
    train1, val1, train_label, val_label = train_test_split(
        pre_feas, labels, test_size=val_ratio, random_state=split_seed)
    train2, val2, _, _ = train_test_split(
        features, labels, test_size=val_ratio, random_state=split_seed)
    train3, val3, _, _ = train_test_split(
        names, labels, test_size=val_ratio, random_state=split_seed)
    train4, val4, _, _ = train_test_split(
        seqs, labels, test_size=val_ratio, random_state=split_seed)

    # for i, v in enumerate(train_label):
    #     if v == 1:
    #         with open(f'{opt.Dataset_dir}/train_pos.fa', 'a') as f:
    #             f.write(f'>{train3[i]}\n')
    #             f.write(f'{train4[i]}\n')
    #     elif v == 0:
    #         with open(f'{opt.Dataset_dir}/train_neg.fa', 'a') as f:
    #             f.write(f'>{train3[i]}\n')
    #             f.write(f'{train4[i]}\n')
    # for i, v in enumerate(val_label):
    #     if v == 1:
    #         with open(f'{opt.Dataset_dir}/val_pos.fa', 'a') as f:
    #             f.write(f'>{val3[i]}\n')
    #             f.write(f'{val4[i]}\n')
    #     elif v == 0:
    #         with open(f'{opt.Dataset_dir}/val_neg.fa', 'a') as f:
    #             f.write(f'>{val3[i]}\n')
    #             f.write(f'{val4[i]}\n')

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


"""
training part
"""


def train(opt, device, model, train_data, valid_data, test_data):
    train_dataloader = DataLoader(train_data, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                           patience=opt.early_stop_epochs // 2,
                                                           min_lr=1e-6)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    criterion = torch.nn.BCELoss()
    epoch_begin = 0

    # print('** loss function: {}'.format(criterion))

    # save_path = f'{opt.submodel_path}/model_initial.pth'
    # print('save initial weight: ', save_path)
    # torch.save({
    #     'model_state_dict': model.state_dict(),
    #     # 'optimizer': optimizer,
    #     # 'criterion': criterion,
    # }, save_path)

    model.to(device)
    criterion.to(device)

    loss_meter = meter.AverageValueMeter()

    early_stop_iter = 0
    max_metric_val = -1
    nsave_model = 0
    begintime = datetime.datetime.now()
    print('Time:', begintime)
    for epoch in range(epoch_begin, opt.epoch):
        nstep = len(train_dataloader)
        for ii, data in enumerate(train_dataloader):
            model.train()
            pre_feas, feas, target = data
            # pre_feas, feas, seq_feas, target = data
            pre_feas, feas, target = pre_feas.to(device), feas.to(device), target.to(device)
            # pre_feas, feas, seq_feas, target = pre_feas.to(device), feas.to(device), seq_feas.to(device), target.to(
            #     device)
            optimizer.zero_grad()
            score = model(pre_feas, feas)
            # score = model(pre_feas, feas, seq_feas)
            target = target.float()
            loss = criterion(score, target)

            # l2_reg = torch.tensor(0.).to(device)
            # for param in model.parameters():
            #     l2_reg += torch.norm(param, 2)
            # loss += l2_reg * 0.001

            loss.backward()
            optimizer.step()
            loss_meter.add(loss.item())

            # if ii % (nstep - 1) == 0 and ii != 0:
        nowtime = datetime.datetime.now()
        print('|| Epoch{} step{} || lr={:.6f} | train_loss={:.5f}'.format(epoch, nstep,
                                                                          optimizer.param_groups[0]['lr'],
                                                                          loss_meter.mean))
        print('Time:', nowtime)
        print('Timedelta: %s seconds' % (nowtime - begintime).seconds)
        val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc = val(opt, device, model,
                                                                                            valid_data,
                                                                                            'valid', val_th=None)
        test_th, val_acc, test_rec, test_pre, test_F1, test_spe, test_mcc, test_auc, test_prc = val(opt, device, model,
                                                                                                    test_data, 'test',
                                                                                                    val_th)

        if opt.max_metric == 'AUC':
            metrice_val = val_auc
        elif opt.max_metric == 'MCC':
            metrice_val = val_mcc
        elif opt.max_metric == 'F1':
            metrice_val = val_F1
        elif opt.max_metric == 'ACC':
            metrice_val = val_acc
        elif opt.max_metric == 'PRC':
            metrice_val = val_prc
        else:
            print('ERROR: opt.max_metric.')
            raise ValueError

        if metrice_val > max_metric_val:
            max_metric_val = metrice_val
            if nsave_model < opt.saved_model_num:
                save_path = '{}/model{}.pth'.format(opt.submodel_path, nsave_model)
                print('save net: ', save_path)
                # torch.save([model, criterion, optimizer, val_th, epoch], save_path)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    # 'optimizer': optimizer,
                    # 'criterion': criterion,
                    'val_th': val_th,
                    # 'epoch': epoch,
                }, save_path)
                nsave_model += 1
            else:
                save_path = '{}/model{}.pth'.format(opt.submodel_path, nsave_model - 1)
                print('save net: ', save_path)
                for model_i in range(1, opt.saved_model_num):
                    os.system(
                        'mv {}/model{}.pth {}/model{}.pth'.format(opt.submodel_path, model_i, opt.submodel_path,
                                                                  model_i - 1))
                # torch.save([model, criterion, optimizer, val_th, epoch], save_path)
                torch.save({
                    'model_state_dict': model.state_dict(),
                    # 'optimizer': optimizer,
                    # 'criterion': criterion,
                    'val_th': val_th,
                    # 'epoch': epoch,
                }, save_path)

            early_stop_iter = 0
        else:
            early_stop_iter += 1
            if early_stop_iter == opt.early_stop_epochs:
                break

        scheduler.step(metrice_val)
        loss_meter.reset()

    return


def val(opt, device, model, valid_data, dataset_type, val_th=None):
    valid_dataloader = DataLoader(valid_data, batch_size=2 * opt.batch_size, shuffle=False, pin_memory=True)
    model.eval()
    if val_th is not None:
        AUC_meter = meter.AUCMeter()
        PRC_meter = meter.APMeter()
        Confusion_meter = meter.ConfusionMeter(k=2)
        with torch.no_grad():
            for ii, data in enumerate(valid_dataloader):
                pre_feas, feas, target = data
                # pre_feas, feas, seq_feas, target = data
                pre_feas, feas, target = pre_feas.to(device), feas.to(device), target.to(device)
                # pre_feas, feas, seq_feas, target = pre_feas.to(device), feas.to(device), seq_feas.to(device), target.to(
                #     device)
                # score = model(pre_feas, feas, seq_feas)
                score = model(pre_feas, feas).float()
                # target = target.float()
                AUC_meter.add(score, target)
                PRC_meter.add(score, target)
                pred_bi = target.data.new(score.shape).fill_(0)
                pred_bi[score > val_th] = 1
                Confusion_meter.add(pred_bi, target)

        val_auc = AUC_meter.value()[0]
        val_prc = PRC_meter.value().item()
        cfm = Confusion_meter.value()
        val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc = CFM_eval_metrics(cfm)

        try:
            print(
                '{} result: th={:.2f} acc={:.3f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} PRC={:.3f}'
                .format(dataset_type, val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc))
        except:
            print(dataset_type, val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc)
    else:
        AUC_meter = meter.AUCMeter()
        PRC_meter = meter.APMeter()
        for j in range(2, 100, 2):
            th = j / 100.0
            locals()['Confusion_meter_' + str(th)] = meter.ConfusionMeter(k=2)
        with torch.no_grad():
            for ii, data in enumerate(valid_dataloader):
                pre_feas, feas, target = data
                # pre_feas, feas, seq_feas, target = data
                pre_feas, feas, target = pre_feas.to(device), feas.to(device), target.to(device)
                # pre_feas, feas, seq_feas, target = pre_feas.to(device), feas.to(device), seq_feas.to(device), target.to(
                #     device)
                # score = model(pre_feas, feas, seq_feas)
                target = target.float()
                score = model(pre_feas, feas).float()
                AUC_meter.add(score, target)
                PRC_meter.add(score, target)
                for j in range(2, 100, 2):
                    th = j / 100.0
                    pred_bi = target.data.new(score.shape).fill_(0)
                    pred_bi[score > th] = 1
                    locals()['Confusion_meter_' + str(th)].add(pred_bi, target)
        val_auc = AUC_meter.value()[0]
        val_prc = PRC_meter.value().item()
        val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc = -1, -1, -1, -1, -1, -1
        for j in range(2, 100, 2):
            th = j / 100.0
            cfm = locals()['Confusion_meter_' + str(th)].value()
            acc, rec, pre, F1, spe, mcc = CFM_eval_metrics(cfm)
            prc = 0
            if F1 > val_F1:
                val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc = acc, rec, pre, F1, spe, mcc
                val_th = th
        try:
            print(
                '{} result: th={:.2f} acc={:.3f} sen={:.3f} pre={:.3f} F1={:.3f}, spe={:.3f} MCC={:.3f} AUC={:.3f} PRC={:.3f}'
                .format(dataset_type, val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc))
        except:
            print(dataset_type, val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc)

    return val_th, val_acc, val_rec, val_pre, val_F1, val_spe, val_mcc, val_auc, val_prc


def test(opt, device, model, test_data):
    avg_test_probs = []
    avg_test_targets = []

    for model_i in range(opt.saved_model_num):
        model_path = '{}/model{}.pth'.format(opt.submodel_path, model_i)
        # model, criterion, optimizer, th, _ = torch.load(model_path)
        checkpoints = torch.load(model_path)
        model.load_state_dict(checkpoints['model_state_dict'])
        model.to(device)
        model.eval()

        test_dataloader = DataLoader(test_data, batch_size=2 * opt.batch_size, shuffle=False, pin_memory=True)
        test_probs = []
        test_targets = []
        with torch.no_grad():
            for ii, data in enumerate(test_dataloader):
                pre_feas, feas, target = data
                # pre_feas, feas, seq_feas, target = data
                pre_feas, feas, target = pre_feas.to(device), feas.to(device), target.to(device)
                # pre_feas, feas, seq_feas, target = pre_feas.to(device), feas.to(device), seq_feas.to(device), target.to(
                #     device)
                # score = model(pre_feas, feas, seq_feas)
                target = target.float()
                score = model(pre_feas, feas).float()
                test_probs += score.tolist()
                test_targets += target.tolist()
        test_probs = np.array(test_probs)
        test_targets = np.array(test_targets)
        avg_test_probs.append(test_probs.reshape(-1, 1))
        avg_test_targets.append(test_targets.reshape(-1, 1))

    avg_test_probs = np.concatenate(avg_test_probs, axis=1)
    avg_test_probs = np.average(avg_test_probs, axis=1)

    avg_test_targets = np.concatenate(avg_test_targets, axis=1)
    avg_test_targets = np.average(avg_test_targets, axis=1)

    return avg_test_probs, avg_test_targets


def main(opt, device, shuffle_seed=13, split_seed=7, val_ratio=0.2):
    with open(f'{opt.model_path}/params.pkl', 'wb') as f:
        pickle.dump(opt, f)
    feature_path = opt.feature_path
    print('Loading the data...')
    pre_feas_tr_eval, feature_tr_eval, label_tr_eval, names_tr_eval, seqs_tr_eval = data_pre(
        fasta_file=opt.train_fasta,
        feature_path=feature_path,
        theta=opt.theta,
        mode='train')
    pre_feas_test, feature_test, label_test, names_test, seqs_test = data_pre(fasta_file=opt.test_fasta,
                                                                              feature_path=feature_path,
                                                                              theta=opt.theta,
                                                                              mode='test')
    print('Finish loading data!')
    train_data, train_label, valid_data, valid_label = split_dataset(pre_feas_tr_eval, feature_tr_eval,
                                                                     label_tr_eval,
                                                                     names_tr_eval,
                                                                     seqs_tr_eval,
                                                                     split_seed=split_seed,
                                                                     val_ratio=val_ratio)

    train_data = [torch.Tensor(v) for v in train_data]
    valid_data = [torch.Tensor(v) for v in valid_data]
    test_data = [torch.Tensor(v) for v in [pre_feas_test, feature_test]]
    # test_data = [torch.Tensor(v) for v in [pre_feas_test, feature_test, seq_feas_test]]
    train_label = torch.Tensor(train_label).long()
    valid_label = torch.Tensor(valid_label).long()
    test_label = torch.Tensor(label_test).long()

    train_data = myDataset(train_data[0], train_data[1], train_label)
    valid_data = myDataset(valid_data[0], valid_data[1], valid_label)
    test_data = myDataset(test_data[0], test_data[1], test_label)
    # train_data = myDataset(train_data[0], train_data[1], train_data[2], train_label)
    # valid_data = myDataset(valid_data[0], valid_data[1], valid_data[2], valid_label)
    # test_data = myDataset(test_data[0], test_data[1], test_data[2], test_label)

    print('Loading the model...')
    # model = AIMP(pre_feas_dim=1024, feas_dim=34, hidden=opt.hidden, dropout=opt.drop)
    model = AIMP(pre_feas_dim=1024, feas_dim=feature_test.shape[-1], hidden=opt.hidden, n_transformer=opt.n_transformer,
                 dropout=opt.drop)

    print("Train...")
    train(opt, device, model, train_data, valid_data, test_data)
    print('Training is finished!')

    # predict
    print('Test...')
    valid_probs, valid_labels = test(opt, device, model, valid_data)
    test_probs, test_labels = test(opt, device, model, test_data)

    th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_, pred_class = eval_metrics(valid_probs, valid_labels)
    valid_matrices = th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_

    th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_, pred_class = th_eval_metrics(th_, test_probs, test_labels)
    test_matrices = th_, acc_, rec_, pre_, f1_, spe_, mcc_, roc_, prc_,

    print_results(valid_matrices, test_matrices)

    results = {'valid_probs': valid_probs, 'valid_labels': valid_labels, 'test_probs': test_probs,
               'test_labels': test_labels}
    with open(opt.sublog_path + '/results.pkl', 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    arguments = parse_args()
    checkargs(arguments)
    opt = Config(arguments)
    sys.stdout = Logger(opt.model_path + '/training.log')
    opt.print_config()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    # print(device)
    main(opt, device)
    sys.stdout.log.close()
