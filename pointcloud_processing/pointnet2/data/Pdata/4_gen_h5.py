import argparse
import os
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['non_pipe', 'pipe']
NUM_CLASSES = len(classes)

class PipeSegDataset(Dataset):
    def __init__(self, h5_path, num_point=4096, block_size=16384):
        self.num_point = num_point
        self.block_size = block_size
        
        f = h5py.File(h5_path, 'r')
        self.points = f['points'][:]
        self.labels = f['labels'][:]
        f.close()
        
        # 计算每个房间可以分成多少块
        self.room_blocks = []
        for i in range(len(self.points)):
            num_blocks = len(self.points[i]) // block_size
            for j in range(num_blocks):
                self.room_blocks.append((i, j*block_size))
        
        # 计算类别权重
        label_weights = np.zeros(NUM_CLASSES)
        for seg in self.labels:
            tmp, _ = np.histogram(seg, range(NUM_CLASSES + 1))
            label_weights += tmp
        label_weights = label_weights.astype(np.float32)
        label_weights = label_weights / np.sum(label_weights)
        self.labelweights = np.power(np.amax(label_weights) / label_weights, 1/3.0)

    def __len__(self):
        return len(self.room_blocks)

    def __getitem__(self, idx):
        room_idx, start_idx = self.room_blocks[idx]
        end_idx = start_idx + self.block_size
        
        points = self.points[room_idx][start_idx:end_idx].copy()
        labels = self.labels[room_idx][start_idx:end_idx].copy()
        
        # 从block中随机采样
        if len(points) > self.num_point:
            choice = np.random.choice(len(points), self.num_point, replace=False)
        else:
            choice = np.random.choice(len(points), self.num_point, replace=True)
            
        return torch.FloatTensor(points[choice]), torch.LongTensor(labels[choice])

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--decay_rate', type=float, default=1e-4)
    parser.add_argument('--npoint', type=int, default=4096)
    parser.add_argument('--block_size', type=int, default=16384)
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    # 设置GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    # 创建日志目录
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('pipe_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # 设置日志
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    # 数据路径
    TRAIN_DATA_PATH = 'data/Pdata/scenes/h5_data/indoor3d_pipe_sem_seg_train.h5'
    TEST_DATA_PATH = 'data/Pdata/scenes/h5_data/indoor3d_pipe_sem_seg_test.h5'
    VAL_DATA_PATH = 'data/Pdata/scenes/h5_data/indoor3d_pipe_sem_seg_val.h5'
    
    # 加载数据集
    print("Loading training data ...")
    TRAIN_DATASET = PipeSegDataset(h5_path=TRAIN_DATA_PATH, num_point=args.npoint, block_size=args.block_size)
    print("Loading test data ...")
    TEST_DATASET = PipeSegDataset(h5_path=TEST_DATA_PATH, num_point=args.npoint, block_size=args.block_size)
    print("Loading validation data ...")
    VAL_DATASET = PipeSegDataset(h5_path=VAL_DATA_PATH, num_point=args.npoint, block_size=args.block_size)

    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valDataLoader = DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # 计算类别权重
    class_freq = np.zeros(NUM_CLASSES)
    for labels in TRAIN_DATASET.labels:
        tmp, _ = np.histogram(labels, bins=range(NUM_CLASSES + 1))
        class_freq += tmp
    class_weights = 1.0 / (class_freq + 1e-10)
    class_weights = class_weights / np.sum(class_weights)
    weights = torch.FloatTensor(class_weights).cuda()

    # 加载模型
    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(NUM_CLASSES).cuda()
    criterion = torch.nn.NLLLoss(weight=weights).cuda()

    # 加载预训练模型或从头开始
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))

        # 训练
        classifier.train()
        train_loss = 0.0
        count = 0.0
        
        for points, target in tqdm(trainDataLoader, total=len(trainDataLoader), desc=f'Training Epoch {epoch}'):
            optimizer.zero_grad()
            
            points = points.float().cuda()
            target = target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, _ = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            count += 1

        train_loss = train_loss / count
        log_string('Train loss: %f' % train_loss)

        # 保存检查点
        if epoch % 5 == 0:
            savepath = str(checkpoints_dir) + f'/model_epoch_{epoch}.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        # 验证
        with torch.no_grad():
            classifier.eval()
            val_loss = 0.0
            count = 0
            
            for points, target in tqdm(valDataLoader, total=len(valDataLoader), desc=f'Validation Epoch {epoch}'):
                points = points.float().cuda()
                target = target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, _ = classifier(points)
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target)

                val_loss += loss.item()
                count += 1

            val_loss = val_loss / count
            log_string('Validation loss: %f' % val_loss)

            if val_loss < best_iou:
                best_iou = val_loss
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('New best model! Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'loss': val_loss,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        global_epoch += 1

if __name__ == '__main__':
    args = parse_args()
    main(args)
