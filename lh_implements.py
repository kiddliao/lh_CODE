import datetime
import os
import argparse
import traceback
import time
import torch
import yaml
import numpy as np
#base
def get_args():
    parser = argparse.ArgumentParser('base detector')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-n', '--num_workers', type=int, default=16, help='num_workers of dataloader')
    parser.add_argument('--batch_size', type=int, default=6, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--alpha', type=float, default=0.25, help='BN')
    parser.add_argument('--gamma', type=float, default=1.5, help='BN')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=100, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='../datasets/coco_flir/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='saved')
    parser.add_argument('--lr_num_decay', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False, help='whether visualize the predicted boxes of trainging, '
                                                                  'the output images will be in test/')
    args = parser.parse_args()
    return args

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)
    
def train(opt):
    params = Params(f'projects/{opt.project}.yml') #读取各个路径和anchor和class

    if params.num_gpus == 0:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    else:
        torch.manual_seed(42)

    opt.saved_path = opt.saved_path + f'/{params.project_name}/'
    opt.log_path = opt.log_path + f'/{params.project_name}/tensorboard/'
    os.makedirs(opt.log_path, exist_ok=True)
    os.makedirs(opt.saved_path, exist_ok=True)
    #读取数据集
    training_params = {'batch_size': opt.batch_size,
                       'shuffle': True,
                       'drop_last': True,
                       'collate_fn': collater,
                       'num_workers': opt.num_workers}

    val_params = {'batch_size': opt.batch_size,
                  'shuffle': False,
                  'drop_last': True,
                  'collate_fn': collater,
                  'num_workers': opt.num_workers}

    input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536]
    training_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.train_set,
                               transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                             Augmenter(),
                                                             Resizer(input_sizes[opt.compound_coef])]))
    training_generator = DataLoader(training_set, **training_params)

    val_set = CocoDataset(root_dir=os.path.join(opt.data_path, params.project_name), set=params.val_set,
                          transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                        Resizer(input_sizes[opt.compound_coef])]))
    val_generator = DataLoader(val_set, **val_params)
    print(val_set[0])
    #定义网络
    model = EfficientDetBackbone(num_classes=len(params.obj_list), compound_coef=opt.compound_coef,
                                 ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))

    # load last weights
    if opt.load_weights is not None:
        if opt.load_weights.endswith('.pth'):
            weights_path = opt.load_weights
        else:
            weights_path = get_last_weights(opt.saved_path)
        try:
            last_step = int(os.path.basename(weights_path).split('_')[-1].split('.')[0])
        except:
            last_step = 0

        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print(
                '[Warning] Don\'t panic if you see this, this might be because you load a pretrained weights with different number of classes. The rest of the weights should be loaded already.')

        print(f'[Info] loaded weights: {os.path.basename(weights_path)}, resuming checkpoint from step: {last_step}')
    else:
        last_step = 0
        print('[Info] initializing weights...')
        init_weights(model)

    # freeze backbone if train head_only
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')

    # https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
    # apply sync_bn when using multiple gpu and batch_size per gpu is lower than 4
    #  useful when gpu memory is limited.
    # because when bn is disable, the training will be very unstable or slow to converge,
    # apply sync_bn can solve it,
    # by packing all mini-batch across all gpus as one batch and normalize, then send it back to all gpus.
    # but it would also slow down the training by a little bit.
    if params.num_gpus > 1 and opt.batch_size // params.num_gpus < 4:
        model.apply(replace_w_sync_bn)
        use_sync_bn = True
    else:
        use_sync_bn = False
    #创建日志文件夹
    writer = SummaryWriter(opt.log_path + f'/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}/')

    # warp the model with loss function, to reduce the memory usage on gpu0 and speedup
    model = ModelWithLoss(model, debug=opt.debug)

    if params.num_gpus > 0:
        model = model.cuda()
        if params.num_gpus > 1:
            model = CustomDataParallel(model, params.num_gpus)
            if use_sync_bn:
                patch_replication_callback(model)

    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    #学习率衰减
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    epoch = 0
    best_loss = 1e5
    best_epoch = 0
    step = max(0, last_step)
    model.train()

    num_iter_per_epoch = len(training_generator)

    #进行三次学习率衰减 每当15个epochloss没有下降就进行一次衰减
    num_decay=opt.lr_num_decay
    print(f'num_dacay={num_decay}')

    try:
        for epoch in range(opt.num_epochs):
            last_epoch = step // num_iter_per_epoch
            if epoch < last_epoch:#恢复训练
                continue

            epoch_loss = []
            progress_bar = tqdm(training_generator)
            for iter, data in enumerate(progress_bar):
                if iter < step - last_epoch * num_iter_per_epoch:
                    progress_bar.update()
                    continue
                try:
                    imgs = data['img']
                    annot = data['annot']

                    if params.num_gpus == 1:
                        # if only one gpu, just send it to cuda:0
                        # elif multiple gpus, send it to multiple gpus in CustomDataParallel, not here
                        imgs = imgs.cuda()
                        annot = annot.cuda()

                    optimizer.zero_grad()
                    cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss = cls_loss + reg_loss
                    if loss == 0 or not torch.isfinite(loss):
                        continue

                    loss.backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                    optimizer.step()

                    epoch_loss.append(float(loss))

                    progress_bar.set_description(
                        'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                            step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                            reg_loss.item(), loss.item()))
                    writer.add_scalars('Loss', {'train': loss}, step)
                    writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
                    writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

                    # log learning_rate
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', current_lr, step)

                    step += 1

                    if step % opt.save_interval == 0 and step > 0:
                        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
                        print('checkpoint...')

                except Exception as e:
                    print('[Error]', traceback.format_exc())
                    print(e)
                    continue
            scheduler.step(np.mean(epoch_loss))
            if epoch % opt.val_interval == 0:
                model.eval()
                loss_regression_ls = []
                loss_classification_ls = []
                for iter, data in enumerate(val_generator):
                    with torch.no_grad():
                        imgs = data['img']
                        annot = data['annot']

                        if params.num_gpus == 1:
                            imgs = imgs.cuda()
                            annot = annot.cuda()

                        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                        cls_loss = cls_loss.mean()
                        reg_loss = reg_loss.mean()

                        loss = cls_loss + reg_loss
                        if loss == 0 or not torch.isfinite(loss):
                            continue

                        loss_classification_ls.append(cls_loss.item())
                        loss_regression_ls.append(reg_loss.item())

                cls_loss = np.mean(loss_classification_ls)
                reg_loss = np.mean(loss_regression_ls)
                loss = cls_loss + reg_loss

                print(
                    'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                        epoch, opt.num_epochs, cls_loss, reg_loss, loss))
                writer.add_scalars('Loss', {'val': loss}, step)
                writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
                writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)

                if loss + opt.es_min_delta < best_loss:
                    best_loss = loss
                    best_epoch = epoch

                    save_checkpoint(model, f'best_efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')

                model.train()
                           
                # Early stopping
                if epoch - best_epoch > opt.es_patience > 0:
                    if num_decay:
                        num_decay-=1
                        
                        optimizer.param_groups[0]['lr']/=10
                        x=optimizer.param_groups[0]['lr']
                        print(f'num_decay:{num_decay},learning_rate:{x}')
                        best_loss = loss
                        best_epoch = epoch

                    else:
                        print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                        break
    except KeyboardInterrupt:
        save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        writer.close()
    writer.close()

def save_checkpoint(model, name):
    if isinstance(model, CustomDataParallel):
        torch.save(model.module.model.state_dict(), os.path.join(opt.saved_path, name))
    else:
        torch.save(model.model.state_dict(), os.path.join(opt.saved_path, name))

if __name__ == '__main__':
    opt = get_args()
    start=time.time()
    train(opt)
    end=time.time()
    print(f'训练用时{round((end-start)/3600,1)}h')

#手动调整
#1早停法和早停法搭配学习率衰减(当lr_num_decay设置为0就是单纯的早停法,如果不是,每当es_patience个epoch的loss没有下降,学习率衰减10倍然后bestepoch为当前epoch)
epoch = 0
best_loss = 1e5
best_epoch = 0
step = max(0, last_step)
model.train()
num_iter_per_epoch = len(training_generator)
num_decay=opt.lr_num_decay
print(f'num_dacay={num_decay}')
for epoch in range(opt.num_epochs):
    epoch_loss = []
    progress_bar = tqdm(training_generator)
    for iter, data in enumerate(progress_bar):
        pass
        optimizer.zero_grad()
        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue
        loss.backward()

        optimizer.step()
        epoch_loss.append(float(loss))
        progress_bar.set_description(
            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                reg_loss.item(), loss.item()))
        writer.add_scalars('Loss', {'train': loss}, step)
        writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

        # log learning_rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, step)

        step += 1

        if step % opt.save_interval == 0 and step > 0:
            save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
            print('checkpoint...')

    if epoch % opt.val_interval == 0:
        model.eval()
        loss_regression_ls = []
        loss_classification_ls = []
        for iter, data in enumerate(val_generator):
            with torch.no_grad():
                pass
                cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue
                loss_classification_ls.append(cls_loss.item())
                loss_regression_ls.append(reg_loss.item())
        cls_loss = np.mean(loss_classification_ls)
        reg_loss = np.mean(loss_regression_ls)
        loss = cls_loss + reg_loss
        print(
            'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                epoch, opt.num_epochs, cls_loss, reg_loss, loss))
        writer.add_scalars('Loss', {'val': loss}, step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
        if loss + opt.es_min_delta < best_loss: #找到best_loss
            best_loss = loss
            best_epoch = epoch
            save_checkpoint(model, f'best_efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
        model.train()  
        # Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
            if num_decay:
                num_decay-=1
                
                optimizer.param_groups[0]['lr']/=10
                x=optimizer.param_groups[0]['lr']
                print(f'num_decay:{num_decay},learning_rate:{x}')
                best_loss = loss
                best_epoch = epoch
            else:
                print('[Info] Stop training at epoch {}. The lowest loss achieved is {}'.format(epoch, best_loss))
                break


#2step学习率衰减
decay_step = {40000: 0.1, 45000, 0.1}
epoch = 0
best_loss = 1e5
best_epoch = 0
step = max(0, last_step)
model.train()
print(f'num_dacay={num_decay}')
for epoch in range(opt.num_epochs):
    epoch_loss = []
    progress_bar = tqdm(training_generator)
    for iter, data in enumerate(progress_bar):
        pass
        optimizer.zero_grad()
        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue
        loss.backward()

        optimizer.step()
        epoch_loss.append(float(loss))
        progress_bar.set_description(
            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                reg_loss.item(), loss.item()))
        writer.add_scalars('Loss', {'train': loss}, step)
        writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

        # log learning_rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, step)

        step += 1
        #step学习率衰减
        if step in decay_step.keys():
            optimizer.param_groups[0]['lr'] *= decay_step[step]

        if step % opt.save_interval == 0 and step > 0:
            save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
            print('checkpoint...')

    if epoch % opt.val_interval == 0:
        model.eval()
        loss_regression_ls = []
        loss_classification_ls = []
        for iter, data in enumerate(val_generator):
            with torch.no_grad():
                pass
                cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue
                loss_classification_ls.append(cls_loss.item())
                loss_regression_ls.append(reg_loss.item())
        cls_loss = np.mean(loss_classification_ls)
        reg_loss = np.mean(loss_regression_ls)
        loss = cls_loss + reg_loss
        print(
            'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                epoch, opt.num_epochs, cls_loss, reg_loss, loss))
        writer.add_scalars('Loss', {'val': loss}, step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
        

#api调整学习率
decay_step = {40000: 0.1, 45000, 0.1}
epoch = 0
best_loss = 1e5
best_epoch = 0
step = max(0, last_step)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
#1 lambda
lambda1 = lambda epoch: np.sin(epoch) / epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
#2 过几个epoch衰减gamma
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
#3 三段式step学习率衰减 进入范围衰减一次 离开范围衰减一次
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 80], gamma=0.1)
#4 指数衰减 每个epoch学习率都衰减gamma
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
#5 余弦退火 T_max 对应1/2个cos周期所对应的epoch数值 eta_min 为最小的lr值，默认为0
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
#6 早停加学习率衰减 也就是loss不再降低或acc不再提高之后降低学习率
# mode：'min'模式检测metric是否不再减小，'max'模式检测metric是否不再增大 
# factor: 触发条件后lr*=factor
# patience:不再减小（或增大）的累计次数
# verbose:触发条件后print
# threshold:只关注超过阈值的显著变化
# threshold_mode:有rel和abs两种阈值计算模式，rel规则：max模式下如果超过best(1+threshold)为显著，min模式下如果低于best(1-threshold)为显著；abs规则：max模式下如果超过best+threshold为显著，min模式下如果低于best-threshold为显著
# cooldown：触发一次条件后，等待一定epoch再进行检测，避免lr下降过速
# min_lr:最小的允许lr
# eps:如果新旧lr之间的差异小与1e-8，则忽略此次更新
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True, factor=0.1)
#7 gradual warmup https://github.com/ildoonet/pytorch-gradual-warmup-lr
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
class GradualWarmupScheduler(_LRScheduler):
    """ Gradually warm-up(increasing) learning rate in optimizer.
    Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        multiplier: target learning rate = base lr * multiplier if multiplier > 1.0. if multiplier = 1.0, lr starts from 0 and ends up with the base_lr.
        total_epoch: target learning rate is reached at total_epoch, gradually
        after_scheduler: after target_epoch, use this scheduler(eg. ReduceLROnPlateau)
    """
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater thant or equal to 1.')
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        super(GradualWarmupScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

    def step_ReduceLROnPlateau(self, metrics, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch if epoch != 0 else 1  # ReduceLROnPlateau is called at the end of epoch, whereas others are called at beginning
        if self.last_epoch <= self.total_epoch:
            warmup_lr = [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]
            for param_group, lr in zip(self.optimizer.param_groups, warmup_lr):
                param_group['lr'] = lr
        else:
            if epoch is None:
                self.after_scheduler.step(metrics, None)
            else:
                self.after_scheduler.step(metrics, epoch - self.total_epoch)

    def step(self, epoch=None, metrics=None):
        if type(self.after_scheduler) != ReduceLROnPlateau:
            if self.finished and self.after_scheduler:
                if epoch is None:
                    self.after_scheduler.step(None)
                else:
                    self.after_scheduler.step(epoch - self.total_epoch)
                self._last_lr = self.after_scheduler.get_last_lr()
            else:
                return super(GradualWarmupScheduler, self).step(epoch)
        else:
            self.step_ReduceLROnPlateau(metrics, epoch)
# scheduler_warmup is chained with schduler_steplr
scheduler_steplr = torch.optim.lr_scheduler.StepLR(optim, step_size=10, gamma=0.1)
# 10个epoch后开始下一个学习率衰减策略
scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=10, after_scheduler=scheduler_steplr)


print(f'num_dacay={num_decay}')
for epoch in range(opt.num_epochs):
    epoch_loss = []
    progress_bar = tqdm(training_generator)
    for iter, data in enumerate(progress_bar):
        pass
        optimizer.zero_grad()
        cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
        cls_loss = cls_loss.mean()
        reg_loss = reg_loss.mean()
        loss = cls_loss + reg_loss
        if loss == 0 or not torch.isfinite(loss):
            continue
        loss.backward()

        optimizer.step()
        epoch_loss.append(float(loss))
        progress_bar.set_description(
            'Step: {}. Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                step, epoch, opt.num_epochs, iter + 1, num_iter_per_epoch, cls_loss.item(),
                reg_loss.item(), loss.item()))
        writer.add_scalars('Loss', {'train': loss}, step)
        writer.add_scalars('Regression_loss', {'train': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'train': cls_loss}, step)

        # log learning_rate
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('learning_rate', current_lr, step)

        step += 1
        #step学习率衰减
        if step in decay_step.keys():
            optimizer.param_groups[0]['lr'] *= decay_step[step]

        if step % opt.save_interval == 0 and step > 0:
            save_checkpoint(model, f'efficientdet-d{opt.compound_coef}_{epoch}_{step}.pth')
            print('checkpoint...')
    scheduler.step()
    if epoch % opt.val_interval == 0:
        model.eval()
        loss_regression_ls = []
        loss_classification_ls = []
        for iter, data in enumerate(val_generator):
            with torch.no_grad():
                pass
                cls_loss, reg_loss = model(imgs, annot, obj_list=params.obj_list)
                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0 or not torch.isfinite(loss):
                    continue
                loss_classification_ls.append(cls_loss.item())
                loss_regression_ls.append(reg_loss.item())
        cls_loss = np.mean(loss_classification_ls)
        reg_loss = np.mean(loss_regression_ls)
        loss = cls_loss + reg_loss
        print(
            'Val. Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                epoch, opt.num_epochs, cls_loss, reg_loss, loss))
        writer.add_scalars('Loss', {'val': loss}, step)
        writer.add_scalars('Regression_loss', {'val': reg_loss}, step)
        writer.add_scalars('Classfication_loss', {'val': cls_loss}, step)
        
