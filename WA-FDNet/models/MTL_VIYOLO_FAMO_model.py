import time
import cv2
from pathlib import Path
import functools
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.parallel import DataParallel, DistributedDataParallel
import os
from os import path as osp
import numpy as np
from basicsr.utils import get_root_logger,tensor2img,imwrite
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.models.base_model import BaseModel
from basicsr.utils.registry import MODEL_REGISTRY
from core.Metric_fusion.eval_one_method import evaluation_one_method_fast, evaluation_one_method_test
import core.weights_init
from tqdm import tqdm
from ultralytics.utils import ops
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils import LOGGER
from losses.MTL_VI_loss import MTL_fuse_v8DetectionLoss
from scripts.util import RGB2YCrCb, YCrCb2RGB
from optim import *

@MODEL_REGISTRY.register()
class MTL_VIYOLO_FAMO(BaseModel):
    def __init__(self, opt):
        super(MTL_VIYOLO_FAMO, self).__init__(opt)
        logger = get_root_logger()
        # define network and load pretrained models
        self.net_encoder =  build_network(opt['network_Encoder'])
        self.net_encoder = self.model_to_device(self.net_encoder)
        self.net_fusion =  build_network(opt['network_Fusion'])
        self.net_fusion = self.model_to_device(self.net_fusion)
        self.net_detection =  build_network(opt['network_Detection'])
        self.net_detection = self.model_to_device(self.net_detection)
        if opt['logger'].get('print_net', False):
            self.print_network(self.net_encoder)
            self.print_network(self.net_encoder)
            self.print_network(self.net_detection)

        if isinstance(self.net_encoder, (DataParallel, DistributedDataParallel)):
            self.net_encoder = self.net_encoder.module
        else:
            self.net_encoder = self.net_encoder
        if isinstance(self.net_fusion, (DataParallel, DistributedDataParallel)):
            self.net_fusion = self.net_fusion.module
        else:
            self.net_fusion = self.net_fusion
        if isinstance(self.net_detection, (DataParallel, DistributedDataParallel)):
            self.net_detection = self.net_detection.module
        else:
            self.net_detection = self.net_detection

        if self.is_train:
            self.init_training_settings()
        else:
            self.net_encoder.eval()
            self.net_fusion.eval()
            self.net_detection.eval()

        load_path = self.opt['path'].get('pretrain_network_VIFusionYOLO', None)
        if load_path is not None:
            self.load_network(self.net_encoder, load_path, self.opt['path'].get('strict_load_g', True), 'params_encoder')
            self.load_network(self.net_fusion, load_path, self.opt['path'].get('strict_load_g', True), 'params_fusion')
            self.load_network(self.net_detection, load_path, self.opt['path'].get('strict_load_g', True), 'params_detection')
            logger.info(f"Pretrained model is successfully loaded from {opt['path']['pretrain_network_VIFusionYOLO']}")

        self.current_iter = 0
        self.alpha=opt['train']["alpha"]
        # self._initialize_weights()  

        self.seen = 0
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])
        self.plots = False
        self.single_cls = False
        self.conf = opt['Det_labels']['conf']
        self.nc = opt['Det_labels']['nc']
        self.iou = opt['Det_labels']['iou']
        self.iou_thres = opt['Det_labels']['iou_thres']
        self.task = "detect"
        self.matrix = np.zeros((self.nc + 1, self.nc + 1)) if self.task == "detect" else np.zeros((self.nc, self.nc))
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf)
        self.jdict = []
        self.class_map = list(range(1, self.nc + 1))
        self.names = {i: name for i, name in enumerate(opt['Det_labels']['names'])}
        self.save_conf = True
        self.save_dir = Path(opt['path']['experiments_root'] if 'experiments_root' in opt['path'] else opt['path']['results_root'] )
        self.metrics = DetMetrics(save_dir=self.save_dir)
        self.metrics.names = self.names
        self.metrics.plot = self.plots
 
    def transfer_weights(self, small_model, large_model):  
        small_model_state_dict = small_model.state_dict()  
        large_model_state_dict = large_model.state_dict()   
         
        for name, param in small_model_state_dict.items():  
            if name in large_model_state_dict:  
                large_model_state_dict[name].data.copy_(param.data)  

    def _initialize_weights(self):  
        logger = get_root_logger()
        weights_init = functools.partial(core.weights_init.weights_init_normal)
        self.netfusion.apply(weights_init)
        logger.info(f"Initialize weights of model")
    
    def init_training_settings(self):
        self.net_encoder.train()
        self.net_fusion.train()
        self.net_detection.train()
        self.loss_dict_all = OrderedDict()
        self.loss_dict_all['loss_all'] = []
        self.loss_dict_all['loss_det'] = []
        self.loss_dict_all['loss_fusion'] = []
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.loss_FD = build_loss(self.opt['train']['Loss_FD']).to(self.device)
        self.balancer = FAMO(n_tasks=2, device="cuda", gamma=0.001)

    def setup_optimizers(self):
        train_opt = self.opt['train']

        optim_netencoder_params = list(self.net_encoder.parameters())
        optim_params_g_netencoder = [{  # add normal params first
            'params': optim_netencoder_params,
            'lr': train_opt['optimizer_encoder']['lr']
        }]
        optim_type = train_opt['optimizer_encoder'].pop('type')
        lr = train_opt['optimizer_encoder']['lr']
        self.optimizer_g_netencoder = self.get_optimizer(optim_type, optim_params_g_netencoder, lr)
        self.optimizers.append(self.optimizer_g_netencoder)

        optim_netfusion_params = list(self.net_fusion.parameters())
        optim_params_g_netfusion = [{  # add normal params first
            'params': optim_netfusion_params,
            'lr': train_opt['optimizer_fusion']['lr']
        }]
        optim_type = train_opt['optimizer_fusion'].pop('type')
        lr = train_opt['optimizer_fusion']['lr']
        self.optimizer_g_netfusion = self.get_optimizer(optim_type, optim_params_g_netfusion, lr)
        self.optimizers.append(self.optimizer_g_netfusion)

        optim_netdetection_params = list(self.net_detection.parameters())
        optim_params_g_netdetection = [{  # add normal params first
            'params': optim_netdetection_params,
            'lr': train_opt['optimizer_detection']['lr']
        }]
        optim_type = train_opt['optimizer_detection'].pop('type')
        lr = train_opt['optimizer_detection']['lr']
        self.optimizer_g_netdetection = self.get_optimizer(optim_type, optim_params_g_netdetection, lr)
        self.optimizers.append(self.optimizer_g_netdetection)
     
    # Feeding all data to the DF model
    def feed_data(self, train_data):
        self.batch = train_data
        self.data = {}
        self.data['vi'] = self.set_device(self.batch['img'])
        self.data['ir'] = self.set_device(self.batch['an_img'])
    
    # Optimize the parameters of the DF model
    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        self.current_iter = current_iter
        self.optimizer_g_netencoder.zero_grad()
        self.optimizer_g_netfusion.zero_grad()
        self.optimizer_g_netdetection.zero_grad()
        loss_dict = OrderedDict() 
        x11,x12,x21,x22,x31,x32,x41f,x42f,x41,x42,x51,x52,x61,x62=self.net_encoder(self.data)
        self.pred_img = self.net_fusion(x11,x12,x21,x22,x31,x32,x41f,x42f,x51,x52,x61,x62)
        self.preds = self.net_detection(x41,x42,x51,x52,x61,x62)
        loss_det, loss_ss  = self.loss_FD(self.alpha, image_vis=self.data['vi'], image_ir=self.data['ir'],  generate_img=self.pred_img, preds=self.preds, batch =self.batch)
        loss = torch.stack((loss_ss.mean(), loss_det.mean())).to(self.device)
        self.balancer.backward(loss)
        with torch.no_grad():
            x11,x12,x21,x22,x31,x32,x41f,x42f,x41,x42,x51,x52,x61,x62=self.net_encoder(self.data)
            self.pred_img = self.net_fusion(x11,x12,x21,x22,x31,x32,x41f,x42f,x51,x52,x61,x62)
            self.preds = self.net_detection(x41,x42,x51,x52,x61,x62)
            loss_det, loss_ss = self.loss_FD(self.alpha, image_vis=self.data['vi'], image_ir=self.data['ir'],  generate_img=self.pred_img, preds=self.preds, batch =self.batch)
            new_loss = torch.stack((loss_ss.mean(), loss_det.mean())).to(self.device)
            self.balancer.update(new_loss)
        # x11,x12,x21,x22,x31,x32,x41f,x42f,x41,x42,x51,x52,x61,x62=self.net_encoder(self.data)
        # self.pred_img = self.net_fusion(x11,x12,x21,x22,x31,x32,x41f,x42f,x51,x52,x61,x62)
        # self.preds = self.net_detection(x41,x42,x51,x52,x61,x62)
        # loss_det, loss_ss  = self.loss_FD(self.alpha, image_vis=self.data["vi"], image_ir=self.data["ir"],  generate_img=self.pred_img, preds=self.preds, batch =self.batch)
        # # logger.info(f"[info] iter {current_iter:3d} | avg loss {loss.mean().item():.4f}")
        # loss_all = loss_det + loss_ss
        # loss_all.backward()
        self.optimizer_g_netencoder.step()
        self.optimizer_g_netfusion.step()
        self.optimizer_g_netdetection.step()

        loss_dict['loss_all'] = loss_ss + loss_det
        loss_dict['loss_fusion'] = loss_ss
        loss_dict['loss_det'] = loss_det
        loss_dict = self.set_device(loss_dict)
        self.log_dict = self.reduce_loss_dict(loss_dict)
        for name, value in self.log_dict.items():
            self.loss_dict_all[name].append(value)

    # Testing on given data
    def test(self):
        self.net_encoder.eval()
        self.net_fusion.eval()
        self.net_detection.eval()
        with torch.no_grad():
            x11,x12,x21,x22,x31,x32,x41f,x42f,x41,x42,x51,x52,x61,x62=self.net_encoder(self.data)
            self.pred_img = self.net_fusion(x11,x12,x21,x22,x31,x32,x41f,x42f,x51,x52,x61,x62)
            self.preds = self.net_detection(x41,x42,x51,x52,x61,x62)
        self.net_encoder.train()
        self.net_fusion.train()
        self.net_detection.train()

        return self.pred_img, self.preds
    
    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    if key == 'label':
                        x[key] = item.to(self.device, dtype=torch.int)
                    else:
                        if isinstance(x[key], list):
                            for item_1 in x[key]:
                                if item_1 is not None and not isinstance(item_1, str) :
                                    item_1 = item_1.to(self.device, dtype=torch.float)
                        else:
                            x[key] = item.to(self.device, dtype=torch.float)
        elif isinstance(x, list):
            for item in x:
                if item is not None:
                    item = item.to(self.device, dtype=torch.float)
        else:
            x = x.to(self.device, dtype=torch.float)
        return x
    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = self.opt['datasets']['val'].get('type')
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)
        self.crop_size = self.opt['datasets']['val'].get('crop_size', None)
        self.data_dir = self.opt['val'].get('data_dir')
        self.dataset_name = self.opt['val'].get('name')
        self.save_plot_det = self.opt['val'].get('plot_det', False)
        self.save_img_y = self.opt['val'].get('save_img_y', False)
        self.save_json = True
        self.save_txt = self.opt['val']['save_txt']
        self.names = self.opt['Det_labels']['names']
        self.conf_test = self.opt['Det_labels']['conf_save']
        self.init_metrics()
        self.det = MTL_fuse_v8DetectionLoss(self.device)
        logger = get_root_logger()

        if with_metrics and not hasattr(self, 'metric_results'):  # only execute in the first run
            self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
        # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
        self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # for idx, val_data in enumerate(dataloader):
        for idx, val_data in enumerate(dataloader):
            im_name = val_data['im_name']
            self.feed_data(val_data)
            self.pred_img, self.preds = self.test()
            self.preds = self.postprocess(self.preds)
            self.update_metrics(self.preds, self.batch)
            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals['pred_img'].detach(), min_max=(0, 1))
            # sr_img_y = tensor2img(visuals['pred_img_y'].detach(), min_max=(0, 1))
            metric_data['img'] = sr_img
            # tentative for out of GPU memory
            torch.cuda.empty_cache()

            n=len(visuals['pred_img'])
            for i in range(n):
                sr_img = tensor2img(visuals['pred_img'][i].detach(), min_max=(0, 1))
                sr_img_y = tensor2img(visuals['pred_img_y'][i].detach(), min_max=(0, 1))
                img_name = osp.splitext(im_name[i])[0]
                preds = self.preds[i]
                if save_img:
                    if self.opt['is_train']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                f'{img_name}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter),
                                                    f'{img_name}.png')
                    imwrite(sr_img, save_img_path)
                if self.save_img_y:
                    if self.opt['is_train']:
                        save_img_path_y = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter)+'_y',
                                                f'{img_name}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_path_y = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter)+'_y',
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_path_y = osp.join(self.opt['path']['visualization'], dataset_name, str(current_iter)+'_y',
                                                f'{img_name}.png')
                    imwrite(sr_img_y, save_img_path_y)
                if self.save_plot_det:
                    self.label_img = self.plot_det(preds, sr_img)
                    if self.opt['is_train']:
                        save_img_label_path = osp.join(self.opt['path']['visualization'], dataset_name + '_label', str(current_iter),
                                                f'{img_name}.png')
                    else:
                        if self.opt['val']['suffix']:
                            save_img_label_path = osp.join(self.opt['path']['visualization'], dataset_name + '_label', str(current_iter),
                                                    f'{img_name}_{self.opt["val"]["suffix"]}.png')
                        else:
                            save_img_label_path = osp.join(self.opt['path']['visualization'], dataset_name + '_label', str(current_iter),
                                                    f'{img_name}.png')
                    imwrite(self.label_img, save_img_label_path)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            stats = self.get_stats()
            self.speed = None
            self.finalize_metrics()
            log_str = ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")
            pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
            log_str1 = pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results())
            logger.info(log_str)
            logger.info(log_str1)
            for i, c in enumerate(self.metrics.ap_class_index):
                logger.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )
            self.current_det_metric = self.metrics.mean_results()[2] + self.metrics.mean_results()[3]
            metric_f=['mAP50', 'mAP50-95']
            for index, metric in enumerate(metric_f):
                self.metric_results[metric] = self.metrics.mean_results()[2+index]
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            # self.print_results()
            r_dir, f_name = os.path.split(save_img_path) 
            save_dir =  osp.join(self.opt['path']['visualization'], 'metric')
            os.makedirs(save_dir, exist_ok=True)
            metric_r = evaluation_one_method_fast(dataset_name=self.dataset_name, data_dir=self.data_dir, result_dir=r_dir, save_dir= save_dir+ f'/{current_iter}' + '_metric_VI.xlsx', Method='VI' , with_mean=True, crop_size=(self.crop_size, self.crop_size))
            metric_f=['EN', 'SF', 'AG', 'SD', 'CC', 'SCD', 'MSE', 'PSNR', 'Qabf', 'Nabf']
            for index, metric in enumerate(metric_f):
                self.metric_results[metric] = metric_r[index][0]
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self.current_fusion_metric = self.metric_results['Qabf']
            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
            log_str = f'Validation {dataset_name}\n'
            for metric, value in self.metric_results.items():
                log_str += f'\t # {metric}: {value:.4f}'
                if hasattr(self, 'best_metric_results'):
                    log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                                f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
                log_str += '\n'

            logger = get_root_logger()
            logger.info(log_str)
            if tb_logger:
                for metric, value in self.metric_results.items():
                    tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    # Get current log
    def get_current_iter_log(self):
        #self.update_loss()
        return self.log_dict
    
    def get_current_log(self):
        for name, value in self.log_dict.items():
            self.log_dict[name] = np.average(self.loss_dict_all[name])
        return self.log_dict
    
    def save_current_log_img(self):
        visuals = self.get_current_visuals()
        grid_img = torch.cat((visuals['pred_img'].detach(),
                                    visuals['gt_vi'],
                                    visuals['gt_ir']), dim=0)
        grid_img = tensor2img(grid_img, min_max=(0, 1))
        save_img_path = os.path.join(self.opt['path']['visualization'],'img_fused_iter_{}.png'.format(self.current_iter))
        imwrite(grid_img, save_img_path)

    # Get current visuals
    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['pred_img_y'] = self.pred_img
        out_dict['gt_vi'] = self.data["vi"]
        out_dict['gt_ir'] = self.data["ir"]
        out_dict['pred_img'] = YCrCb2RGB(torch.cat((out_dict['pred_img_y'], RGB2YCrCb(out_dict['gt_vi'])[:,1:,:,:]), dim=1)) 
        return out_dict
    
    def get_current_model_score(self):
        return self.current_det_metric if hasattr(self, 'current_det_metric') else 0
    
    def save(self, epoch, current_iter):
        self.save_network([self.net_encoder, self.net_fusion, self.net_detection], 'net_fe_g', current_iter, param_key=['params_encoder','params_fusion','params_detection'])
        self.save_training_state(epoch, current_iter)
    
    def save_best(self, current_iter):
        logger = get_root_logger()
        self.save_network([self.net_encoder, self.net_fusion, self.net_detection], 'net_best', current_iter, param_key=['params_encoder','params_fusion','params_detection'])
        logger.info(f"Saving new best-model")

    def remove(self, pre_iter):
        logger = get_root_logger()
        if pre_iter > 0:
            model_path = os.path.join(self.opt['path']['models'], f"net_fe_g_{pre_iter}.pth")
            state_path = os.path.join(self.opt['path']['training_states'], f'{pre_iter}.state')
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted old model file: {model_path}")
            else:
                logger.info(f"Old model file not found: {model_path}")
            if os.path.exists(state_path):
                os.remove(state_path)
                logger.info(f"Deleted old state file: {state_path}")
            else:
                logger.info(f"Old state file not found: {state_path}")
    
    def remove_best(self, best_iter):
        logger = get_root_logger()
        if best_iter > 0:
            model_path = os.path.join(self.opt['path']['models'], f"net_best_{best_iter}.pth")
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"Deleted old best-model file: {model_path}")
            else:
                logger.info(f"Old best-model file not found: {model_path}")

    def init_metrics(self):
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def plot_det(self, preds, img):
        # pred = preds[0]
        # img = np.stack((img,) * 3, axis=-1)  
        for (x1, y1, x2, y2, conf, cls) in preds.cpu().detach().numpy():
            if conf >= self.conf_test:   
                color = (0, 0, 255) 
                color1 = (255, 255, 255) 
                x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  

                label = f"{self.names[int(cls)]}: {conf:.2f}"  
                background_tl = (x1, y1-18)   
                background_br = (x1+93, y1)
 
                cv2.rectangle(img, background_tl, background_br, color, thickness=-1) 
                cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color1, 1, lineType=cv2.LINE_AA)
                
        return img

    def update_metrics(self, preds, batch):
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[torch.Tensor]): List of predictions from the model.
            batch (dict): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            cls = cls.to(self.device)
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
            if self.plots:
                self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.save_txt:
                self.save_one_txt(
                    predn,
                    self.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" /str(self.current_iter)/ f"{Path(batch['im_file'][si]).stem}.txt",
                )
    
    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[torch.Tensor]): Processed predictions after NMS.
        """
        return ops.non_max_suppression(
            preds,
            self.conf,
            self.iou,
            labels=[],
            nc=self.nc,
            multi_label=True,
            agnostic=False,
            max_det=300,
            end2end=False,
            rotated=False
        )

    def _prepare_batch(self, si, batch):
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (dict): Batch data containing images and annotations.

        Returns:
            (dict): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = bbox.to(self.device)
            cls = cls.to(self.device)
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            bbox.to('cpu')
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}
    def _prepare_pred(self, pred, pbatch):
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (torch.Tensor): Model predictions.
            pbatch (dict): Prepared batch information.

        Returns:
            (torch.Tensor): Prepared predictions in native space.
        """
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn
    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape (N, 6) representing detections where each detection is
                (x1, y1, x2, y2, conf, class).
            gt_bboxes (torch.Tensor): Tensor of shape (M, 4) representing ground-truth bounding box coordinates. Each
                bounding box is of the format: (x1, y1, x2, y2).
            gt_cls (torch.Tensor): Tensor of shape (M,) representing target class indices.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)
    
    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """
        Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        # matches = matches[matches[:, 2].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)
    
    def save_one_txt(self, predn, save_conf, shape, file):
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            save_conf (bool): Whether to save confidence scores.
            shape (tuple): Shape of the original image.
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=predn[:, :6],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn, filename):
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (torch.Tensor): Predictions in the format (x1, y1, x2, y2, conf, class).
            filename (str): Image filename.
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )
    def finalize_metrics(self, *args, **kwargs):
        """
        Set final values for metrics speed and confusion matrix.

        Args:
            *args (Any): Variable length argument list.
            **kwargs (Any): Arbitrary keyword arguments.
        """
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        
    def get_stats(self):
        """
        Calculate and return metrics statistics.

        Returns:
            (dict): Dictionary containing metrics results.
        """
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats):
            self.metrics.process(**stats, on_plot=self.on_plot)
        return self.metrics.results_dict

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}
    
    def print_results(self):
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if not self.is_train and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )