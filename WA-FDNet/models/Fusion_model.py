import functools
from collections import OrderedDict
import torch
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
from scripts.util import RGB2YCrCb, YCrCb2RGB
from optim import *

@MODEL_REGISTRY.register()
class VIFusion(BaseModel):
    def __init__(self, opt):
        super(VIFusion, self).__init__(opt)
        logger = get_root_logger()
        # define network and load pretrained models
        self.net_encoder =  build_network(opt['network_Encoder'])
        self.net_encoder = self.model_to_device(self.net_encoder)
        self.print_network(self.net_encoder)
        self.net_fusion =  build_network(opt['network_Fusion'])
        self.net_fusion = self.model_to_device(self.net_fusion)
        self.print_network(self.net_fusion)

        if isinstance(self.net_encoder, (DataParallel, DistributedDataParallel)):
            self.net_encoder = self.net_encoder.module
        else:
            self.net_encoder = self.net_encoder
        if isinstance(self.net_fusion, (DataParallel, DistributedDataParallel)):
            self.net_fusion = self.net_fusion.module
        else:
            self.net_fusion = self.net_fusion

        if self.is_train:
            self.init_training_settings()
        else:
            self.net_encoder.eval()
            self.net_fusion.eval()

        load_path = self.opt['path'].get('pretrain_network_VIFusion', None)
        if load_path is not None:
            self.load_network(self.net_encoder, load_path, self.opt['path'].get('strict_load_g', True), 'params_encoder')
            self.load_network(self.net_fusion, load_path, self.opt['path'].get('strict_load_g', True), 'params_fusion')
            logger.info(f"Pretrained model is successfully loaded from {opt['path']['pretrain_network_VIFusion']}")

        self.current_iter = 0
        self.alpha=opt['train']["alpha"]
        # self._initialize_weights()  
 
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
        self.loss_dict_all = OrderedDict()
        self.loss_dict_all['loss_fusion'] = []
        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()
        self.loss_fusion = build_loss(self.opt['train']['Loss']).to(self.device)

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
     
    # Feeding all data to the DF model
    def feed_data(self, train_data):
        self.data = self.set_device(train_data)
    
    # Optimize the parameters of the DF model
    def optimize_parameters(self, current_iter):
        logger = get_root_logger()
        self.current_iter = current_iter
        self.optimizer_g_netencoder.zero_grad()
        self.optimizer_g_netfusion.zero_grad()
        loss_dict = OrderedDict() 

        x11,x12,x21,x22,x31,x32,x41f,x42f=self.net_encoder(self.data)
        self.pred_img = self.net_fusion(x11,x12,x21,x22,x31,x32,x41f,x42f)
        loss_fusion  = self.loss_fusion(self.alpha, image_vis=self.data["vi"], image_ir=self.data["ir"],  generate_img=self.pred_img)
        # logger.info(f"[info] iter {current_iter:3d} | avg loss {loss.mean().item():.4f}")
        loss_fusion.backward()

        self.optimizer_g_netencoder.step()
        self.optimizer_g_netfusion.step()

        loss_dict['loss_fusion'] = loss_fusion
        loss_dict = self.set_device(loss_dict)
        self.log_dict = self.reduce_loss_dict(loss_dict)
        for name, value in self.log_dict.items():
            self.loss_dict_all[name].append(value)

    # Testing on given data
    def test(self):
        self.net_encoder.eval()
        self.net_fusion.eval()
        with torch.no_grad():
            x11,x12,x21,x22,x31,x32,x41f,x42f=self.net_encoder(self.data)
            self.pred_img = self.net_fusion(x11,x12,x21,x22,x31,x32,x41f,x42f)
        self.net_encoder.train()
        self.net_fusion.train()

        return self.pred_img
    
    def set_device(self, x):
        if isinstance(x, dict):
            for key, item in x.items():
                if item is not None:
                    if key == 'label':
                        x[key] = item.to(self.device, dtype=torch.int)
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
        self.save_img_y = self.opt['val'].get('save_img_y', False)
        self.dataset_name = self.opt['val'].get('name')
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
            im_name = val_data[1]['im_name']
            self.feed_data(val_data[0])
            self.pred_img = self.test()
            visuals = self.get_current_visuals()
            sr_img = tensor2img(visuals['pred_img'].detach(), min_max=(0, 1))
            metric_data['img'] = sr_img
            torch.cuda.empty_cache()

            n=len(visuals['pred_img'])
            for i in range(n):
                sr_img = tensor2img(visuals['pred_img'][i].detach(), min_max=(0, 1))
                sr_img_y = tensor2img(visuals['pred_img_y'][i].detach(), min_max=(0, 1))
                img_name = osp.splitext(im_name[i])[0]
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
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
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
                                    visuals['gt_ir'].repeat(1, 3, 1, 1)), dim=0)
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
        return self.current_fusion_metric if hasattr(self, 'current_fusion_metric') else 0
    
    def save(self, epoch, current_iter):
        self.save_network([self.net_encoder, self.net_fusion], 'net_fe_g', current_iter, param_key=['params_encoder','params_fusion'])
        self.save_training_state(epoch, current_iter)
    
    def save_best(self, current_iter):
        logger = get_root_logger()
        self.save_network([self.net_encoder, self.net_fusion], 'net_best', current_iter, param_key=['params_encoder','params_fusion'])
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
