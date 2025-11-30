
import os,cv2
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
matplotlib.use('Agg')
from os.path import dirname, join, basename, isfile
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, w_norm, moco_loss
from configs import data_configs

from datasets.images_dataset_copy import ImagesDataset

from criteria.lpips.lpips import LPIPS

from models.psp_copy import pSp

from training.ranger import Ranger


from models.syncnet import SyncNet_color as SyncNet

use_cuda = torch.cuda.is_available()
def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint

def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
    global global_step
    global global_epoch

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    if overwrite_global_states:
        global_step = checkpoint["global_step"]
        global_epoch = checkpoint["global_epoch"]

    return model

def save_sample_images(x, g, gt,mouth, global_step, checkpoint_dir):



    # x1 = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # g1 = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)
    # gt1 = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    x1 = (x.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.)
    g1 = (g.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.)
    gt1 = (gt.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.)
    mouth1 = (mouth.detach().cpu().numpy().transpose(0, 2, 3, 4, 1) * 255.)

    x=np.clip(x1,0,255).astype(np.uint8)
    g = np.clip(g1, 0, 255).astype(np.uint8)
    gt = np.clip(gt1, 0, 255).astype(np.uint8)
    mouth = np.clip(mouth1, 0, 255).astype(np.uint8)



    #mouth = (mouth1.numpy().transpose(0, 2, 3, 4, 1) * 255.).astype(np.uint8)

    #refs, inps = x[..., 3:], x[..., :3]
    folder = join(checkpoint_dir, "samples_step{:09d}".format(global_step))
    if not os.path.exists(folder): os.mkdir(folder)
    collage = np.concatenate((x,mouth, gt,g), axis=-2)

    for batch_idx, c in enumerate(collage):
        for t in range(len(c)):
            cv2.imwrite('{}/{}_{}.jpg'.format(folder, batch_idx, t), c[t])

    # for i in range(g.shape[0]):
    #     # 提取第i张图像
    #     img = g[i]
	#
    #     image_path = join(folder, f'image_{i}.jpg')
    #     cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


syncnet_T = 5
syncnet_mel_step_size = 16
device = torch.device("cuda" if use_cuda else "cpu")
syncnet = SyncNet().to(device).eval()
for p in syncnet.parameters():
   p.requires_grad = False

logloss = nn.BCELoss()
def cosine_loss(a, v, y):

    d = nn.functional.cosine_similarity(a, v)
    loss = logloss(d.unsqueeze(1), y)
    return loss

def compute_psnr(y, y_hat):
		# 图像值范围为[0, 1]，因此最大值MAX_I = 1.0
		MAX_I = 1.0
		
		# 初始化累计 PSNR 的变量
		psnr_total = 0.0
		batch_size = y.shape[0]
		num_images = y.shape[2]  # 5 张图像
		
		# 对每张图像计算 MSE 和 PSNR
		for i in range(num_images):
			# 从每个批次和每张图像中提取图像
			y_image = y[:, :, i, :, :]
			y_hat_image = y_hat[:, :, i, :, :]
			
			# 计算均方误差 (MSE)
			mse = torch.mean((y_hat_image - y_image) ** 2, dim=(1, 2, 3))  # 对每个样本计算 MSE
			
			# 计算 PSNR
			psnr = 10 * torch.log10((MAX_I ** 2) / mse)
			
			# 累加 PSNR
			psnr_total += torch.mean(psnr)  # 对每个批次求平均 PSNR
		
		# 平均化所有图像的 PSNR
		psnr_avg = psnr_total / num_images
		
		return psnr_avg



#预训练模型是96尺寸，本文是256，batchsize必须大于等于2才能用sync
def get_sync_loss(mel, g):
    g = g[:, :, :, g.size(3)//2:]
    g = torch.cat([g[:, :, i] for i in range(syncnet_T)], dim=1)
    # B, 3 * T, H//2, W


    a, v = syncnet(mel, g)
    y = torch.ones(g.size(0), 1).float().to(device)
    return cosine_loss(a, v, y)

class Coach:
	def __init__(self, opts):
		self.opts = opts

		#self.global_step = 0
		self.global_step = 0
		
		self.device = 'cuda:0'  # TODO: Allow multiple GPU? currently using CUDA_VISIBLE_DEVICES
		self.opts.device = self.device

		if self.opts.use_wandb:
			from utils.wandb_utils import WBLogger
			self.wb_logger = WBLogger(self.opts)

		# Initialize network
		self.net = pSp(self.opts).to(self.device)

		# Estimate latent_avg via dense sampling if latent_avg is not available
		if self.net.latent_avg is None:
			self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

		# Initialize loss
		if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0:
			raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')

		self.mse_loss = nn.MSELoss().to(self.device).eval()
		self.l1_loss = nn.L1Loss().to(self.device).eval()

		if self.opts.lpips_lambda > 0:
			self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
		if self.opts.id_lambda > 0:
			self.id_loss = id_loss.IDLoss().to(self.device).eval()
		if self.opts.w_norm_lambda > 0:
			self.w_norm_loss = w_norm.WNormLoss(start_from_latent_avg=self.opts.start_from_latent_avg)
		if self.opts.moco_lambda > 0:
			self.moco_loss = moco_loss.MocoLoss().to(self.device).eval()

		# Initialize optimizer
		self.optimizer = self.configure_optimizers()

		# Initialize dataset
		self.train_dataset, self.test_dataset = self.configure_datasets()
		self.train_dataloader = DataLoader(self.train_dataset,
										   batch_size=self.opts.batch_size,
										   shuffle=True,
										   num_workers=int(self.opts.workers),
										   drop_last=True)
		self.test_dataloader = DataLoader(self.test_dataset,
										  batch_size=self.opts.test_batch_size,
										  shuffle=False,
										  num_workers=int(self.opts.test_workers),
										  drop_last=True)

		# device = torch.device("cuda" if use_cuda else "cpu")
		# syncnet = SyncNet().to(device)
		# for p in syncnet.parameters():
		# 	p.requires_grad = False
		# load_checkpoint('/home/lzx/wxb/2024.6.27/pixel2style2pixel-master/pertrained_models/lipsync_expert.pth', syncnet, None, reset_optimizer=True,overwrite_global_states=False)

		# Initialize logger
		log_dir = os.path.join(opts.exp_dir, 'logs')
		os.makedirs(log_dir, exist_ok=True)
		self.logger = SummaryWriter(log_dir=log_dir)

		# Initialize checkpoint dir
		self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
		os.makedirs(self.checkpoint_dir, exist_ok=True)
		self.best_val_loss = None
		if self.opts.save_interval is None:
			self.opts.save_interval = self.opts.max_steps



	def train(self):
		
		epoch=0
		self.net.train()
		while self.global_step < self.opts.max_steps:
			for batch_idx,(x, indiv_mels,mel,mouth) in enumerate(self.train_dataloader):
				
				self.optimizer.zero_grad()
				#y:[b,3,5,256,256]
				y=x

				x = torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
				#x:[b*5,3,256,256]
				x, y, indiv_mels, mel = x.to(self.device).float(), y.to(self.device).float(), indiv_mels.to(self.device).float(),mel.to(self.device).float()
				#y_hat:[b,3,5,256,256]
				
				y_hat, latent = self.net.forward(x,indiv_mels, return_latents=True)
				min_val = y_hat.min()
				max_val = y_hat.max()
				y_hat = (y_hat - min_val) / (max_val - min_val)
				

				
				#g_mouth和mouth
				g_mouth=y_hat.clone()
				g_mouth[:, :, :, g_mouth.size(3)//2:] = 0.


				loss, loss_dict, id_logs = self.calc_loss(y,y_hat, latent,mel)



				loss.backward()

				self.optimizer.step()

				# Logging related
				if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0):
					# for name, parms in self.net.named_parameters():
					# 	print(name, parms.grad)
					#self.parse_and_log_images(id_logs, x, x, y_hat, title='images/train/faces')
					save_sample_images(y,y_hat,y,g_mouth,self.global_step,'/home/wls/wxb/logs/train')
				if self.global_step % self.opts.board_interval == 0:
					self.print_metrics(loss_dict, prefix='train')
					self.log_metrics(loss_dict, prefix='train')

				# Log images of first batch to wandb
				if self.opts.use_wandb and batch_idx == 0:
					self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="train", step=self.global_step, opts=self.opts)
				# print("test")
				# loss=self.validate()
				# Validation related
				val_loss_dict = None
				if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:
					val_loss_dict = self.validate()


					if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):

						self.best_val_loss = val_loss_dict['loss']
						self.checkpoint_me(val_loss_dict, is_best=True)

				if self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps:
					if val_loss_dict is not None:
						self.checkpoint_me(val_loss_dict, is_best=False)
					else:
						self.checkpoint_me(loss_dict, is_best=False)

				if self.global_step == self.opts.max_steps:
					print('OMG, finished training!')
					break

				self.global_step += 1
			epoch+=1
			print("epoch:",epoch)

	def validate(self):
		self.net.eval()
		agg_loss_dict = []
		for batch_idx,(x, indiv_mels,mel,mouth) in enumerate(self.test_dataloader):
			
			y=x

			with torch.no_grad():
				x = torch.cat([x[:, :, i] for i in range(x.size(2))], dim=0)
				# x:[b*5,3,256,256]
				x, y, indiv_mels, mel = x.to(self.device).float(), y.to(self.device).float(), indiv_mels.to(
					self.device).float(), mel.to(self.device).float()
				# y_hat:[b,3,5,256,256]
				y_hat, latent = self.net.forward(x, indiv_mels, return_latents=True)
				min_val = y_hat.min()
				max_val = y_hat.max()
				y_hat = (y_hat - min_val) / (max_val - min_val)
				# g_mouth和mouth
				g_mouth = y_hat.clone()
				g_mouth[:, :, :, :g_mouth.size(3)//2] = 0.
				loss, cur_loss_dict, id_logs =self.calc_loss(y, y_hat, latent,mel)

				# if (cur_loss_dict['loss_l2'] < float(0.004)):
				# 	self.opts.lipsyn = float(0.03)
			agg_loss_dict.append(cur_loss_dict)

			# Logging related
			# self.parse_and_log_images(id_logs, x, y, y_hat,
			# 						  title='images/test/faces',
			# 						  subscript='{:04d}'.format(batch_idx))
			save_sample_images(y,y_hat,y,g_mouth, self.global_step, '/home/wls/wxb/logs/test')
			
			
			# Log images of first batch to wandb
			if self.opts.use_wandb and batch_idx == 0:
				self.wb_logger.log_images_to_wandb(x, y, y_hat, id_logs, prefix="test", step=self.global_step, opts=self.opts)
			

			# For first step just do sanity test on small amount of data
			if self.global_step == 0 and batch_idx >= 4:
				self.net.train()
				return None  # Do not log, inaccurate in first batch
			

		loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
		

		self.log_metrics(loss_dict, prefix='test')
		
			

		self.print_metrics(loss_dict, prefix='test')
		
		self.net.train()

		return loss_dict

	def checkpoint_me(self, loss_dict, is_best):
		save_name = 'best_model.pt' if is_best else f'iteration_{self.global_step}.pt'
		save_dict = self.__get_save_dict()
		checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
		torch.save(save_dict, checkpoint_path)
		with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
			if is_best:
				f.write(f'**Best**: Step - {self.global_step}, Loss - {self.best_val_loss} \n{loss_dict}\n')
				if self.opts.use_wandb:
					self.wb_logger.log_best_model()
			else:
				f.write(f'Step - {self.global_step}, \n{loss_dict}\n')

	def configure_optimizers(self):
		params = list(self.net.encoder.parameters())
		if self.opts.train_decoder:
			params += list(self.net.decoder.parameters())
		if self.opts.optim_name == 'adam':
			optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
		else:
			optimizer = Ranger(params, lr=self.opts.learning_rate)
		return optimizer

	def configure_datasets(self):
		if self.opts.dataset_type not in data_configs.DATASETS.keys():
			Exception(f'{self.opts.dataset_type} is not a valid dataset_type')
		print(f'Loading dataset for {self.opts.dataset_type}')
		dataset_args = data_configs.DATASETS[self.opts.dataset_type]
		transforms_dict = dataset_args['transforms'](self.opts).get_transforms()

		train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'],
									  target_root=dataset_args['train_target_root'],
										split="train",
									  source_transform=transforms_dict['transform_source'],
									  target_transform=transforms_dict['transform_gt_train'],
									  opts=self.opts)
		test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'],
									 target_root=dataset_args['test_target_root'],
										split="val",
									 source_transform=transforms_dict['transform_source'],
									 target_transform=transforms_dict['transform_test'],
									 opts=self.opts)
		if self.opts.use_wandb:
			self.wb_logger.log_dataset_wandb(train_dataset, dataset_name="Train")
			self.wb_logger.log_dataset_wandb(test_dataset, dataset_name="Test")
		print(f"Number of training samples: {len(train_dataset)}")
		print(f"Number of test samples: {len(test_dataset)}")
		return train_dataset, test_dataset

	def calc_loss(self, y, y_hat, latent,mel):
		
		loss_dict = {}
		loss = 0.0
		id_logs = None
		if self.opts.id_lambda > 0:
			loss_id, sim_improvement, id_logs = self.id_loss(y_hat, y, y)
			loss_dict['loss_id'] = float(loss_id)
			loss_dict['id_improve'] = float(sim_improvement)
			loss = loss_id * self.opts.id_lambda
		if self.opts.l2_lambda > 0:

			loss_l2 = F.mse_loss(y, y_hat)
			#loss_l2=self.l1_loss(y,y_hat)
			loss_dict['loss_l2'] = float(loss_l2)
			loss += loss_l2 * self.opts.l2_lambda
		if self.opts.lpips_lambda > 0:
			
			loss_lpips = self.lpips_loss(y_hat, y)
			loss_dict['loss_lpips'] = float(loss_lpips)
			

			loss += loss_lpips * self.opts.lpips_lambda
		# if self.opts.lpips_lambda > 0:
		# 	print(y.shape)
		# 	print(y_hat.shape)
		# 	exit()
		# 	loss_lpips=float()
		# 	for b in range(y.shape[0]):
		# 		for i in range(y.shape[2]):
		# 			loss_lpips = loss_lpips+self.lpips_loss(y_hat[b,:,i], y[b,:,i])

		# 	loss_dict['loss_lpips'] = float(loss_lpips/y.shape[2])

		# 	loss += loss_lpips * self.opts.lpips_lambda
		if self.opts.lpips_lambda_crop > 0:
			loss_lpips_crop = self.lpips_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_lpips_crop'] = float(loss_lpips_crop)
			loss += loss_lpips_crop * self.opts.lpips_lambda_crop
		if self.opts.l2_lambda_crop > 0:
			loss_l2_crop = F.mse_loss(y_hat[:, :, 35:223, 32:220], y[:, :, 35:223, 32:220])
			loss_dict['loss_l2_crop'] = float(loss_l2_crop)
			loss += loss_l2_crop * self.opts.l2_lambda_crop
		if self.opts.w_norm_lambda > 0:
			loss_w_norm = self.w_norm_loss(latent, self.net.latent_avg)
			loss_dict['loss_w_norm'] = float(loss_w_norm)
			loss += loss_w_norm * self.opts.w_norm_lambda
		if self.opts.moco_lambda > 0:
			loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat, y, y)
			loss_dict['loss_moco'] = float(loss_moco)
			loss_dict['id_improve'] = float(sim_improvement)
			loss += loss_moco * self.opts.moco_lambda
		if self.opts.lipsyn > 0:
			sync_loss=get_sync_loss(mel,y_hat)
			loss_dict['sync_loss'] = float(sync_loss)
			loss+=sync_loss*self.opts.lipsyn
		if self.opts.psnr > 0:
			psnr_value = compute_psnr(y, y_hat)
			loss_dict['psnr'] = float(psnr_value)
		

		loss_dict['loss'] = float(loss)
		return loss, loss_dict, id_logs

	
	def log_metrics(self, metrics_dict, prefix):
		for key, value in metrics_dict.items():
			self.logger.add_scalar(f'{prefix}/{key}', value, self.global_step)
		if self.opts.use_wandb:
			self.wb_logger.log(prefix, metrics_dict, self.global_step)

	def print_metrics(self, metrics_dict, prefix):
		print(f'Metrics for {prefix}, step {self.global_step}')
		for key, value in metrics_dict.items():
			print(f'\t{key} = ', value)

	def parse_and_log_images(self, id_logs, x, y, y_hat, title, subscript=None, display_count=2):
		im_data = []
		for i in range(display_count):
			cur_im_data = {
				'input_face': common.log_input_image(x[i], self.opts),
				'target_face': common.tensor2im(y[i]),
				'output_face': common.tensor2im(y_hat[i]),
			}
			if id_logs is not None:
				for key in id_logs[i]:
					cur_im_data[key] = id_logs[i][key]
			im_data.append(cur_im_data)
		self.log_images(title, im_data=im_data, subscript=subscript)

	def log_images(self, name, im_data, subscript=None, log_latest=False):
		fig = common.vis_faces(im_data)
		step = self.global_step
		if log_latest:
			step = 0
		if subscript:
			path = os.path.join(self.logger.log_dir, name, f'{subscript}_{step:04d}.jpg')
		else:
			path = os.path.join(self.logger.log_dir, name, f'{step:04d}.jpg')
		os.makedirs(os.path.dirname(path), exist_ok=True)
		fig.savefig(path)
		plt.close(fig)

	def __get_save_dict(self):
		save_dict = {
			'state_dict': self.net.state_dict(),
			'opts': vars(self.opts)
		}
		# save the latent avg in state_dict for inference if truncation of w was used during training
		if self.opts.start_from_latent_avg:
			save_dict['latent_avg'] = self.net.latent_avg
		return save_dict
