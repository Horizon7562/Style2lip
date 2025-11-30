from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import os, random,cv2
from multiprocessing import Manager

#cv2.setNumThreads(0)
#cv2.ocl.setUseOpenCL(False)
import torch
import numpy as np
from os.path import dirname, join, basename, isfile
from models.hparams import hparams, get_image_list
from models import audio
from glob import glob

syncnet_T = 5
syncnet_mel_step_size = 16




class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root,split, opts, target_transform=None, source_transform=None):
		#self.source_paths = sorted(data_utils.make_dataset(source_root))
		#self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_paths = get_image_list(source_root, split)

		self.target_paths = get_image_list(target_root, split)

		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts
		


  
  
	def get_frame_id(self, frame):
		return int(basename(frame).split('.')[0])


	def get_window(self, start_frame):
		start_id = self.get_frame_id(start_frame)
		vidname = dirname(start_frame)

		window_fnames = []
		for frame_id in range(start_id, start_id + syncnet_T):
			frame = join(vidname, '{}.jpg'.format(frame_id))
			if not isfile(frame):
				return None
			window_fnames.append(frame)
		return window_fnames

	def read_window(self, window_fnames):
		if window_fnames is None: return None
		window = []
		#window=Manager().list()
		for fname in window_fnames:
			img = cv2.imread(fname)
			if img is None:
				return None
			try:
				img = cv2.resize(img, (hparams.img_size, hparams.img_size))
			except Exception as e:
				return None

			window.append(img)

		return window

  
	def crop_audio_window(self, spec, start_frame):
		if type(start_frame) == int:
			start_frame_num = start_frame
		else:
			start_frame_num = self.get_frame_id(start_frame) # 0-indexing ---> 1-indexing
		start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
		end_idx = start_idx + syncnet_mel_step_size

		return spec[start_idx : end_idx, :]
  
	def get_segmented_mels(self, spec, start_frame):
		mels = []
		assert syncnet_T == 5
		start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
		if start_frame_num - 2 < 0: return None
		for i in range(start_frame_num, start_frame_num + syncnet_T):
			m = self.crop_audio_window(spec, i - 2)
			if m.shape[0] != syncnet_mel_step_size:
				return None
			mels.append(m.T)

		mels = np.asarray(mels)

		return mels

	def prepare_window(self, window):
        # 3 x T x H x W
		x = np.asarray(window) / 255.
		#x = 2 * (x - 0.5)  # Step 2: Scale to [-1, 1]后加的
		x = np.transpose(x, (3, 0, 1, 2))
		return x

	def __len__(self):
		# print(len(self.source_paths))
		# exit()
		return len(self.source_paths)

	def __getitem__(self, index):
		while 1:
			
			# from_path = self.source_paths[index]
			# from_im = Image.open(from_path)
			# from_im = from_im.convert('RGB') if self.opts.label_nc == 0 else from_im.convert('L')

			# to_path = self.target_paths[index]
			# to_im = Image.open(to_path).convert('RGB')
			# if self.target_transform:
			# 	to_im = self.target_transform(to_im)

			# if self.source_transform:
			# 	from_im = self.source_transform(from_im)
			# else:
			# 	from_im = to_im

			index = random.randint(0, len(self.source_paths) - 1)

			vidname = self.source_paths[index]
			img_names = list(glob(join(vidname, '*.jpg')))

			#img_names =Manager().list(glob(join(vidname, '*.jpg')))
			if len(img_names) <= 3 * syncnet_T:
				continue
			img_name = random.choice(img_names)
			window_fnames = self.get_window(img_name)
			if window_fnames is None:
				continue
			window = self.read_window(window_fnames)
			if window is None:
				continue
			try:
				wavpath = join(vidname, "audio.wav")
				wav = audio.load_wav(wavpath, hparams.sample_rate)

				orig_mel = audio.melspectrogram(wav).T
			except Exception as e:
				continue
			mel = self.crop_audio_window(orig_mel.copy(), img_name)
			if (mel.shape[0] != syncnet_mel_step_size):
				continue

			indiv_mels = self.get_segmented_mels(orig_mel.copy(), img_name)
			if indiv_mels is None: continue
			mel = torch.FloatTensor(mel.T).unsqueeze(0)
			indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)


			window = self.prepare_window(window)
			y=window.copy()

			y[:, :, :y.shape[2]//2] = 0.


			return window, indiv_mels,mel,y
