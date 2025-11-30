dataset_paths = {
	'celeba_train': '',
	'celeba_test': '',
	'celeba_train_sketch': '',
	'celeba_test_sketch': '',
	'celeba_train_segmentation': '',
	'celeba_test_segmentation': '',
	'ffhq': '',
    'lrs2':'/home/wls/wxb/processed',
    'lrs2_cmlr':'/home/wls/wxb/mutidataset'
}

model_paths = {
	'stylegan_ffhq': '/home/wls/wxb/2024.6.27/pixel2style2pixel-master/pertrained_models/stylegan2-ffhq-config-f.pt',
	'ir_se50': '/home/wls/wxb/2024.6.27/pixel2style2pixel-master/pertrained_models/model_ir_se50.pth',
	'circular_face': 'pretrained_models/CurricularFace_Backbone.pth',
	'mtcnn_pnet': 'pretrained_models/mtcnn/pnet.npy',
	'mtcnn_rnet': 'pretrained_models/mtcnn/rnet.npy',
	'mtcnn_onet': 'pretrained_models/mtcnn/onet.npy',
	'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
	'moco': 'pretrained_models/moco_v2_800ep_pretrain.pth.tar'
}
