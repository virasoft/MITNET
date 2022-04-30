import os
import shutil
from argparse import Namespace
from datetime import datetime
from glob import glob

import numpy as np
import pytorch_lightning as pl
import torch

from cnn_finetune import make_model
import torchvision.transforms as cv_transforms
from pytorch_lightning.callbacks import EarlyStopping
#from pytorch_lightning.logging import CometLogger
from sklearn.metrics import confusion_matrix
from torch import jit
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets  # , transforms
import torch.nn as nn
import argparse

def make_classifier(in_features, num_classes):
	return nn.Sequential(
		nn.Linear(in_features, out_features=4096),
		nn.ReLU(inplace=True),
		nn.Dropout(),
		nn.Linear(in_features=4096, out_features=2048),
		nn.ReLU(inplace=True),
		nn.Dropout(),
		nn.Linear(in_features=2048, out_features=1024),
		nn.ReLU(inplace=True),
		nn.Dropout(),
		nn.Linear(in_features=1024, out_features=num_classes),
	)


MITNET_rec = make_model('vgg11', num_classes=2, pretrained=True, input_size=(100, 100), classifier_factory=make_classifier)

available_models = {'MITNET_rec': MITNET_rec}
model_idx = 0  # the model index from the "available_models" dictionary
net = list(available_models.keys())[model_idx]

exp_info = 'MITNET_rec_'

model = available_models[net]
batch_size = 8
lr = 1e-3 * batch_size / 8
lr_scheduler = 'ReduceLROnPlateau'  # 'ReduceLROnPlateau'  'StepLR'
num_epoch = 100
overfit_pct = 100  # the percentage of data to be used for training. This is useful for quick tests/troubleshooting

args = {
	'batch_size': batch_size,
	'lr': lr,
	'lr_scheduler': lr_scheduler,
	'num_epoch': num_epoch,
	'net': net
}
params = Namespace(**args)

# image transformations required
im_transform = cv_transforms.Compose([  # cv_transforms.Resize([224, 224]),
	cv_transforms.RandomVerticalFlip(),  # flipping the image vertically
	cv_transforms.RandomHorizontalFlip(),
	cv_transforms.ToTensor(),
	cv_transforms.Normalize(mean=[0.485, 0.456, 0.406],
	                        std=[0.229, 0.224, 0.225])
])


class MetricsCallback(pl.Callback):
	"""PyTorch Lightning metric callback."""

	def __init__(self):
		super().__init__()
		self.metrics = []

	def on_train_end(self, trainer, pl_module):
		self.metrics.append(trainer.callback_metrics)

	def on_validation_end(self, trainer, pl_module):
		self.metrics.append(trainer.callback_metrics)


class MitosisClassifier(pl.LightningModule):
	def __init__(self, hparams,data_dir):
		super(MitosisClassifier, self).__init__()
		self.hparams = hparams

		# init the loss weights
		self.num_classes = 2
		self.batch_size = self.hparams.batch_size
		self.lr = self.hparams.lr
		self.scheduler = self.hparams.lr_scheduler
		self.model = model
		self.net = self.hparams.net
		self.data_dir=data_dir

	# noinspection PyAttributeOutsideInit
	def prepare_data(self):

		# prepare transforms standard to ImageNet
		self.train_images = datasets.ImageFolder(self.data_dir + '/training', transform=im_transform)
		self.val_images = datasets.ImageFolder(self.data_dir + '/validation', transform=im_transform)

	# Forward pass
	def forward(self, x):
		representations = self.model(x)
		return representations

	# Training step
	def training_step(self, train_batch, batch_idx):
		# extracting input and output from the batch
		x, labels = train_batch

		# doing a forward pass
		logits = self.forward(x)
		pred = logits.argmax(dim=1, keepdim=True)

		# calculating the loss and accuracy
		loss = self.cross_entropy_loss(logits, labels)
		correct = pred.eq(labels.view_as(pred)).sum().item()
		acc = torch.tensor(correct / x.size(0)).type_as(loss)

		output = {
			# REQUIRED: It is required for us to return "loss"
			"loss": loss,
			"accuracy": acc
		}
		return output

	def training_epoch_end(self, outputs):
		avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
		avg_acc = torch.stack([x['accuracy'] for x in outputs]).mean()

		logs = {'train_loss': avg_loss, 'train_acc': avg_acc}

		# log training accuracy at the end of an epoch
		return {
			'avg_train_loss': avg_loss,
			'avg_train_acc': avg_acc,
			'log': logs,
			'progress_bar': {'train_loss': avg_loss, 'train_acc': avg_acc}
		}

	def validation_step(self, val_batch, batch_idx):
		return self._eval_step(val_batch, 'val')

	def _eval_step(self, batch, prefix):
		x, y = batch
		y_hat = self.forward(x)

		# calculate loss
		loss = self.cross_entropy_loss(y_hat, y)

		# calculate accuracy
		labels_hat = torch.argmax(y_hat, dim=1)
		acc = torch.sum(y == labels_hat).detach() / (len(y) * 1.0)
		acc = torch.tensor(acc)

		# compute confusion matrix
		conf_mat = confusion_matrix(y.cpu(), torch.argmax(y_hat, dim=1).cpu(), labels=range(2))
		conf_mat = torch.tensor(conf_mat)

		# if self.on_gpu:
		#     acc = acc.cuda(loss.device.index)

		# in DP mode (default) make sure if result is scalar, there's another dim in the beginning
		if self.trainer.use_dp or self.trainer.use_ddp2:
			loss = loss.unsqueeze(0)
			acc = acc.unsqueeze(0)
			conf_mat = conf_mat.unsqueeze(0)

		return {f'{prefix}_loss': loss, f'{prefix}_acc': acc, f'{prefix}_conf_mat': conf_mat}

	def validation_epoch_end(self, outputs):
		return self._eval_epoch_end(outputs, 'val')

	def _eval_epoch_end(self, outputs, prefix):
		"""
		Called at the end of test/validation to aggregate outputs
		:param outputs: list of individual outputs of each validation step
		:return:
		"""
		# if returned a scalar from validation_step, outputs is a list of tensor scalars
		# we return just the average in this case (if we want)
		# return torch.stack(outputs).mean()

		loss_mean = 0
		acc_mean = 0
		conf_matrix = np.zeros((self.num_classes, self.num_classes))
		for output in outputs:

			loss = output[f'{prefix}_loss']
			# reduce manually when using dp
			if self.trainer.use_dp or self.trainer.use_ddp2:
				loss = torch.mean(loss)
			loss_mean += loss

			acc = output[f'{prefix}_acc']
			# reduce manually when using dp
			if self.trainer.use_dp or self.trainer.use_ddp2:
				acc = torch.mean(acc)
			acc_mean += acc

			conf_mat = output[f'{prefix}_conf_mat']
			# reduce manually when using dp
			if self.trainer.use_dp or self.trainer.use_ddp2:
				conf_mat = torch.sum(conf_mat)
			conf_matrix += conf_mat.numpy()

		loss_mean /= len(outputs)
		acc_mean /= len(outputs)

		# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		TN, FP, FN, TP = conf_matrix.ravel()
		TNR = TN / (TN + FP)
		TPR_Recall = TP / (TP + FN)
		PPV_Precission = TP / (TP + FP)
		# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		tqdm_dict = {f'{prefix}_loss': loss_mean, f'{prefix}_acc': acc_mean}
		self.logger.experiment.log_confusion_matrix(
			labels=["mitosis", "not_mitosis"],
			matrix=conf_matrix,
			file_name=f"confusion-matrix_{self.current_epoch}.json", )
		result = {
			'progress_bar': tqdm_dict,
			'log': {f'{prefix}_loss': loss_mean,
			        f'{prefix}_acc': acc_mean,
			        # f'{prefix}_conf_mat': conf_matrix,
			        f'{prefix}_TNR': TNR,
			        f'{prefix}_TPR_Recall': TPR_Recall,
			        f'{prefix}_PPV_Precission': PPV_Precission,
			        },
			f'{prefix}_loss': loss_mean,
			f'{prefix}_acc': acc_mean,
			f'batch_size': self.batch_size,
			f'lr': self.lr}

		print('')
		print(f'TN: {TN}')
		print(f'TP: {TP}')
		print(f'FP: {FP}')
		print(f'FN: {FN}')
		print(f'TNR: {TNR.round(3)}')
		print(f'TPR/Recall: {TPR_Recall.round(3)}')
		print(f'PPV/Precission: {PPV_Precission.round(3)}')
		return result

	# Optimizer
	def configure_optimizers(self):
		# Essential function
		# we are using SGD optimizer for our model
		optimizer = torch.optim.SGD(self.parameters(),
		                            lr=self.lr,
		                            momentum=0.9,
		                            nesterov=True,
		                            weight_decay=0.0005
		                            )

		if self.scheduler == 'StepLR':
			scheduler = StepLR(optimizer, step_size=2, gamma=0.96)
		else:
			scheduler = {
				'scheduler': ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, threshold=0.001),
				'monitor': 'val_loss',  # Default: val_loss
				'interval': 'epoch',
				'frequency': 1
			}
		return [optimizer], [scheduler]

	@staticmethod
	def cross_entropy_loss(logits, labels):
		# calculating the loss

		return F.cross_entropy(logits, labels)

	# Data loaders
	def train_dataloader(self):
		# This is an essential function. Needs to be included in the code
		return DataLoader(self.train_images, shuffle=True, batch_size=self.hparams.batch_size, num_workers=0)

	def val_dataloader(self):
		return DataLoader(self.val_images, batch_size=self.hparams.batch_size, num_workers=0)


if __name__ == "__main__":


	# Create the parser and add arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('--savedir',default='../pt/', help="dir to save pt")
	parser.add_argument("--datadir",default='../data/', help="main source dir for training and validation")

	# Parse and print the results
	args = parser.parse_args()

	DIR = args.savedir
	data_dir =args.datadir

	# callback for saving the best model
	filepath = DIR + exp_info + r'_mitosis_{epoch} {val_loss:.2f} {batch_size} {lr:.2e}'
	checkpoint_callback = pl.callbacks.ModelCheckpoint(filepath=filepath,
	                                                   monitor="val_loss",
	                                                   mode='min',
	                                                   verbose=True,
	                                                   save_top_k=1)
	early_stopping = EarlyStopping('val_loss', patience=10, verbose=True)
	metrics_callback = MetricsCallback()
	lr_logger = pl.callbacks.LearningRateLogger()

	now = datetime.now()
	dt_string = now.strftime("%d-%m-%Y %H:%M:%S")

	# use comet logger
	# comet_logger = CometLogger(
	# 	api_key="**************************",
	# 	experiment_name=f'**************************',
	# 	project_name="**************************"
	# )

	myTrainer = pl.Trainer(gpus=1,
	                       log_gpu_memory='all',
	                       # use_amp=True,
	                       # amp_level='O1',
	                       # precision=16,
	                       max_epochs=num_epoch,
	                       #logger=comet_logger,
	                       checkpoint_callback=checkpoint_callback,
	                       # early_stop_callback=early_stopping,
	                       callbacks=[lr_logger, metrics_callback],
	                       profiler=True,
	                       check_val_every_n_epoch=1,
	                       show_progress_bar=False,
	                       fast_dev_run=False,
	                       auto_lr_find=False,  # finds learning rate automatically,
	                       # limit_train_batches=0.01,
	                       # limit_val_batches=0.01
	                       )
	_model = MitosisClassifier(params,data_dir)
	myTrainer.fit(_model)


	# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx Save model as trace file xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

	def createTracedModel(network, random_input):
		date_now = datetime.now()
		date_stamp = date_now.strftime("%d%m%Y_%H%M")
		traced_net = jit.trace(network, random_input)
		traced_net.save(fr'{DIR}HE_mitosis_trace_{exp_info}.pt')
		print("Success - model_trace was saved!")


	val_images = datasets.ImageFolder(data_dir + '/validation', transform=im_transform)
	sample_data = DataLoader(val_images, batch_size=batch_size, num_workers=0)

	# Load the best model and create a pt file
	# model = MitosisClassifier(params)
	saved_model = glob(f'{DIR}{exp_info}*.ckpt')[0]

	pre_trained_model = _model.load_from_checkpoint(saved_model)

	inputs, _ = next(iter(sample_data))
	createTracedModel(pre_trained_model.eval(), inputs)

	# comet_logger.experiment.log_model(os.path.basename(saved_model)[:-5],
	#                                   saved_model,
	#                                   os.path.basename(saved_model)[:-5])
	# move the saved model to the models directory
	shutil.move(saved_model, f'{DIR}models/{os.path.basename(saved_model)}')
