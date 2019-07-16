# import torch module
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset

# import python module
import argparse
import glob
import os
import pdb
import numpy as np
import PIL.Image as Image
from tqdm import tqdm


args = {
    'trained_model':'writing saved_models format',
    'test_dir':'writing test directory',
    'nClass':'writing class number',
    'nEpochs':'writing epoch number',
    'patchsize':'writing patchsize',
    'batchsize':'writing batchsize'
}
print(args)
device = torch.device('cuda')

print('===Loading Data===')
# get image path
test_image_paths = glob.glob(os.path.join(args["test_dir"],'*.png'))
test_image_pathes.sort()
assert len(test_image_paths) != 0

# convert image into tensor and normalize
transform = transforms.Compose([
    transforms.CenterCrop(args['patchsize']),
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])
image_tensors = [transform(Image.open(path)).unsqueeze(0) for path in test_images_pathes]
tmp = torch.Tensor(len(image_tensors), 3, args['patchsize'], args['patchsize'])

test_dst = TensorDataset(image_tensors)
test_dataloader = DataLoader(test_dst,
                             batch_size = args['batchsize'],
                             shuffle = False,
                             num_workers = 4)


print('===Loading Model===')
model = torch.load(args['trained_model'])
model.to(device)
model.eval()

print('===Output as CSV===')
test_image_names = [path.split('/')[-1] for path in test_image_pathes]

if not os.path.exists('results'):
    os.mkdir('results')

model_name = args['trained_model'].split('/')[-1].split('.')[0]
output_path = os.path.join('results', model_name + '.csv')

with open(output_path, 'w') as f:
    for name, pred in zip(test_image_names, prediction):
        f.write('{}, {}\n'.format(name, pred))
