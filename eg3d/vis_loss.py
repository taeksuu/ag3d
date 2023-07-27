import os
import argparse
import glob
import ffmpeg
import shutil
import jsonlines
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file_dir', type=str, required=True)
args = parser.parse_args()

output_dir = args.file_dir

deformer = []
eikonal = []
zero_delta_sdf = []
G = []
D = []


with jsonlines.open(os.path.join(output_dir, 'stats.jsonl')) as f:
    for idx, line in enumerate(f.iter()):
        deformer.append(line['Loss/deformer/']['mean'])
        eikonal.append(line['Loss/eikonal/']['mean'])
        zero_delta_sdf.append(line['Loss/zero_delta_sdf/']['mean'])
        G.append(line['Loss/G/loss']['mean'])
        D.append(line['Loss/D/loss']['mean'])

plt.plot([i for i in range(len(deformer))], deformer)
plt.xlabel('num images(K)')
plt.ylabel('deformer loss')
plt.savefig(os.path.join(output_dir, 'loss_deformer.png'))
plt.clf()

plt.plot([i for i in range(len(deformer))], eikonal)
plt.xlabel('num images(K)')
plt.ylabel('eikonal loss')
plt.savefig(os.path.join(output_dir, 'loss_eikonal.png'))
plt.clf()

plt.plot([i for i in range(len(deformer))], zero_delta_sdf)
plt.xlabel('num images(K)')
plt.ylabel('delta_sdf loss')
plt.savefig(os.path.join(output_dir, 'loss_zero_delta_sdf.png'))
plt.clf()

plt.plot([i for i in range(len(deformer))], G)
plt.xlabel('num images(K)')
plt.ylabel('G loss')
plt.savefig(os.path.join(output_dir, 'loss_G.png'))
plt.clf()

plt.plot([i for i in range(len(deformer))], D)
plt.xlabel('num images(K)')
plt.ylabel('D loss')
plt.savefig(os.path.join(output_dir, 'loss_D.png'))
plt.clf()
        
