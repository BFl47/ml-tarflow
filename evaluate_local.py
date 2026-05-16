import torch
import torchvision as tv
import utils
import pathlib
import umap

import matplotlib.pyplot as plt
import numpy as np

from transformer_flow import Model
from evaluate_fid_mnist import run_fid_evaluation

utils.set_random_seed(100)
notebook_output_path = pathlib.Path('runs/notebook')

# import model from runs/notebook/mnist_model_4_128_4_4_0.10.pth
model_name = '4_128_4_4_0.10'
dataset = 'mnist'
num_classes = 10
img_size = 28
channel_size = 1

# we use a small model for fast demonstration, increase the model size for better results
patch_size = 4
channels = 128
blocks = 4
layers_per_block = 4
# try different noise levels to see its effect
noise_std = 0.1

batch_size = 256
tsne_perplexity = 5.0

fid_denoising_lr = 1.0 # 0 to no denoising, increase for stronger denoising effect

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'  # if on mac
else:
    device = 'cpu'  # if mps not available
print(f'using device {device}')

transform = tv.transforms.Compose([
    tv.transforms.Resize((img_size, img_size)),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.5,), (0.5,)),
])

sample_dir = notebook_output_path / f'{dataset}_eval_{model_name}'
sample_dir.mkdir(exist_ok=True, parents=True)

ckpt_file = notebook_output_path / f'{dataset}_model_{model_name}.pth'

model = Model(
    in_channels=channel_size,
    img_size=img_size,
    patch_size=patch_size,
    channels=channels,
    num_blocks=blocks,
    layers_per_block=layers_per_block,
    num_classes=num_classes,
).to(device)
ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=True)
model.load_state_dict(ckpt, strict=True)
model.eval()
print('checkpoint loaded')

# now we can also evaluate the model by turning it into a classifier with Bayes rule, p(y|x) = p(y)p(x|y)/p(x)
latents = []
labels = []

data = tv.datasets.MNIST('.', transform=transform, train=False, download=False)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False)
num_correct = 0
num_examples = 0
for x, y in data_loader:
    x = x.to(device)
    y = y.to(device)
    eps = noise_std * torch.randn_like(x)
    x = x.repeat(num_classes, 1, 1, 1)
    y_ = torch.arange(num_classes, device=device).view(-1, 1).repeat(1, y.size(0)).flatten()
    with torch.no_grad():
        z, outputs, logdets = model(x, y_)
        losses = 0.5 * z.pow(2).mean(dim=[1, 2]) - logdets # keep the batch dimension
        pred = losses.reshape(num_classes, y.size(0)).argmin(dim=0)

    # losses has length num_classes * batch_size; reshape to (num_classes, batch_size)
    # then transpose to get (batch_size, num_classes)
    energy_vectors = losses.reshape(num_classes, y.size(0)).T
    probs = torch.softmax(-energy_vectors, dim=1)

    latents.append(probs.cpu())
    labels.append(y.cpu())

    num_correct += (pred == y).sum()
    num_examples += y.size(0)
print(f'Number of examples: {num_examples}; Number of correct: {num_correct}')
print(f'Accuracy: {100 * num_correct / num_examples:.2f}%')

# latent space UMAP visualization
# plot p(y∣x) not the latent space to have semantic discrimination

latents = torch.cat(latents).numpy()
labels = torch.cat(labels).numpy()

latents = latents / np.linalg.norm(latents, axis=1, keepdims=True)

# n_neighbors=15, min_dist=0.1, metric='cosine', init='spectral'
z_emb = umap.UMAP(metric='cosine').fit_transform(latents)

plt.figure(figsize=(8, 6))
plt.scatter(z_emb[:, 0], z_emb[:, 1], c=labels, s=5, cmap='tab10')
plt.colorbar()
plt.title('UMAP')
plt.savefig(sample_dir / 'img_umap.png')


# evaluate FID score of the model, with and without self-denoising
fid_score, fid_score_denoised = run_fid_evaluation(
    model=model,
    data=data,
    sample_dir=sample_dir,
    device=device,
    batch_size=batch_size,
    img_size=img_size,
    patch_size=patch_size,
    channel_size=channel_size,
    num_classes=num_classes,
    denoising_lr=fid_denoising_lr,
    num_samples=len(data),
)

# Print trajectory 
# Get full trajectory (list of tensors at each layer)
trajectory_noise = torch.randn(
    3,
    (img_size // patch_size) ** 2,
    channel_size * patch_size ** 2,
    device=device,
)
trajectory_y_gen = torch.randint(num_classes, (3,), device=device) if num_classes else None
trajectory = model.reverse(trajectory_noise, trajectory_y_gen, return_sequence=True)

# Select 20 equidistant steps
num_steps = 100
indices = [int(i * (len(trajectory) - 1) / (num_steps - 1)) for i in range(num_steps)]
trajectory_sampled = [trajectory[i] for i in indices]

# Visualize trajectory steps 
grid_images = []
for step_idx, (idx, sample) in enumerate(zip(indices, trajectory_sampled)):
    sample_3 = sample[:3].detach().cpu().float()
    tv.utils.save_image(sample_3, sample_dir / f'trajectory_step_{step_idx:02d}_layer_{idx}.png', nrow=1, normalize=True)
    grid_images.append(sample_3)



