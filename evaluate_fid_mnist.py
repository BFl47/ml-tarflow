import argparse
import math
import pathlib

import torch
import torchvision as tv

from transformer_flow import Model
import utils

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def to_fid_input(x: torch.Tensor) -> torch.Tensor:
    x = 0.5 * (x.clamp(min=-1, max=1) + 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)
    return x


def run_fid_evaluation(
    model: torch.nn.Module,
    data: torch.utils.data.Dataset,
    sample_dir: pathlib.Path,
    device: str,
    batch_size: int,
    img_size: int,
    patch_size: int,
    channel_size: int,
    num_classes: int,
    denoising_lr: float = 1.0,
    num_samples: int | None = None,
) -> tuple[float, float | None]:
    sample_dir.mkdir(exist_ok=True, parents=True)

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, drop_last=False
    )

    fid = utils.FID(reset_real_features=True, normalize=True).to(device)

    print("computing real-image FID features")
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device)
            fid.update(to_fid_input(x), real=True)

    num_samples = num_samples if num_samples is not None else len(data)
    num_batches = math.ceil(num_samples / batch_size)

    print(f"drawing {num_samples} samples")
    samples_to_save = []
    all_samples = []
    all_y = []

    for i in range(num_batches):
        b = batch_size if i < num_batches - 1 else num_samples - (num_batches - 1) * batch_size
        noise = torch.randn(
            b,
            (img_size // patch_size) ** 2,
            channel_size * patch_size**2,
            device=device,
        )
        y = torch.randint(num_classes, (b,), device=device) if num_classes > 0 else None

        with torch.no_grad():
            samples = model.reverse(noise, y)
        fid.update(to_fid_input(samples), real=False)

        all_samples.append(samples.cpu())
        if y is not None:
            all_y.append(y.cpu())

        if len(samples_to_save) < 20:
            n_to_save = min(20 - len(samples_to_save), samples.shape[0])
            samples_to_save.append(samples[:n_to_save].cpu())

    fid_score = fid.compute().item()
    print(f"FID score (no denoising): {fid_score:.2f}")

    samples_grid = torch.cat(samples_to_save, dim=0)
    tv.utils.save_image(samples_grid, sample_dir / "img_fid_sample_noised.png", nrow=10, normalize=True)

    fid_score_denoised: float | None = None
    if denoising_lr > 0:
        print("Denoising samples...")
        for p in model.parameters():
            p.requires_grad = False

        all_samples_cat = torch.cat(all_samples, dim=0).to(device)
        all_y_cat = torch.cat(all_y, dim=0).to(device) if all_y else None

        denoised_samples = []
        samples_to_save_den = []
        for j in range(0, len(all_samples_cat), batch_size):
            batch_end = min(j + batch_size, len(all_samples_cat))
            x = torch.clone(all_samples_cat[j:batch_end]).detach()
            x.requires_grad = True
            y_batch = all_y_cat[j:batch_end] if all_y_cat is not None else None

            base_lr = batch_size * img_size**2 * channel_size * 0.1**2
            lr = denoising_lr * base_lr

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                z, _, logdets = model(x, y_batch)
            loss = model.get_loss(z, logdets)
            grad = torch.autograd.grad(loss, [x])[0]
            x.data.add_(grad, alpha=-lr)
            x_denoised = x.detach().cpu()
            denoised_samples.append(x_denoised)

            if len(samples_to_save_den) < 20:
                n_to_save = min(20 - len(samples_to_save_den), x_denoised.shape[0])
                samples_to_save_den.append(x_denoised[:n_to_save])

        denoised_samples_cat = torch.cat(denoised_samples, dim=0).to(device)
        fid_denoised = utils.FID(reset_real_features=True, normalize=True).to(device)

        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(device)
                fid_denoised.update(to_fid_input(x), real=True)

        with torch.no_grad():
            for j in range(0, len(denoised_samples_cat), batch_size):
                batch_end = min(j + batch_size, len(denoised_samples_cat))
                fid_denoised.update(to_fid_input(denoised_samples_cat[j:batch_end]), real=False)

        fid_score_denoised = fid_denoised.compute().item()
        print(f"FID score (denoised): {fid_score_denoised:.2f}")

        samples_grid_den = torch.cat(samples_to_save_den, dim=0)
        denoised_path = sample_dir / "img_fid_sample_denoised.png"
        tv.utils.save_image(samples_grid_den, denoised_path, nrow=10, normalize=True)

    return fid_score, fid_score_denoised


def main(args: argparse.Namespace) -> None:
    utils.set_random_seed(args.seed)
    device = get_device()
    print(f"using device {device}")

    transform = tv.transforms.Compose(
        [
            tv.transforms.Resize((args.img_size, args.img_size)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize((0.5,), (0.5,)),
        ]
    )

    data = tv.datasets.MNIST(args.data, transform=transform, train=False, download=False)
    data_loader = torch.utils.data.DataLoader(
        data, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    model = Model(
        in_channels=args.channel_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        num_blocks=args.blocks,
        layers_per_block=args.layers_per_block,
        num_classes=args.num_classes,
    ).to(device)
    ckpt = torch.load(args.ckpt_file, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model.eval()
    run_fid_evaluation(
        model=model,
        data=data,
        sample_dir=pathlib.Path(args.output_dir),
        device=device,
        batch_size=args.batch_size,
        img_size=args.img_size,
        patch_size=args.patch_size,
        channel_size=args.channel_size,
        num_classes=args.num_classes,
        denoising_lr=args.denoising_lr,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=".", type=pathlib.Path, help="Path to MNIST data")
    parser.add_argument("--ckpt_file", required=True, type=str, help="Path to model checkpoint")

    parser.add_argument("--img_size", default=28, type=int, help="Image size")
    parser.add_argument("--channel_size", default=1, type=int, help="Image channel size")
    parser.add_argument("--num_classes", default=10, type=int, help="Number of classes")

    parser.add_argument("--patch_size", default=4, type=int, help="Patch size")
    parser.add_argument("--channels", default=128, type=int, help="Model width")
    parser.add_argument("--blocks", default=4, type=int, help="Number of flow blocks")
    parser.add_argument("--layers_per_block", default=4, type=int, help="Layers per block")

    parser.add_argument("--batch_size", default=256, type=int, help="Batch size")
    parser.add_argument(
        "--num_samples",
        default=10000,
        type=int,
        help="Number of generated samples for FID; set <=0 to use full test-set size",
    )
    parser.add_argument("--seed", default=100, type=int, help="Random seed")
    parser.add_argument(
        "--denoising_lr",
        default=1.0,
        type=float,
        help="Learning rate for self-denoising refinement; set to 0 to disable denoising",
    )

    main(parser.parse_args())
