from pathlib import Path
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class BreastMRIPairedDataset(Dataset):
    def __init__(
        self,
        image_dir: str | Path,
        seg_dir: str | Path,
        image_size: int = 256
    ):
        self.image_dir = Path(image_dir)
        self.seg_dir = Path(seg_dir)

        self.image_paths = sorted(self.image_dir.glob("*.png"))

        if not self.image_paths:
            raise ValueError(f"No PNG files found in {self.image_dir}")

        # Check that every image has a corresponding segmentation
        missing_segs = [
            path.name
            for path in self.image_paths
            if not (self.seg_dir / path.name).exists()
        ]

        if missing_segs:
            raise ValueError(
                f"Missing {len(missing_segs)} segmentation masks. "
                f"Example: {missing_segs[:5]}"
            )

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),

            # Diffusion models typically expect images in [-1, 1]
            transforms.Normalize([0.5], [0.5])
        ])

        self.seg_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        seg_path = self.seg_dir / image_path.name

        # Breast MRI is grayscale
        image = Image.open(image_path).convert("L")
        seg = Image.open(seg_path).convert("L")

        image = self.image_transform(image)
        seg = self.seg_transform(seg)

        return {
            "image": image,
            "seg": seg,
            "filename": image_path.name
        }