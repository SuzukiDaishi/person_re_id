import argparse
from pathlib import Path
from typing import Any

import torch  # type: ignore
import torchreid  # type: ignore
from PIL import Image
from torchvision import transforms  # type: ignore


def is_same_person(
    model: Any,
    img1: Image.Image,
    img2: Image.Image,
    transform: transforms.Compose = transforms.Compose(
        [
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ),
) -> float:
    """
    画像1と画像2が同一人物かどうかを判定する関数
    """
    img1_tensor = transform(img1).unsqueeze(0)
    img2_tensor = transform(img2).unsqueeze(0)
    with torch.inference_mode():
        feature1: torch.Tensor = model(img1_tensor) # torch.Size([1, 512])
        feature2: torch.Tensor = model(img2_tensor) # torch.Size([1, 512])

    # feature1とfeature2のcosine類似度を計算
    cos_sim_tensor = torch.nn.functional.cosine_similarity(feature1, feature2)
    cos_sim: float = cos_sim_tensor.item()  # tensorをfloatに変換
    return cos_sim

if __name__ == "__main__":

    # 事前学習済み OSNet (osnet_x1_0) の読み込み
    model = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,  # 利用するデータセットに合わせて変更
        pretrained=True,
    )
    model.eval()

    dataset_root = Path("dataset_ETHZ")
    for img1_path in sorted(dataset_root.glob("**/*.png")):
        for img2_path in sorted(dataset_root.glob("**/*.png")):
            if img1_path == img2_path:
                continue

            # 画像の読み込み
            img1 = Image.open(img1_path).convert("RGB")
            img2 = Image.open(img2_path).convert("RGB")

            # 同一人物かどうかを判定
            cos_sim = is_same_person(model, img1, img2)
            print(str(img1_path).split("/")[-2], str(img2_path).split("/")[-2], "同一人物" if cos_sim > 0.7 else "異なる人物", cos_sim)
            # 閾値を設定して判定
