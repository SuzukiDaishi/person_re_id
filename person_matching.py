from typing import Any

import cv2
import numpy as np
import torch  # type: ignore
import torchreid  # type: ignore


def preprocess_img(img: np.ndarray) -> torch.Tensor:
    # cv2ではリサイズのサイズ指定は (width, height) となるので注意
    resized = cv2.resize(img, (128, 256))  # 元: (height, width)=(256,128)
    # 0~255の値を0~1に正規化し、float32に変換
    img_float = resized.astype("float32") / 255.0
    # HWC -> CHW の順に変換
    tensor = torch.from_numpy(img_float).permute(2, 0, 1)
    # チャンネル毎の正規化
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


def is_same_person(
    img1: np.ndarray,
    img2: np.ndarray,
    model: Any = torchreid.models.build_model(
        name="osnet_x1_0",
        num_classes=1000,  # 利用するデータセットに合わせて変更
        pretrained=True,
    ),
) -> float:
    """
    画像1と画像2が同一人物かどうかを判定する関数
    """
    model.eval()
    # cv2で読み込んだ画像はBGRなのでRGBに変換する
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # 前処理
    img1_tensor = preprocess_img(img1_rgb).unsqueeze(0)
    img2_tensor = preprocess_img(img2_rgb).unsqueeze(0)
    with torch.inference_mode():
        feature1: torch.Tensor = model(img1_tensor)  # torch.Size([1, 512])
        feature2: torch.Tensor = model(img2_tensor)  # torch.Size([1, 512])
    # feature1とfeature2のcosine類似度を計算
    cos_sim_tensor = torch.nn.functional.cosine_similarity(feature1, feature2)
    cos_sim: float = cos_sim_tensor.item()  # tensorをfloatに変換
    return cos_sim


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # 引数のパース
    parser = argparse.ArgumentParser(description="person re-identification")
    parser.add_argument(
        "person_1",
        type=Path,
        help="画像1",
    )
    parser.add_argument(
        "person_2",
        type=Path,
        help="画像2",
    )

    args = parser.parse_args()
    person_1: Path = args.person_1
    person_2: Path = args.person_2
    print(f"画像1: {person_1}")
    print(f"画像2: {person_2}")

    # cv2を用いて画像の読み込み
    img1 = cv2.imread(str(person_1))
    img2 = cv2.imread(str(person_2))
    # 読み込み失敗時のチェック
    if img1 is None:
        raise ValueError(f"画像1の読み込みに失敗しました: {person_1}")
    if img2 is None:
        raise ValueError(f"画像2の読み込みに失敗しました: {person_2}")

    # 同一人物かどうかを判定
    cos_sim = is_same_person(img1, img2)
    print(f"cosine similarity: {cos_sim:.4f}")
    # 閾値を設定して判定
    if cos_sim > 0.7:
        print("同一人物")
    else:
        print("異なる人物")
