from pathlib import Path
import cv2

from person_matching import is_same_person

DATASET_ROOT = Path("dataset_ETHZ")

if __name__ == "__main__":
    dataset_root = Path(DATASET_ROOT)
    for img1_path in sorted(dataset_root.glob("**/*.png")):
        for img2_path in sorted(dataset_root.glob("**/*.png")):
            if img1_path == img2_path:
                continue

            # cv2を用いて画像の読み込み
            img1 = cv2.imread(str(img1_path))
            img2 = cv2.imread(str(img2_path))
            
            # 読み込みに失敗した場合はスキップ
            if img1 is None or img2 is None:
                continue

            # 同一人物かどうか判定
            cos_sim = is_same_person(img1, img2)
            print(
                str(img1_path).split("/")[-2],
                str(img2_path).split("/")[-2],
                "同一人物" if cos_sim > 0.7 else "異なる人物",
                cos_sim
            )
