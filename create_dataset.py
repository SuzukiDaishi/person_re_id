from person_detect import PersonDetectorTracker
from pathlib import Path
import cv2
import numpy as np
import random
from shapely.geometry import box
from shapely.ops import unary_union

# 動画の一覧をあらかじめ配列で指定
VIDEO_PATHS = [
    "dataset_video/202504061058.mp4",
    "dataset_video/202504061115_cam1.mp4",
    "dataset_video/202504061115_cam2.mp4",
    "dataset_video/202504061143_cam1.mp4",
    "dataset_video/202504061143_cam2.mp4",
    "dataset_video/202504061200_cam1.mp4",
    "dataset_video/202504061200_cam2.mp4",
    "dataset_video/1733542045385.mp4",
    "dataset_video/1733542264653.mp4",
    "dataset_video/1733549485394.mp4",
    "dataset_video/1733555190328.mp4",
]

def non_overlap_area_fraction(rect_a, rect_list):
    """
    rect_a: 対象となる四角形 (x1, y1, x2, y2)
    rect_list: 重なり候補となる四角形のリスト [(x1, y1, x2, y2), ...]
    四角形Aの面積を1と正規化したとき、
    rect_list に含まれる四角形と重なっていない割合を返します。
    """
    poly_a = box(*rect_a)
    overlap_polys = []
    for rect in rect_list:
        poly = box(*rect)
        inter = poly.intersection(poly_a)
        if not inter.is_empty:
            overlap_polys.append(inter)
    if overlap_polys:
        union_overlap = unary_union(overlap_polys)
        overlap_area = union_overlap.area
    else:
        overlap_area = 0.0
    area_a = poly_a.area
    if area_a == 0:
        non_overlap_fraction = 0
    else :
        non_overlap_fraction = (area_a - overlap_area) / area_a
    return non_overlap_fraction

def point_in_box(x, y, rect):
    """ rect: (x1, y1, x2, y2) """
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def count_non_overlapped_keypoints(bbox, pose_results, other_boxes):
    """
    bbox: 対象領域 (x1, y1, x2, y2)
    pose_results: 各人物のキーポイント情報。
                  構造は [person_pose, ...] で、person_pose は [(x, y), ...] のリストです。
    other_boxes: bbox以外の重なり候補のリスト [(x1, y1, x2, y2), ...]
    bbox内にあるが、他の領域に含まれないキーポイントの数を返します。
    """
    count = 0
    for person_pose in pose_results:
        for keypoints in person_pose:
            for x, y in keypoints:
                if point_in_box(x, y, bbox):
                    if not any(point_in_box(x, y, other) for other in other_boxes):
                        count += 1
    return count

def compute_clip_score(bbox, pose_results, other_boxes):
    """
    bbox: (x1, y1, x2, y2)
    pose_results: 各人物のキーポイント情報（count_non_overlapped_keypoints と同じ構造）
    other_boxes: bbox以外の候補領域のリスト [(x1, y1, x2, y2), ...]
    
    被っていないエリアのキーポイント数と、非重なり面積の割合を掛け合わせたスコアを返します。
    また、非重なり面積の割合 (fraction) も返します。
    """
    fraction = non_overlap_area_fraction(bbox, other_boxes)
    non_overlap_count = count_non_overlapped_keypoints(bbox, pose_results, other_boxes)
    score = non_overlap_count * fraction
    return score, fraction

if __name__ == "__main__":
    # データセット作成のためのベースディレクトリを指定（例: "dataset_crops"）
    base_dataset_dir = Path("dataset_crops")
    base_dataset_dir.mkdir(exist_ok=True, parents=True)
    
    # 各動画ファイルごとに処理を実施
    for video_path in VIDEO_PATHS:
        print(f"処理中の動画: {video_path}")
        # 動画ファイル名（拡張子なし）を取得し、保存先ディレクトリとする
        video_stem = Path(video_path).stem
        video_dir = base_dataset_dir / video_stem
        video_dir.mkdir(exist_ok=True, parents=True)
        
        # 初期化：人物検出・追跡用モジュールを生成
        detector_tracker = PersonDetectorTracker()
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"動画ファイルを開けませんでした: {video_path}")
            continue
        
        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # フレーム番号（ファイル名に利用）
            frame_index += 1

            # 1. YOLO11+Deep SORT による人物検出と追跡ID付与
            results_with_id = detector_tracker.get_person_area_and_trackid(frame)
            detection_boxes = []  # 各検出領域 (x1, y1, x2, y2) のリスト
            for person, track_id in results_with_id:
                detection_boxes.append((person.x1, person.y1, person.x2, person.y2))
            
            # 2. ポーズ推定の実施
            pose_results = detector_tracker.get_person_pose(frame)
            
            # 3. 各検出領域ごとにclip scoreを計算
            for i, (person, track_id) in enumerate(results_with_id):
                bbox = (person.x1, person.y1, person.x2, person.y2)
                # 自分以外のbboxリストを作成
                other_boxes = detection_boxes[:i] + detection_boxes[i+1:]
                score, fraction = compute_clip_score(bbox, pose_results, other_boxes)
                
                # スコアが低い／重なりが大きい場合は保存しない（ここでは fraction > 0.9 の場合のみ採用）
                if fraction <= 0.9:
                    continue
                
                # 4. 検出領域を切り出し
                crop = frame[person.y1:person.y2, person.x1:person.x2]
                if crop.size == 0:
                    continue

                # 5. 保存先のディレクトリ作成: [動画]/[person_id]/
                person_dir = video_dir / f"{track_id}"
                person_dir.mkdir(exist_ok=True, parents=True)
                
                # 6. 画像を [動画]/[person_id]/[フレーム番号].png として保存
                save_path = person_dir / f"{frame_index:06d}.png"
                cv2.imwrite(str(save_path), crop)
                print(f"Saved: {save_path} (score: {score:.2f}, fraction: {fraction:.2f})")
            
            # オリジナルフレームのコピーを作成
            display_frame = frame.copy()
            
            # 検出された各人物について、バウンディングボックスと追跡IDを描画
            for person, track_id in results_with_id:
                cv2.rectangle(display_frame, (person.x1, person.y1), (person.x2, person.y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"ID: {track_id}", (person.x1, max(person.y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 各人物のポーズ推定結果からキーポイントを描画
            for person_pose in pose_results:
                for keypoints in person_pose:
                    for x, y in keypoints:
                        cv2.circle(display_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            # キャンバスに描画した結果を表示
            cv2.imshow("Frame", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        print(f"動画 {video_path} の処理が完了しました。")
    
    # cv2.destroyAllWindows()
    print("すべての動画処理が完了しました。")
