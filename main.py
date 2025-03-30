from person_matching import is_same_person
from person_detect import PersonDetectorTracker
from pathlib import Path
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from shapely.geometry import box
from shapely.ops import unary_union
import matplotlib.pyplot as plt
import random

VIDEO_PATHS = [
    "dataset_youtube/segment_2.mp4",
    "dataset_youtube/segment_4.mp4",   
]

def non_overlap_area_fraction(rect_a, rect_list):
    """
    rect_a: 四角Aの座標 (x1, y1, x2, y2)
    rect_list: 四角Aに対して重なり得る複数の四角のリスト [(x1, y1, x2, y2), ...]
    四角Aの面積を1と正規化したとき、rect_list に含まれる四角と重なっていないエリアの割合を返します。
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
    non_overlap_fraction = (area_a - overlap_area) / area_a
    return non_overlap_fraction

def point_in_box(x, y, rect):
    """ rect: (x1, y1, x2, y2) """
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def count_non_overlapped_keypoints(bbox, pose_results, other_boxes):
    """
    bbox: 対象領域 (x1, y1, x2, y2)
    pose_results: 各人物のキーポイント情報。構造は、[person_pose, ...] 
                  で、person_pose は [keypoints, ...]、keypoints は [(x, y), ...] です。
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
    pose_results: 各人物のキーポイント情報（構造は count_non_overlapped_keypoints と同様）
    other_boxes: bbox以外の候補領域のリスト [(x1, y1, x2, y2), ...]
    被っていないエリアのキーポイント数と、非重なり面積の割合を掛け合わせたスコアを返す。
    また、非重なり面積の割合 (fraction) も返す。
    """
    fraction = non_overlap_area_fraction(bbox, other_boxes)
    non_overlap_count = count_non_overlapped_keypoints(bbox, pose_results, other_boxes)
    score = non_overlap_count * fraction
    return score, fraction

if __name__ == "__main__":
    # video_crops: { video_path: { track_id: [(score, fraction, crop), ...] } }
    video_crops = {}
    
    for video_path in VIDEO_PATHS:
        detector_tracker = PersonDetectorTracker()
        video_crops[video_path] = {}
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"動画ファイルを開けませんでした: {video_path}")
            continue
        
        window_name = f"Tracking - {video_path}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # 追跡処理：人物検出およびID付与
            results_with_id = detector_tracker.get_person_area_and_trackid(frame)
            detection_boxes = []
            for person, track_id in results_with_id:
                detection_boxes.append((person.x1, person.y1, person.x2, person.y2))
            
            # ポーズ推定の実行
            pose_results = detector_tracker.get_person_pose(frame)
            
            # 各検出領域ごとにスコア計算
            clip_scores = []
            for i, (person, track_id) in enumerate(results_with_id):
                bbox = (person.x1, person.y1, person.x2, person.y2)
                other_boxes = detection_boxes[:i] + detection_boxes[i+1:]
                score, fraction = compute_clip_score(bbox, pose_results, other_boxes)
                clip_scores.append((score, fraction, person, track_id))

            # 各検出領域の保存（各トラックごとに上位3枚保持）
            for score, fraction, person, track_id in clip_scores:
                crop = frame[person.y1:person.y2, person.x1:person.x2]
                if crop.size == 0:
                    continue
                # もし非重なり割合が0.9以下なら採用しない
                if fraction <= 0.9:
                    continue
                
                if track_id not in video_crops[video_path]:
                    video_crops[video_path][track_id] = []
                video_crops[video_path][track_id].append((score, fraction, crop))
                # 同じスコアの場合はランダムな順序にするために先にシャッフル
                random.shuffle(video_crops[video_path][track_id])
                # スコアの高い順にソートして上位3枚のみ保持
                video_crops[video_path][track_id].sort(key=lambda x: x[0], reverse=True)
                video_crops[video_path][track_id] = video_crops[video_path][track_id][:3]
            
            # 表示用にframeのコピーを作成し、そこにbboxとキーポイントを描画する
            show_frame = frame.copy()
            # bboxの描画とIDの表示（矩形の上にIDを描画）
            for person, track_id in results_with_id:
                cv2.rectangle(show_frame, (person.x1, person.y1), (person.x2, person.y2), (0, 255, 0), 2)
                cv2.putText(show_frame, f"ID: {track_id}", (person.x1, max(person.y1 - 10, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            # キーポイントの描画
            for person_pose in pose_results:
                for keypoints in person_pose:
                    for x, y in keypoints:
                        cv2.circle(show_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
            
            cv2.imshow(window_name, show_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyWindow(window_name)
    cv2.destroyAllWindows()
    
    # 各動画ごとに保存された各トラックのクロップ画像枚数を確認
    for video, tracks in video_crops.items():
        print(f"動画: {video}")
        for track_id, crops in tracks.items():
            print(f"  ID: {track_id} - 保存枚数: {len(crops)}")
    
    # representative_matches の作成（キー: (video1, track_id1, video2, track_id2)、値: (crop1, crop2, best_cos_sim)）
    threshold = 0.7
    min_max_score = 5  # 各トラックの最大スコアがこれ未満の場合は比較対象外
    print("\n=== Cross-video person matching representative results ===")
    representative_matches = {}

    videos = list(video_crops.keys())
    for i in range(len(videos)):
        for j in range(i+1, len(videos)):
            video1 = videos[i]
            video2 = videos[j]
            for track_id1, crops1 in video_crops[video1].items():
                best_score1 = max(c[0] for c in crops1)
                if best_score1 < min_max_score:
                    continue
                for track_id2, crops2 in video_crops[video2].items():
                    best_score2 = max(c[0] for c in crops2)
                    if best_score2 < min_max_score:
                        continue
                    # 各画像組み合わせについて最高のcosine similarityを求める
                    best_cos_sim = -1.0
                    best_pair = None
                    for score1, frac1, crop1 in crops1:
                        for score2, frac2, crop2 in crops2:
                            cos_sim = is_same_person(crop1, crop2)
                            if cos_sim >= threshold and cos_sim > best_cos_sim:
                                best_cos_sim = cos_sim
                                best_pair = (crop1, crop2)
                    if best_pair is not None:
                        key = (video1, track_id1, video2, track_id2)
                        representative_matches[key] = (best_pair[0], best_pair[1], best_cos_sim)
                        print(f"Video '{video1}' ID {track_id1} and Video '{video2}' ID {track_id2}: Best Cosine similarity = {best_cos_sim:.4f}")

    # --- ここから1対1の対応に絞る処理 ---
    # 各候補を cosine similarity の降順にソートし、両方の動画側で未採用なら採用する（グリーディー法）
    sorted_candidates = sorted(representative_matches.items(), key=lambda x: x[1][2], reverse=True)
    final_matches = {}
    used_video1 = set()  # (video, track_id)
    used_video2 = set()  # (video, track_id)

    for key, (crop1, crop2, cos_sim) in sorted_candidates:
        video1, track_id1, video2, track_id2 = key
        if (video1, track_id1) in used_video1 or (video2, track_id2) in used_video2:
            continue
        final_matches[key] = (crop1, crop2, cos_sim)
        used_video1.add((video1, track_id1))
        used_video2.add((video2, track_id2))

    # --- 表示部分 (pyplot) ---
    n_matches = len(final_matches)
    if n_matches == 0:
        print(f"{threshold}以上のマッチングは見つかりませんでした。")
    else:
        fig, axes = plt.subplots(n_matches, 2, figsize=(10, 5 * n_matches))
        if n_matches == 1:
            axes = [axes]
        for idx, ((video1, track_id1, video2, track_id2), (crop1, crop2, cos_sim)) in enumerate(final_matches.items()):
            ax_left = axes[idx][0]
            ax_left.imshow(cv2.cvtColor(crop1, cv2.COLOR_BGR2RGB))
            ax_left.axis("off")
            ax_left.set_title(f"Video '{video1}'\nID {track_id1}")
            
            ax_right = axes[idx][1]
            ax_right.imshow(cv2.cvtColor(crop2, cv2.COLOR_BGR2RGB))
            ax_right.axis("off")
            ax_right.set_title(f"Video '{video2}'\nID {track_id2}\nCosine sim = {cos_sim:.4f}")
        
        plt.tight_layout()
        plt.show()
