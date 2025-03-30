from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, NamedTuple, Tuple
from deep_sort_realtime.deepsort_tracker import DeepSort

class PersonArea(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

class PersonDetectorTracker:
    def __init__(self, 
                 model_path: str = "models/yolo11l.pt",
                 pose_model_path: str = "models/yolo11l-pose.pt",
                 max_age: int = 30):
        # コンストラクタでYOLO11モデルとDeep SORTトラッカーをロード
        self.model = YOLO(model_path)
        self.model_pose = YOLO(pose_model_path)  # ポーズ推定用のモデル
        self.tracker = DeepSort(max_age=max_age)
    
    def get_person_area(self, img: np.ndarray) -> List[PersonArea]:
        """画像からYOLO11で人物検出を行い、PersonAreaリストを返す"""
        areas: List[PersonArea] = []
        results = self.model(img)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()   # [x1, y1, x2, y2]
            class_ids = result.boxes.cls.cpu().numpy()  # クラスID
            scores = result.boxes.conf.cpu().numpy()    # 信頼度スコア
            for box, cls_id, score in zip(boxes, class_ids, scores):
                if int(cls_id) == 0:  # 人物はCOCOでID 0
                    x1, y1, x2, y2 = map(int, box)
                    areas.append(PersonArea(x1, y1, x2, y2, score))
        return areas

    def get_person_area_and_trackid(self, img: np.ndarray) -> List[Tuple[PersonArea, int]]:
        """
        画像からYOLO11で人物検出を行い、Deep SORTで追跡IDを付与した結果を返す。
        trackerはクラス内で管理されるため、連続フレーム間で追跡が可能です。
        """
        # YOLO11で検出した人物（COCO ID=0）の結果を取得し、バウンディングボックス形式を[x1, y1, width, height, score]に変換
        detections = []  # 各検出: [x1, y1, width, height, score]
        results = self.model(img)
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            scores = result.boxes.conf.cpu().numpy()
            for box, cls_id, score in zip(boxes, class_ids, scores):
                if int(cls_id) == 0:
                    x1, y1, x2, y2 = map(int, box)
                    w = x2 - x1
                    h = y2 - y1
                    detections.append([[x1, y1, w, h], score])

        # Deep SORTトラッカーで追跡更新（フレーム間の連続追跡が可能）
        tracks = self.tracker.update_tracks(detections, frame=img)
        
        results_with_id = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            ltrb = track.to_ltrb()  # [x1, y1, x2, y2]
            track_id = track.track_id
            person_area = PersonArea(int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]), 0.0)
            results_with_id.append((person_area, track_id))
        return results_with_id
    
    def get_person_pose(self, img: np.ndarray) -> List[float]:
        """
        画像からYOLO11で人物検出を行い、ポーズ推定を行う。
        ポーズ推定の結果は、各関節の(x, y)座標をリストで返す。
        """
        results = self.model_pose(img)
        pose_results = []
        for result in results:
            keypoints = result.keypoints.cpu().numpy()
            for kp in keypoints:
                pose_results.append(kp.xy)
        return pose_results

if __name__ == "__main__":
    detector_tracker = PersonDetectorTracker()

    # 単一画像での検出結果表示例
    image_path = "dataset_imgs/example.jpg"
    img = cv2.imread(image_path)
    person_areas = detector_tracker.get_person_area(img)
    pose_results = detector_tracker.get_person_pose(img)
    for person in person_areas:
        x1, y1, x2, y2, _ = person
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    for person_pose in pose_results:
        for pose in person_pose:
            for x, y in pose:
                cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.imshow("Detected Persons", img)
    cv2.waitKey(0)

    # 動画での連続追跡例
    video_path = "dataset_video/1733542264653.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("動画ファイルを開けませんでした")
        exit()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results_with_id = detector_tracker.get_person_area_and_trackid(frame)
        for person, track_id in results_with_id:
            cv2.rectangle(frame, (person.x1, person.y1), (person.x2, person.y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (person.x1, max(person.y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        cv2.imshow("YOLO11 + Deep SORT Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
