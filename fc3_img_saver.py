import cv2
import pyzed.sl as sl
import numpy as np
from numpy.linalg import norm
import pymongo
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
import os
import math
import base64
import time

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "face_db"
COLLECTION_NAME = "identities"
GFPGAN_MODEL_PATH = "GFPGANv1.3.pth"
SIMILARITY_THRESHOLD = 0.5


# ==========================================
# PART 1: Super Resolution Engine
# ==========================================
class FaceEnhancer:
    def __init__(self):
        self.restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )

    def enhance(self, full_frame, box):
        x1, y1, x2, y2 = box
        h, w, _ = full_frame.shape
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = full_frame[y1:y2, x1:x2]
        if face_crop.size == 0: return None

        try:
            _, restored_faces, _ = self.restorer.enhance(face_crop, has_aligned=False, only_center_face=False,
                                                         paste_back=False)
            if restored_faces:
                return restored_faces[0]
        except Exception:
            pass
        return None


# ==========================================
# PART 2: Database Logic
# ==========================================
class SmartFaceDB:
    def __init__(self):
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.known_faces = []
        self.next_id = 1
        self.last_refresh = 0
        self.refresh_local_cache()

    def refresh_local_cache(self):
        # Refresh every 5 seconds to pick up renames from Dashboard
        if time.time() - self.last_refresh < 5:
            return

        self.known_faces = []
        cursor = self.collection.find()
        max_id = 0
        for doc in cursor:
            embedding = np.array(doc['embedding'], dtype=np.float32)
            self.known_faces.append({
                'name': doc['name'],
                'embedding': embedding / norm(embedding),
                'age': doc.get('age', 0),
                'gender': doc.get('gender', 'U')
            })
            if doc['name'].startswith("Person_"):
                try:
                    curr_id = int(doc['name'].split("_")[1])
                    if curr_id > max_id: max_id = curr_id
                except ValueError:
                    pass
        self.next_id = max_id + 1
        self.last_refresh = time.time()
        # Silent update (don't spam console)

    def find_or_create(self, new_embedding, age, gender, distance, pose_status, face_img):
        # Check for updates from Dashboard
        self.refresh_local_cache()

        new_norm = new_embedding / norm(new_embedding)

        current_threshold = SIMILARITY_THRESHOLD
        if distance > 1.5: current_threshold = 0.45

        best_score = -1
        best_match = None

        for face in self.known_faces:
            score = np.dot(new_norm, face['embedding'])
            if score > best_score:
                best_score = score
                best_match = face

        if best_score > current_threshold:
            return best_match['name'], best_score, best_match['gender'], best_match['age'], (0, 255, 0)

        if distance > 1.5:
            return "Too Far", best_score, "U", 0, (0, 0, 255)

        if pose_status != "Center":
            return "Turn Head!", best_score, "U", 0, (0, 255, 255)

        # --- SAVE THUMBNAIL ---
        img_str = ""
        try:
            if face_img is not None and face_img.size > 0:
                # Resize to max 100x100 to save space
                img_h, img_w, _ = face_img.shape
                scale = min(100 / img_w, 100 / img_h)
                if scale < 1:
                    face_img = cv2.resize(face_img, (0, 0), fx=scale, fy=scale)

                _, buffer = cv2.imencode('.jpg', face_img)
                img_str = base64.b64encode(buffer).decode('utf-8')
        except Exception:
            pass

        new_name = f"Person_{self.next_id}"
        gender_str = "M" if gender == 1 else "F"

        new_doc = {
            "name": new_name, "embedding": new_embedding.tolist(),
            "age": int(age), "gender": gender_str,
            "thumbnail": img_str, "created_at": "now"
        }
        self.collection.insert_one(new_doc)

        self.known_faces.append({
            'name': new_name, 'embedding': new_norm, 'age': int(age), 'gender': gender_str
        })
        self.next_id += 1
        return new_name, best_score, gender_str, int(age), (0, 255, 0)


# ==========================================
# PART 3: Helper Functions
# ==========================================
def estimate_yaw(landmarks):
    if landmarks is None: return "Unknown"
    left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
    dist_left = nose[0] - left_eye[0]
    total_dist = right_eye[0] - left_eye[0]
    if total_dist == 0: return "Center"
    ratio = dist_left / total_dist
    if ratio < 0.35:
        return "Right"
    elif ratio > 0.65:
        return "Left"
    return "Center"


def is_inside_zed_box(face_box, zed_objects):
    fx1, fy1, fx2, fy2 = face_box
    face_area = (fx2 - fx1) * (fy2 - fy1)

    for obj in zed_objects:
        if obj.label != sl.OBJECT_CLASS.PERSON:
            continue

        raw_box = obj.bounding_box_2d
        zx1 = min(raw_box[0][0], raw_box[3][0])
        zy1 = min(raw_box[0][1], raw_box[1][1])
        zx2 = max(raw_box[1][0], raw_box[2][0])
        zy2 = max(raw_box[2][1], raw_box[3][1])

        ix1 = max(fx1, zx1)
        iy1 = max(fy1, zy1)
        ix2 = min(fx2, zx2)
        iy2 = min(fy2, zy2)

        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            if intersection / face_area > 0.3:
                return True, obj.id

    return False, None


# ==========================================
# PART 4: Main Loop
# ==========================================
def main():
    print("Loading AI Models...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    if not os.path.exists(GFPGAN_MODEL_PATH):
        print(f"❌ Error: {GFPGAN_MODEL_PATH} not found!")
        return
    enhancer = FaceEnhancer()
    db = SmartFaceDB()

    print("Opening ZED Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ ZED Error")
        return

    print("Enabling Positional Tracking...")
    pos_tracking_param = sl.PositionalTrackingParameters()
    pos_tracking_param.enable_area_memory = False
    if zed.enable_positional_tracking(pos_tracking_param) != sl.ERROR_CODE.SUCCESS:
        print("❌ Positional Tracking Failed")
        return

    print("Enabling ZED Object Detection...")
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
    obj_param.enable_tracking = True
    obj_param.enable_segmentation = False

    if zed.enable_object_detection(obj_param) != sl.ERROR_CODE.SUCCESS:
        print("❌ Object Detection Failed to Enable")
        return

    rt_params = sl.RuntimeParameters()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    objects = sl.Objects()

    print("\n✅ System Running!")

    while True:
        if zed.grab(rt_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)
            zed.retrieve_objects(objects, obj_runtime_param)

            frame_bgra = image_zed.get_data()
            frame_bgr = frame_bgra[:, :, :3].copy()

            faces = app.get(frame_bgr)

            for face in faces:
                box = face.bbox.astype(int)
                face_width = box[2] - box[0]

                # Check Reality (ZED Cross-Reference)
                is_real_person, zed_id = is_inside_zed_box(box, objects.object_list)

                if not is_real_person:
                    cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, "FAKE / NO BODY", (box[0], box[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue

                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                err, distance = depth_zed.get_value(center_x, center_y)
                if not np.isfinite(distance) or distance <= 0: distance = 10.0

                pose_status = estimate_yaw(face.kps)

                embedding_to_use = face.embedding
                was_enhanced = False

                # --- CAPTURE CROP FOR DASHBOARD ---
                # Ensure coordinates are within image bounds
                h, w, _ = frame_bgr.shape
                cx1, cy1 = max(0, box[0]), max(0, box[1])
                cx2, cy2 = min(w, box[2]), min(h, box[3])
                face_crop = frame_bgr[cy1:cy2, cx1:cx2]

                # Super Resolution Logic
                if face_width < 60 or distance > 2.0:
                    restored_face = enhancer.enhance(frame_bgr, box)
                    if restored_face is not None:
                        results_high_res = app.get(restored_face)
                        if len(results_high_res) > 0:
                            embedding_to_use = results_high_res[0].embedding
                            was_enhanced = True
                            # If enhanced, save the nice enhanced image instead of the blurry one
                            face_crop = restored_face

                name, score, gender, age, color = db.find_or_create(
                    embedding_to_use, face.age, face.gender, distance, pose_status, face_crop
                )

                cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)
                if was_enhanced:
                    cv2.circle(frame_bgr, (box[0], box[1]), 5, (255, 255, 0), -1)

                label_top = f"{name} ({distance:.1f}m)"
                cv2.putText(frame_bgr, label_top, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                label_bot = f"{pose_status} | Conf: {score:.2f}"
                cv2.putText(frame_bgr, label_bot, (box[0], box[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("ZED Final System", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()