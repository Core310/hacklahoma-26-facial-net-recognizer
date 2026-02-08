# Sends face over socket

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
from collections import deque, Counter
from flask import Flask, Response
import threading

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "face_db"
COLLECTION_NAME = "identities"
GFPGAN_MODEL_PATH = "GFPGANv1.3.pth"
SIMILARITY_THRESHOLD = 0.5
BLUR_THRESHOLD = 200

# --- FLASK SETUP ---
app = Flask(__name__)
output_frame = None
lock = threading.Lock()


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
# PART 2: Trackers & Buffers
# ==========================================
class FaceTracker:
    def __init__(self, max_history=10):
        self.max_history = max_history
        self.history = {}

    def update(self, zed_id, name):
        if zed_id not in self.history:
            self.history[zed_id] = deque(maxlen=self.max_history)
        self.history[zed_id].append(name)

    def get_stable_name(self, zed_id):
        if zed_id not in self.history or not self.history[zed_id]:
            return "Scanning..."
        votes = Counter(self.history[zed_id])
        winner, count = votes.most_common(1)[0]
        if count / len(self.history[zed_id]) > 0.5:
            return winner
        else:
            return "Verifying..."


class RegistrationBuffer:
    def __init__(self, frames_needed=15):
        self.frames_needed = frames_needed
        self.pending_counts = {}

    def update(self, zed_id):
        if zed_id not in self.pending_counts:
            self.pending_counts[zed_id] = 0
        self.pending_counts[zed_id] += 1
        if self.pending_counts[zed_id] > self.frames_needed:
            return True
        return False

    def reset(self, zed_id):
        if zed_id in self.pending_counts:
            del self.pending_counts[zed_id]


# ==========================================
# PART 3: Database Logic
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
        if time.time() - self.last_refresh < 2: return

        self.known_faces = []
        cursor = self.collection.find()
        max_id = 0
        for doc in cursor:
            embedding = np.array(doc['embedding'], dtype=np.float32)
            self.known_faces.append({
                'name': doc['name'],
                'embedding': embedding / norm(embedding),
                'age': doc.get('age', 0),
                'gender': doc.get('gender', 'U'),
                'linkedin': doc.get('linkedin', '')  # <--- LOAD LINK
            })
            if doc['name'].startswith("Person_"):
                try:
                    curr_id = int(doc['name'].split("_")[1])
                    if curr_id > max_id: max_id = curr_id
                except ValueError:
                    pass
        self.next_id = max_id + 1
        self.last_refresh = time.time()

    def find_match(self, new_embedding):
        self.refresh_local_cache()
        new_norm = new_embedding / norm(new_embedding)
        best_score = -1
        best_match = None

        for face in self.known_faces:
            if face['name'] == "IGNORE":
                if np.dot(new_norm, face['embedding']) > 0.6:
                    return "Ignored", 0.0, "U", 0, "", (128, 128, 128)
                continue

            score = np.dot(new_norm, face['embedding'])
            if score > best_score:
                best_score = score
                best_match = face

        if best_score > SIMILARITY_THRESHOLD:
            # Return Link as well
            return best_match['name'], best_score, best_match['gender'], best_match['age'], best_match['linkedin'], (0,
                                                                                                                     255,
                                                                                                                     0)

        return None, best_score, "U", 0, "", (0, 0, 255)

    def register_new(self, new_embedding, age, gender, face_img):
        new_norm = new_embedding / norm(new_embedding)
        img_str = ""
        try:
            if face_img is not None and face_img.size > 0:
                h, w, _ = face_img.shape
                scale = min(100 / w, 100 / h)
                if scale < 1: face_img = cv2.resize(face_img, (0, 0), fx=scale, fy=scale)
                _, buffer = cv2.imencode('.jpg', face_img)
                img_str = base64.b64encode(buffer).decode('utf-8')
        except:
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
        print(f"✅ Registered New User: {new_name}")
        return new_name


# ==========================================
# PART 4: Helper Functions
# ==========================================
def get_blur_score(image):
    if image is None or image.size == 0: return 0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


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
        if obj.label != sl.OBJECT_CLASS.PERSON: continue
        raw_box = obj.bounding_box_2d
        zx1 = min(raw_box[0][0], raw_box[3][0])
        zy1 = min(raw_box[0][1], raw_box[1][1])
        zx2 = max(raw_box[1][0], raw_box[2][0])
        zy2 = max(raw_box[2][1], raw_box[3][1])

        ix1, iy1 = max(fx1, zx1), max(fy1, zy1)
        ix2, iy2 = min(fx2, zx2), min(fy2, zy2)

        if ix2 > ix1 and iy2 > iy1:
            intersection = (ix2 - ix1) * (iy2 - iy1)
            if intersection / face_area > 0.3:
                return True, obj.id
    return False, None


# ==========================================
# PART 5: WEB STREAMING LOGIC
# ==========================================
def generate():
    global output_frame, lock

    blank_image = np.zeros((640, 640, 3), dtype=np.uint8)
    cv2.putText(blank_image, "WAITING FOR CAMERA...", (50, 320),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    while True:
        with lock:
            if output_frame is None:
                frame_to_send = blank_image
            else:
                frame_to_send = output_frame

            (flag, encodedImage) = cv2.imencode(".jpg", frame_to_send)
            if not flag: continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start_flask():
    try:
        app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
    except Exception as e:
        print(f"❌ Flask failed to start: {e}")


# ==========================================
# PART 6: Main Loop
# ==========================================
def main():
    global output_frame, lock

    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()
    print("🌐 Web Stream active at: http://localhost:5001/video_feed")

    print("Loading AI Models...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    if not os.path.exists(GFPGAN_MODEL_PATH):
        print(f"❌ Error: {GFPGAN_MODEL_PATH} not found!")
        return
    enhancer = FaceEnhancer()
    db = SmartFaceDB()
    tracker = FaceTracker(max_history=10)
    reg_buffer = RegistrationBuffer(frames_needed=10)

    print("Opening ZED Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init_params.coordinate_units = sl.UNIT.METER
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS: return

    pos_tracking_param = sl.PositionalTrackingParameters()
    pos_tracking_param.enable_area_memory = False
    zed.enable_positional_tracking(pos_tracking_param)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.MULTI_CLASS_BOX_MEDIUM
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

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

                # Check Reality
                is_real_person, zed_id = is_inside_zed_box(box, objects.object_list)
                if not is_real_person:
                    cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, "FAKE", (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    continue

                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                err, distance = depth_zed.get_value(center_x, center_y)
                if not np.isfinite(distance) or distance <= 0: distance = 10.0

                pose_status = estimate_yaw(face.kps)

                h, w, _ = frame_bgr.shape
                cx1, cy1 = max(0, box[0]), max(0, box[1])
                cx2, cy2 = min(w, box[2]), min(h, box[3])
                face_crop = frame_bgr[cy1:cy2, cx1:cx2]
                blur_score = get_blur_score(face_crop)

                embedding_to_use = face.embedding
                was_enhanced = False
                if face_width < 60 or distance > 2.0:
                    restored_face = enhancer.enhance(frame_bgr, box)
                    if restored_face is not None:
                        results_high_res = app.get(restored_face)
                        if len(results_high_res) > 0:
                            embedding_to_use = results_high_res[0].embedding
                            was_enhanced = True
                            face_crop = restored_face
                            blur_score = 999

                name, score, gender, age, link, color = db.find_match(embedding_to_use)

                if name:
                    reg_buffer.reset(zed_id)
                    tracker.update(zed_id, name)
                    final_name = tracker.get_stable_name(zed_id)
                else:
                    final_name = "Unknown"
                    color = (0, 165, 255)
                    if zed_id is not None:
                        if blur_score > BLUR_THRESHOLD and pose_status == "Center":
                            ready_to_register = reg_buffer.update(zed_id)
                            if ready_to_register:
                                new_name = db.register_new(embedding_to_use, face.age, face.gender, face_crop)
                                tracker.update(zed_id, new_name)
                                final_name = new_name
                                reg_buffer.reset(zed_id)

                cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)
                if was_enhanced: cv2.circle(frame_bgr, (box[0], box[1]), 5, (255, 255, 0), -1)

                # Name Label
                label_top = f"{final_name} ({distance:.1f}m)"
                cv2.putText(frame_bgr, label_top, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # --- LINK DRAWING LOGIC ---
                if link and final_name != "Unknown":
                    # Shorten the URL
                    short_link = link.replace("https://", "").replace("http://", "").replace("www.", "").replace(
                        "linkedin.com/", "")
                    if short_link.endswith("/"): short_link = short_link[:-1]

                    # Draw Cyan text ABOVE the name
                    cv2.putText(frame_bgr, short_link, (box[0], box[1] - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # Bottom Stats
                label_bot = f"Conf: {score:.2f} | Blur: {int(blur_score)} | {pose_status}"
                cv2.putText(frame_bgr, label_bot, (box[0], box[3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200),
                            1)

            with lock:
                output_frame = frame_bgr.copy()

            cv2.imshow("ZED Final System", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()