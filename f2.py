import cv2
import pyzed.sl as sl
import numpy as np
from numpy.linalg import norm
import pymongo
from insightface.app import FaceAnalysis
from gfpgan import GFPGANer
import os
import math

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
        # upscale=2 means it turns a 50px face into a 100px face
        self.restorer = GFPGANer(
            model_path=GFPGAN_MODEL_PATH,
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )

    def enhance(self, full_frame, box):
        """
        Crops the face, upscales it, and returns the restored face image.
        """
        x1, y1, x2, y2 = box
        h, w, _ = full_frame.shape

        # Add padding to help the model see context (hair/chin)
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        face_crop = full_frame[y1:y2, x1:x2]

        if face_crop.size == 0: return None

        # Run GFPGAN
        # cropped_faces, restored_faces, restored_img
        _, restored_faces, _ = self.restorer.enhance(face_crop, has_aligned=False, only_center_face=False,
                                                     paste_back=False)

        if restored_faces:
            return restored_faces[0]  # Return the first restored face found
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
        self.refresh_local_cache()

    def refresh_local_cache(self):
        """Loads all identities from MongoDB into RAM for fast matching."""
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
            # Track ID number to auto-increment Person_X
            if doc['name'].startswith("Person_"):
                try:
                    curr_id = int(doc['name'].split("_")[1])
                    if curr_id > max_id: max_id = curr_id
                except ValueError:
                    pass
        self.next_id = max_id + 1
        print(f"Loaded {len(self.known_faces)} faces. Next ID: {self.next_id}")

    def find_or_create(self, new_embedding, age, gender, distance, pose_status):
        """
        Matches face against DB.
        Creates new entry ONLY if:
        1. No match found
        2. User is close (< 1.5m)
        3. User is looking straight (Center)
        """
        new_norm = new_embedding / norm(new_embedding)

        # DYNAMIC THRESHOLD: Be looser if far away (0.45), stricter if close (0.5)
        current_threshold = SIMILARITY_THRESHOLD
        if distance > 1.5: current_threshold = 0.45

        best_score = -1
        best_match = None

        # 1. Search Known Faces
        for face in self.known_faces:
            score = np.dot(new_norm, face['embedding'])
            if score > best_score:
                best_score = score
                best_match = face

        # 2. MATCH FOUND?
        if best_score > current_threshold:
            return best_match['name'], best_score, best_match['gender'], best_match['age'], (0, 255, 0)

        # 3. REGISTRATION GUARDS
        # Don't create new person if too far
        if distance > 1.5:
            return "Too Far", best_score, "U", 0, (0, 0, 255)  # Red

        # Don't create new person if looking sideways (Bad data)
        if pose_status != "Center":
            return "Turn Head!", best_score, "U", 0, (0, 255, 255)  # Yellow

        # 4. CREATE NEW IDENTITY
        new_name = f"Person_{self.next_id}"
        gender_str = "M" if gender == 1 else "F"

        new_doc = {
            "name": new_name,
            "embedding": new_embedding.tolist(),
            "age": int(age),
            "gender": gender_str,
            "created_at": "now"
        }
        self.collection.insert_one(new_doc)

        # Add to local cache
        self.known_faces.append({
            'name': new_name,
            'embedding': new_norm,
            'age': int(age),
            'gender': gender_str
        })

        self.next_id += 1
        return new_name, best_score, gender_str, int(age), (0, 255, 0)  # Green


# ==========================================
# PART 3: Helper Functions
# ==========================================
def estimate_yaw(landmarks):
    """
    Estimates if head is turned Left/Right based on nose position relative to eyes.
    landmarks format: 5 points (LeftEye, RightEye, Nose, LeftMouth, RightMouth)
    """
    if landmarks is None: return "Unknown"

    left_eye = landmarks[0]
    right_eye = landmarks[1]
    nose = landmarks[2]

    dist_left = nose[0] - left_eye[0]
    total_dist = right_eye[0] - left_eye[0]

    if total_dist == 0: return "Center"

    ratio = dist_left / total_dist

    # Ratios: <0.35 means looking Right (camera perspective). >0.65 means looking Left.
    if ratio < 0.35:
        return "Right"
    elif ratio > 0.65:
        return "Left"
    else:
        return "Center"


# ==========================================
# PART 4: Main Loop
# ==========================================
def main():
    # 1. Initialize AI Models
    print("Loading ArcFace...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    print("Loading GFPGAN (Super Resolution)...")
    if not os.path.exists(GFPGAN_MODEL_PATH):
        print(f"❌ Error: {GFPGAN_MODEL_PATH} not found!")
        return
    enhancer = FaceEnhancer()

    print("Connecting to Database...")
    db = SmartFaceDB()

    # 2. Initialize ZED Camera
    print("Opening ZED Camera...")
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    # NEURAL depth mode is slower but much better for long-range accuracy
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ ZED Error: Could not open camera.")
        return

    rt_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    print("\n✅ System Running!")
    print("   [Q] Quit")

    while True:
        if zed.grab(rt_params) == sl.ERROR_CODE.SUCCESS:
            # A. Retrieve Data
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

            frame_bgra = image_zed.get_data()
            frame_bgr = frame_bgra[:, :, :3].copy()  # Fix memory layout

            # B. Detect Faces (Initial Pass)
            faces = app.get(frame_bgr)

            for face in faces:
                box = face.bbox.astype(int)
                face_width = box[2] - box[0]

                # --- 1. Get Distance (ZED) ---
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                h, w, _ = frame_bgr.shape
                center_x = max(0, min(center_x, w - 1))
                center_y = max(0, min(center_y, h - 1))

                err, distance = depth_zed.get_value(center_x, center_y)
                if not np.isfinite(distance) or distance <= 0: distance = 10.0

                # --- 2. Check Pose (Angle) ---
                pose_status = estimate_yaw(face.kps)

                # --- 3. Super Resolution (Distance Fix) ---
                embedding_to_use = face.embedding
                was_enhanced = False

                # Logic: If face is small (<60px) OR far away (>2m), try to fix it
                if face_width < 60 or distance > 2.0:
                    restored_face = enhancer.enhance(frame_bgr, box)

                    if restored_face is not None:
                        # Re-run ArcFace on the clean, upscaled face
                        results_high_res = app.get(restored_face)

                        if len(results_high_res) > 0:
                            # Success! Use the high-res embedding
                            embedding_to_use = results_high_res[0].embedding
                            was_enhanced = True

                # --- 4. Identify (Database) ---
                name, score, gender, age, color = db.find_or_create(
                    embedding_to_use, face.age, face.gender, distance, pose_status
                )

                # --- 5. Draw Visuals ---
                # Box
                cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Markers
                if was_enhanced:
                    # Cyan dot = Enhanced by AI
                    cv2.circle(frame_bgr, (box[0], box[1]), 5, (255, 255, 0), -1)

                    # Text Labels
                # Top: Name + Distance
                label_top = f"{name} ({distance:.1f}m)"
                cv2.putText(frame_bgr, label_top, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Bottom: Pose + Score
                label_bot = f"{pose_status} | Conf: {score:.2f}"
                cv2.putText(frame_bgr, label_bot, (box[0], box[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("ZED Smart ID System", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()