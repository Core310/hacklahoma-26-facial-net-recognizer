import cv2
import pyzed.sl as sl
import numpy as np
from numpy.linalg import norm
import pymongo
from insightface.app import FaceAnalysis
import math

# --- Configuration ---
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "face_db"
COLLECTION_NAME = "identities"

# Standard threshold for close-up matches
BASE_THRESHOLD = 0.5


class SmartFaceDB:
    def __init__(self):
        self.client = pymongo.MongoClient(MONGO_URI)
        self.db = self.client[DB_NAME]
        self.collection = self.db[COLLECTION_NAME]
        self.known_faces = []
        self.next_id = 1
        self.refresh_local_cache()

    def refresh_local_cache(self):
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
        print(f"Loaded {len(self.known_faces)} faces. Next ID: {self.next_id}")

    def find_or_create(self, new_embedding, age, gender, distance):
        """
        Uses ZED distance to decide whether to Match, Create, or Ignore.
        """
        new_norm = new_embedding / norm(new_embedding)

        # --- LOGIC 1: DYNAMIC THRESHOLD ---
        # If closer than 1.5m, be strict (0.5).
        # If farther, be looser (0.4) because low-res faces are 'noisier'.
        current_threshold = BASE_THRESHOLD
        if distance > 1.5:
            current_threshold = 0.40

        best_score = -1
        best_match = None

        # Search Known Faces
        for face in self.known_faces:
            score = np.dot(new_norm, face['embedding'])
            if score > best_score:
                best_score = score
                best_match = face

        # MATCH FOUND?
        if best_score > current_threshold:
            return best_match['name'], best_score, best_match['gender'], best_match['age'], (0, 255, 0)

        # --- LOGIC 2: REGISTRATION GUARD ---
        # If NO match found, should we create a new person?

        # RULE: Only create new identities if the person is CLOSE (< 1.5m)
        # This prevents creating "Person_55" just because you stood 3m away.
        if distance > 1.5:
            # Too far to be sure. Return "Unknown" but DO NOT save to DB.
            return "Too Far to Register", best_score, "U", 0, (0, 0, 255)  # Red Color

        # If we are here, we are Close + No Match + High Quality -> Create New
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

        self.known_faces.append({
            'name': new_name,
            'embedding': new_norm,
            'age': int(age),
            'gender': gender_str
        })

        self.next_id += 1
        return new_name, best_score, gender_str, int(age), (0, 255, 0)


def main():
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    db = SmartFaceDB()

    # ZED Init
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    # IMPORTANT: Use NEURAL depth if available for better far-range accuracy
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL
    init_params.coordinate_units = sl.UNIT.METER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED Error")
        return

    rt_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()  # <--- Needed for distance

    print("Running... Press 'q' to quit.")

    while True:
        if zed.grab(rt_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)  # <--- Get Depth Map

            frame_bgra = image_zed.get_data()
            frame_bgr = frame_bgra[:, :, :3].copy()

            faces = app.get(frame_bgr)

            for face in faces:
                box = face.bbox.astype(int)

                # 1. GET REAL DISTANCE FROM ZED
                # Find center of face
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)

                # Handle edge cases (face partly off screen)
                h, w, _ = frame_bgr.shape
                center_x = max(0, min(center_x, w - 1))
                center_y = max(0, min(center_y, h - 1))

                err, distance = depth_zed.get_value(center_x, center_y)

                # If ZED returns 'NaN' or infinity, assume far away
                if not math.isfinite(distance):
                    distance = 10.0

                # 2. RUN INTELLIGENT DB CHECK
                name, score, gender, age, color = db.find_or_create(
                    face.embedding,
                    face.age,
                    face.gender,
                    distance  # <--- Pass distance to logic
                )

                # Visuals
                # Top: Name + Dist
                label_top = f"{name} ({distance:.1f}m)"
                cv2.putText(frame_bgr, label_top, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # Bottom: Score + Info
                label_bot = f"Conf: {score:.2f} | {gender} {age}"
                cv2.putText(frame_bgr, label_bot, (box[0], box[3] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)

            cv2.imshow("ZED Smart DB", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()