import cv2
import pyzed.sl as sl
import numpy as np
import os
import pickle
from numpy.linalg import norm
from insightface.app import FaceAnalysis


# ==========================================
# PART 1: The Face Matcher Engine
# ==========================================
class FaceMatcher:
    def __init__(self, db_path='face_db.pkl'):
        self.db_path = db_path
        self.known_faces = {}  # Format: {'name': embedding_norm}
        self.load_db()

    def register_face(self, name, embedding):
        """Saves a normalized embedding to the database."""
        # Normalize: This makes cosine similarity much faster/easier
        self.known_faces[name] = embedding / norm(embedding)
        self.save_db()
        print(f"✅ Registered: {name}")

    def match_face(self, target_embedding, threshold=0.4):
        """Compares input embedding against all known faces."""
        if not self.known_faces:
            return "Unknown", 0.0

        # Normalize the incoming face
        target_norm = target_embedding / norm(target_embedding)

        best_name = "Unknown"
        best_score = -1.0

        for name, known_vec in self.known_faces.items():
            # Cosine Similarity = Dot Product of normalized vectors
            score = np.dot(target_norm, known_vec)
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= threshold:
            return best_name, best_score
        return "Unknown", best_score

    def save_db(self):
        with open(self.db_path, 'wb') as f:
            pickle.dump(self.known_faces, f)

    def load_db(self):
        if os.path.exists(self.db_path):
            with open(self.db_path, 'rb') as f:
                self.known_faces = pickle.load(f)
            print(f"📂 Loaded {len(self.known_faces)} identities from database.")


# ==========================================
# PART 2: Main Execution Loop (ZED + ArcFace)
# ==========================================
def main():
    # 1. Initialize ArcFace (InsightFace)
    # providers=['CUDAExecutionProvider'] is CRITICAL for Jetson Orin performance
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    matcher = FaceMatcher()

    # 2. Initialize ZED Camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.AUTO
    init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # Use ULTRA for higher precision
    init_params.coordinate_units = sl.UNIT.CENTIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("❌ Failed to open ZED Camera")
        return

    # ZED Runtime Parameters
    runtime_parameters = sl.RuntimeParameters()

    # Pre-allocate ZED memory
    image_zed = sl.Mat()
    depth_zed = sl.Mat()

    print("\nControls:")
    print("  [R] - Register the largest detected face")
    print("  [Q] - Quit")

    while True:
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            # --- Capture Data ---
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

            # Convert to standard OpenCV format (Drop Alpha channel: BGRA -> BGR)
            frame_bgra = image_zed.get_data()
            frame_bgr = frame_bgra[:, :, :3].copy()  # Make a copy to ensure memory continuity

            # --- AI Inference ---
            faces = app.get(frame_bgr)

            # Sort faces by size (largest first) to handle the primary user easily
            faces.sort(key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]), reverse=True)

            for face in faces:
                box = face.bbox.astype(int)

                # 1. Depth Check (Anti-Spoofing Lite)
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                err, distance = depth_zed.get_value(center_x, center_y)

                # 2. Identify
                name, score = matcher.match_face(face.embedding)

                # 3. Visualization
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

                # Draw Box
                cv2.rectangle(frame_bgr, (box[0], box[1]), (box[2], box[3]), color, 2)

                # Draw Labels
                label_id = f"{name} ({score:.2f})"
                label_dist = f"Dist: {distance:.2f}m"

                cv2.putText(frame_bgr, label_id, (box[0], box[1] - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame_bgr, label_dist, (box[0], box[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("ZED 2i Face Matcher", frame_bgr)

            # --- Key Inputs ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            # Registration Mode: Press 'r' to save the current largest face
            elif key == ord('r'):
                if len(faces) > 0:
                    primary_face = faces[0]  # Largest face
                    # Ask user for name in console
                    new_name = input("Enter name for this person: ")
                    matcher.register_face(new_name, primary_face.embedding)
                else:
                    print("⚠️ No face detected to register!")

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()