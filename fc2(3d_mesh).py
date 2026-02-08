import math

import cv2
import pyzed.sl as sl
import numpy as np
import mediapipe as mp

# --- Configuration ---
OUTPUT_FILE = "my_face_mesh.ply"


def save_point_cloud_ply(filename, points_3d):
    """
    Saves a list of [x, y, z] points to a standard PLY 3D file.
    You can open this file in Blender, MeshLab, or CloudCompare.
    """
    header = f"""ply
format ascii 1.0
element vertex {len(points_3d)}
property float x
property float y
property float z
end_header
"""
    with open(filename, 'w') as f:
        f.write(header)
        for p in points_3d:
            # Scale up by 100 for better visibility in 3D viewers (meters -> cm)
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    print(f"✅ Saved 3D Mesh to {filename}")


def main():
    # 1. Init MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # 2. Init ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # ULTRA is best for close-up geometry
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_minimum_distance = 0.3  # Allow close-up (30cm)

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("ZED Error")
        return

    rt_params = sl.RuntimeParameters()
    image_zed = sl.Mat()
    depth_zed = sl.Mat()
    point_cloud = sl.Mat()  # Can retrieve full XYZRGBA point cloud if needed

    print("Press 'S' to save the 3D mesh. 'Q' to quit.")

    while True:
        if zed.grab(rt_params) == sl.ERROR_CODE.SUCCESS:
            # Retrieve data
            zed.retrieve_image(image_zed, sl.VIEW.LEFT)
            zed.retrieve_measure(depth_zed, sl.MEASURE.DEPTH)

            # Prepare image for MediaPipe (RGB)
            frame_bgra = image_zed.get_data()
            frame_bgr = frame_bgra[:, :, :3]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Process Mesh
            results = face_mesh.process(frame_rgb)

            # Visualization
            display_frame = frame_bgr.copy()
            current_mesh_points = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    h, w, _ = frame_bgr.shape

                    # Loop through all 468 landmarks
                    for id, lm in enumerate(face_landmarks.landmark):
                        # Instead of manual calculation, ask ZED for the exact 3D point at this pixel
                        err, point3D = point_cloud.get_value(cx, cy)
                        x, y, z = point3D[0], point3D[1], point3D[2]

                        # Check if valid (ZED returns 'nan' for invalid points)
                        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
                            current_mesh_points.append([x, y, z])
                        # Get Pixel Coordinates (u, v)
                        cx, cy = int(lm.x * w), int(lm.y * h)

                        # Clamp to image bounds
                        cx = max(0, min(cx, w - 1))
                        cy = max(0, min(cy, h - 1))

                        # GET REAL DEPTH (Z) FROM ZED
                        err, depth_value = depth_zed.get_value(cx, cy)

                        if np.isfinite(depth_value) and depth_value > 0:
                            # Reproject 2D -> 3D using ZED intrinsics logic (simplified)
                            # Or just store raw [cx, cy, depth] for simple visualization

                            # To get true Metric 3D, we need the Camera Matrix (Intrinsics)
                            # But for a simple mesh, we can just use the raw coordinates + depth
                            # Here we create a simple 3D point:

                            # NOTE: In a real app, use zed.retrieve_measure(point_cloud, sl.MEASURE.XYZ)
                            # and sample the XYZ value directly at (cx, cy). That is more accurate.
                            # But this is "good enough" for testing.
                            current_mesh_points.append([lm.x, -lm.y, depth_value])

                            # Draw on screen
                        cv2.circle(display_frame, (cx, cy), 1, (0, 255, 255), -1)

            cv2.imshow("ZED 3D Scanner", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if current_mesh_points:
                    save_point_cloud_ply(OUTPUT_FILE, current_mesh_points)
                else:
                    print("⚠️ No face detected to save!")

    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()