import cv2
import mediapipe as mp
import numpy as np
import json

from custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=3)


# points_idx = [33, 263, 61, 291, 199]
# points_idx = points_idx + [key for (key, val) in procrustes_landmark_basis]
# points_idx = list(set(points_idx))
# points_idx.sort()

# uncomment next line to use all points for PnP algorithm
points_idx = list(range(0,468)); #points_idx[0:2] = points_idx[0:2:-1];
nose_idx = points_idx.index(4)

# dir = "/home/remmel/workspace/dataset/vv/kinects/mediapipe/k1-left/"
# frame_height, frame_width, channels = (480, 640, 3)
# focal_length = 525

dir = "/home/remmel/workspace/dataset/vv/kinects/mediapipe/k2-move/left/"
frame_height, frame_width, channels = (1080, 1920, 3)
focal_length = 1081

# pseudo camera internals
# focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

dist_coeff = np.zeros((4, 1))


def main():
    pcf = PCF(
        near=1, #1cm
        far=10000,
        frame_height=frame_height,
        frame_width=frame_width,
        fy=camera_matrix[1, 1],
    )

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as face_mesh:

        # for idx, (frame, frame_rgb) in enumerate(source):
        frame = cv2.imread(dir + "image.jpg")
        results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        multi_face_landmarks = results.multi_face_landmarks

        if multi_face_landmarks:
            face_landmarks = multi_face_landmarks[0]
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )
            # print(landmarks.shape)
            landmarks = landmarks.T

            metric_landmarks, pose_transform_mat = get_metric_landmarks(
                landmarks.copy(), pcf
            )
            model_points = metric_landmarks[0:3, points_idx].T
            image_points = (
                landmarks[0:2, points_idx].T
                * np.array([frame_width, frame_height])[None, :]
            )

            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeff,
                flags=cv2.cv2.SOLVEPNP_ITERATIVE,
            )

            exportToJson(face_landmarks, model_points, points_idx, rotation_vector, translation_vector, dir)

            (nose_end_point2D, jacobian) = cv2.projectPoints(
                np.array([(0.0, 0.0, 25.0)]),
                rotation_vector,
                translation_vector,
                camera_matrix,
                dist_coeff,
            )

            for face_landmarks in multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec,
                )

            p1 = (int(image_points[nose_idx][0]), int(image_points[nose_idx][1]))
            p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

            frame = cv2.line(frame, p1, p2, (255, 0, 0), 2)

            # source.show(frame)
            cv2.imshow("rgb", frame)
            cv2.waitKey()

def exportToJson(face_landmarks, model_points, points_idx, rotation_vector, translation_vector, dir):
    export = dict()
    export["points"] = []
    for i, model_point in enumerate(model_points):
        idx = points_idx[i]
        export["points"].append({
            'x': model_point[0]/100,
            'y': model_point[1]/100,
            'z': model_point[2]/100,
            'u': face_landmarks.landmark[idx].x,
            'v': face_landmarks.landmark[idx].y,
            'idx': idx
        })
    export["rotation"] = rotation_vector.T.tolist()[0]
    translation_vector_m = translation_vector / 100
    export["translation"] = translation_vector_m.T.tolist()[0]

    with open(dir + 'facemesh.json', 'w') as outfile:
        json.dump(export, outfile, indent=4)

if __name__ == "__main__":
    main()
