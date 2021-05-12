'''
PSE system code

ASSUMPTION 1: There's only one person - the player - in the frame

'''


import time
import cv2
import dlib
import numpy as np
import matplotlib.pyplot as plt
import paho.mqtt.client as mqtt

import EngagementModule
import MoodModule
import utils

# Log files to record the values of both engagement components for plotting later
# engagement_log_file = "high_eng.txt"



def main(ENGAGEMENT_WINDOW_SIZE, FILTERING_WINDOW_SIZE, MOOD_THRESH, ENGAGEMENT_THRESH, pse_mqtt_client):
    pose_estimator = EngagementModule.PoseEstimator()
    emotion_model = MoodModule.EmotionModel()

    cam = cv2.VideoCapture(0)

    # Lists of elbow angles
    l_elbow_angle_history = []
    r_elbow_angle_history = []

    # List of keypoint coord numpy arrays
    keypoint_coords_history = []


    # A short history of mood score for applying filtering
    mood_score_history = []
    ### For testing filtering
    test_mood_score_history = []
    test_filtered_mood_score_history = []
    ###

    while True:
        acquired_mood = False       # Flags to indicate if mood and engagement have been successfully obtained from the images.
        acquired_engagement = False

        t1 = time.time()
        success, frame = cam.read()
        if not success:
            raise IOError("webcam failure")

        # Pose/Engagement stuff -----------------------------------------------
        posenet_input, gray_frame = pose_estimator.get_inputs(frame)
        pose_scores, keypoint_scores, keypoint_coords =  pose_estimator.predict_poses(posenet_input)

        # Consider pose keypoints for only the 'main' person in the frame -- Under ASSUMPTION 1
        keypoint_coords[1:,:,:] = np.zeros_like(keypoint_coords[1:,:,:])
        keypoint_scores[1:,:] = np.zeros_like(keypoint_scores[1:,:])
        overlay_image = pose_estimator.draw_poses(frame, pose_scores, keypoint_scores, keypoint_coords)
        keypoint_coords = keypoint_coords[0]
        keypoint_coords_history.append(keypoint_coords)
        if len(keypoint_coords_history) > ENGAGEMENT_WINDOW_SIZE:
            keypoint_coords_history.pop(0)


        # Calculate elbow angle diffs
        l_elbow_angle_history, r_elbow_angle_history = utils.elbow_angle(l_elbow_angle_history, r_elbow_angle_history, keypoint_coords)
        if len(l_elbow_angle_history) > ENGAGEMENT_WINDOW_SIZE:
            l_elbow_angle_history.pop(0)
            r_elbow_angle_history.pop(0)
        if len(l_elbow_angle_history) == ENGAGEMENT_WINDOW_SIZE:
            l_elbow_angle_diff, r_elbow_angle_diff = utils.elbow_engagement(l_elbow_angle_history, r_elbow_angle_history)
            elbow_angle_diff = max(l_elbow_angle_diff, r_elbow_angle_diff)

        # Calculate upper body movement
        if len(keypoint_coords_history) == ENGAGEMENT_WINDOW_SIZE:
            avg_upper_body_movement = utils.upper_body_movement(keypoint_coords_history)
            # Scale the body movement so that it's same whether you are up close or far from the camera
            avg_upper_body_movement = utils.scale(avg_upper_body_movement, keypoint_coords)
            #print(avg_upper_body_movement)

        # Calculate the engagement score --
        if len(l_elbow_angle_history) == ENGAGEMENT_WINDOW_SIZE and len(keypoint_coords_history) == ENGAGEMENT_WINDOW_SIZE:
            engagement_score = utils.engagement_heuristic(elbow_angle_diff, avg_upper_body_movement)
            # with open(engagement_log_file,'a') as eng_log_file:
            #     eng_log_file.write(str(elbow_angle_diff) + ',' + str(avg_upper_body_movement) + '\n')

            print("Engagement score: ", engagement_score)
            if engagement_score > ENGAGEMENT_THRESH:
                engagement = 'high'
            else:
                engagement = 'low'

            acquired_engagement = True


        # # the following lines print the  outputs in the image for testing purposes, these can be removed on the final version
        # overlay_image = cv2.putText(overlay_image,
        #                             f'l_elbow_angle: {l_elbow_angle}',
        #                             (10,overlay_image.shape[0]-50),
        #                             cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.5,
        #                             (0,255,0),
        #                             1,
        #                             cv2.LINE_AA)
        # overlay_image = cv2.putText(overlay_image,
        #                             f'r_elbow_angle: {r_elbow_angle}',
        #                             (10,overlay_image.shape[0]-35),
        #                             cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.5,
        #                             (0,255,0),
        #                             1,
        #                             cv2.LINE_AA)
        # overlay_image = cv2.putText(overlay_image,
        #                             f'l_shoulder_ engagement: {l_shoulder_engaged}',
        #                             (10,overlay_image.shape[0]-20),
        #                             cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.5,
        #                             (0,255,0),
        #                             1,
        #                             cv2.LINE_AA)
        # overlay_image = cv2.putText(overlay_image,
        #                             f'r_shoulder_ engagement: {r_shoulder_engaged}',
        #                             (10,overlay_image.shape[0]-5),
        #                             cv2.FONT_HERSHEY_SIMPLEX,
        #                             0.5,
        #                             (0,255,0),
        #                             1,
        #                             cv2.LINE_AA)

        # Face/Emotion stuff --------------------------------------------------
        #cv2.imshow('original_frame', frame)
        gray_emotion_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('gray_emotion_frame', gray_emotion_frame)


        #TRYING TO IMPROVE EMOTION DETECTION RANGE
        #faces_rects = emotion_model._detect_faces(gray_emotion_frame)
        ##detector = dlib.full_object_detections(frame)
        ##faces = dlib.get_face_chips(frame, detector)
        #for (i, rect) in enumerate(faces_rects):
        #    # Finding points for rectangle to draw on face
        #    x1 = rect.left()
        #    y1 = rect.top()
        #    x2 = rect.right() + 1
        #    y2 = rect.bottom() + 1
        #    face = gray_emotion_frame[x1:x2, y1:y2]

        ##print(faces)
        ##print(len(faces))
        ##print(faces[len(faces)-1])
        #cv2.imshow('gray_face', face)

        # Get emotion and probs for all persons in the frame
        emotion, emotion_probabilities = emotion_model.predict_emotion(gray_emotion_frame)

        if len(emotion) > 0:
            # Consider emotion and probs for only the 'main' person in the frame -- Under ASSUMPTION 1
            emotion = emotion[0]
            emotion_probabilities = emotion_probabilities[0][0]

            # Calculate the mood score
            mood_score = utils.mood_heuristic(emotion_probabilities)
            mood_score_history.append(mood_score)
            if len(mood_score_history) > FILTERING_WINDOW_SIZE:
                mood_score_history.pop(0)
            test_mood_score_history.append(mood_score) ### for testing

            # Filter the recent mood score based on a short history of past scores
            mood_score = utils.apply_filter(mood_score_history, FILTERING_WINDOW_SIZE, filter_type='gaussian')
            test_filtered_mood_score_history.append(mood_score) ### for testing
            print("Mood score: ", mood_score)
            print("\n")
            if mood_score > MOOD_THRESH:
                mood = 'good'
            else:
                mood = 'bad'

            acquired_mood = True


        # Compose the PSE output message and send it through the 'pse_output' MQTT topic/channel
        if acquired_mood and acquired_engagement:
            pse_output_msg = mood + ',' + engagement # Separated by a comma
            print("Publishing PSE output to 'pse_output' topic: ", pse_output_msg)
            pse_mqtt_client.publish("topic/pse_output", pse_output_msg)


        overlay_image = emotion_model.draw_annotations(overlay_image)

        t2 = time.time()
        fps = 1/(t2-t1)
        cv2.putText(overlay_image, "FPS: {:.2f}".format(fps), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 1, cv2.LINE_AA)
        cv2.imshow('Output', overlay_image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
    del(pose_estimator)

    ### For testing
    plt.plot(test_mood_score_history, label='Mood score')
    plt.plot(test_filtered_mood_score_history, label='Filtered mood score')
    plt.xlabel("frames")
    plt.ylabel("mood score")
    plt.legend()
    plt.show()
    ###

if __name__ == "__main__":

   ENGAGEMENT_WINDOW_SIZE = 30  # Time window over which elbow angle and upper body movement are computed for engagement
   FILTERING_WINDOW_SIZE = 10   # Window over which the mood score is filtered

   MOOD_THRESH = 0.3
   ENGAGEMENT_THRESH = 0.5

   pse_mqtt_client = mqtt.Client()
   pse_mqtt_client.connect("localhost",1883,60)

   main(ENGAGEMENT_WINDOW_SIZE,
        FILTERING_WINDOW_SIZE,
        MOOD_THRESH,
        ENGAGEMENT_THRESH,
        pse_mqtt_client)