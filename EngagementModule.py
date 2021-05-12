import tensorflow as tf
import posenet


###############################################################################
# Pose estimator class

class PoseEstimator():
    def __init__(self, model_id=101, scale_factor= 0.7125):
        self.sess = tf.Session()
        posenet_model_cfg, posenet_model_outputs = posenet.load_model(model_id, self.sess)
        self.posenet_model_outputs = posenet_model_outputs
        self.output_stride = posenet_model_cfg['output_stride']
        self.scale_factor = scale_factor

    def __del__(self):
        self.sess.close()

    def get_inputs(self, frame):
        posenet_input, gray_frame, output_scale = posenet.process_input(frame, self.scale_factor, self.output_stride)
        self.output_scale = output_scale
        return posenet_input, gray_frame

    def predict_poses(self, posenet_input):
        heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = self.sess.run( self.posenet_model_outputs,
                                                                                                      feed_dict={'image:0': posenet_input} )

        pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses( heatmaps_result.squeeze(axis=0),
                                                                                                    offsets_result.squeeze(axis=0),
                                                                                                    displacement_fwd_result.squeeze(axis=0),
                                                                                                    displacement_bwd_result.squeeze(axis=0),
                                                                                                    output_stride=self.output_stride,
                                                                                                    max_pose_detections=10,
                                                                                                    min_pose_score=0.15 )

        keypoint_coords *= self.output_scale
        return pose_scores, keypoint_scores, keypoint_coords


    def draw_poses(self, frame, pose_scores, keypoint_scores, keypoint_coords):
        overlay_image = posenet.draw_skel_and_kp( frame, pose_scores, keypoint_scores, keypoint_coords,
                                                  min_pose_score=0.15, min_part_score=0.1 )
        return overlay_image