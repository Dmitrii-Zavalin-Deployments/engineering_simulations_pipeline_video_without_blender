import cv2
import numpy as np
import unittest

class TestFluidMotion(unittest.TestCase):
    def setUp(self):
        """Load video file"""
        self.video = cv2.VideoCapture("data/testing-input-output/simulation_final_video_no_blender.mp4")

    def test_optical_flow_tracking(self):
        """Ensure fluid follows expected motion path"""
        success, prev_frame = self.video.read()
        while success:
            success, next_frame = self.video.read()
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

            flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            avg_motion_x = np.mean(flow[..., 0])

            assert abs(avg_motion_x) < 0.1, "Fluid motion deviates from expected behavior!"
            prev_frame = next_frame

if __name__ == "__main__":
    unittest.main()
