#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

from rclpy.qos import QoSProfile, ReliabilityPolicy

FEATURE_LEN = 16  # [valid, score, tri(6), poly4(8)]

# ------------------------------------------------------------------
# Reusable buffer node for integration (non-blocking spin in thread)
# ------------------------------------------------------------------
import threading
from typing import Dict, List

class DetectionInfoBuffer(Node):
    """ROS2 Node that subscribes to `detection_info` and stores latest detections.

    Public methods:
      get_latest() -> Dict[int, List[Dict]]

    Each detection dict contains: { 'score': float, 'tri': np.ndarray(3,2), 'poly4': np.ndarray(4,2) }.
    Camera indices are positional (0..num_cams-1). Remap externally if needed.
    """
    def __init__(self, topic_name: str = "detection_info"):
        super().__init__("detection_info_buffer")
        self._lock = threading.Lock()
        self._latest: Dict[int, List[Dict]] = {}

        qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self._sub = self.create_subscription(
            Float32MultiArray,
            topic_name,
            self._callback,
            qos,
        )

    def _callback(self, msg: Float32MultiArray):
        dims = msg.layout.dim
        if len(dims) != 3:
            return
        num_cams = dims[0].size
        max_det = dims[1].size
        feat_len = dims[2].size
        if feat_len != FEATURE_LEN:
            return
        data = np.asarray(msg.data, dtype=np.float32).reshape((num_cams, max_det, feat_len))
        result: Dict[int, List[Dict]] = {}
        for cam_i in range(num_cams):
            dets: List[Dict] = []
            for row in data[cam_i]:
                if row[0] <= 0.5:
                    continue
                score = float(row[1])
                tri = row[2:8].reshape(3, 2).copy()
                poly4 = row[8:16].reshape(4, 2).copy()
                dets.append({"score": score, "tri": tri, "poly4": poly4})
            result[cam_i] = dets
        with self._lock:
            self._latest = result

    def get_latest(self) -> Dict[int, List[Dict]]:
        with self._lock:
            return {k: list(v) for k, v in self._latest.items()}

def start_detection_info_buffer(topic_name: str = "detection_info"):
    """Creates DetectionInfoBuffer and starts a background spin thread.
    Returns (node, thread) so caller can stop via shutdown/ destroy_node.
    """
    buffer_node = DetectionInfoBuffer(topic_name=topic_name)
    spin_flag = {"run": True}

    def _spin():
        while spin_flag["run"] and rclpy.ok():
            rclpy.spin_once(buffer_node, timeout_sec=0.05)

    th = threading.Thread(target=_spin, daemon=True)
    th.start()
    return buffer_node, th, spin_flag


class DetectionSubscriber(Node):
    def __init__(self):
        super().__init__("detection_subscriber")

        self.sub = self.create_subscription(
            Float32MultiArray,
            "detection_info",
            self.callback,
            10,
        )

        self.get_logger().info("Python detection subscriber started.")

    def callback(self, msg: Float32MultiArray):

        dims = msg.layout.dim
        if len(dims) != 3:
            self.get_logger().error("Invalid dims")
            return

        num_cams    = dims[0].size
        max_det     = dims[1].size
        feature_len = dims[2].size

        if feature_len != FEATURE_LEN:
            self.get_logger().error(f"Expected feature_len {FEATURE_LEN}, got {feature_len}")
            return

        # ---- Raw data reshape ----
        data = np.array(msg.data, dtype=np.float32)
        data = data.reshape((num_cams, max_det, feature_len))

        # ---- per_cams 구조 구축 ----
        per_cams = {}

        for cam_id in range(num_cams):
            cam_data = data[cam_id]

            det_list = []

            for det_id in range(max_det):

                feat = cam_data[det_id]

                valid = feat[0] > 0.5
                if not valid:
                    continue

                score = float(feat[1])
                tri   = feat[2:8].reshape(3, 2)
                poly4 = feat[8:16].reshape(4, 2)

                det_list.append(
                    dict(
                        tri=tri,
                        poly4=poly4,
                        score=score
                    )
                )

            per_cams[cam_id] = det_list

        # ---- 확인용 출력 ----
        for cam_id, dets in per_cams.items():
            self.get_logger().info(f"[Cam {cam_id}] {len(dets)} detections")

            for idx, d in enumerate(dets):
                self.get_logger().info(
                    f" Det {idx}: score={d['score']:.3f}\n"
                    f"  tri=\n{d['tri']}\n"
                    f"  poly4=\n{d['poly4']}\n"
                )

def main():
    rclpy.init()
    node = DetectionSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
