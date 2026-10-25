"""Opt-in low-contrast ArUco decoding inside the existing cube mask.

Use the normal dictionary/error-correction limits. Return corners in the original
undistorted camera pixels; triangulation and all geometric gates remain unchanged.
"""
import cv2
import numpy as np
import atexit
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def configure_recovery(project):
    """Allow subsequent batch invocations to use a verified per-project profile.

    An explicit environment value always takes precedence, including disabling.
    This does not change an already running detector or the installed OpenCV.
    """
    if 'ARUCO_RECOVERY' in os.environ:
        return os.environ['ARUCO_RECOVERY'] == '1'
    project = Path(project)
    path = project / '.aruco-recovery-profile.json'
    if not path.is_file():
        return False
    profile = json.loads(path.read_text(encoding='utf-8'))
    if profile.get('version') != 1 or not profile.get('enabled'):
        return False
    for filename, expected in profile['verified_files'].items():
        if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Recovery profile integrity mismatch: ' + filename)
    compatibility = Path(profile['compatibility_path'])
    if not compatibility.is_absolute():
        raise ValueError('Recovery compatibility path must be absolute')
    os.environ['ARUCO_COMPAT_PATH'] = str(compatibility)
    print('ArUco verified production profile: ' + str(path), flush=True)
    return True


class LowContrastDetector:
    def __init__(self, dictionary):
        self.worker = None
        compatibility = os.environ.get('ARUCO_COMPAT_PATH')
        if compatibility:
            env = os.environ.copy()
            env.update(PYTHONPATH=compatibility, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            self.worker = subprocess.Popen(
                [sys.executable, '-s', str(Path(__file__).resolve()), '--worker'],
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, bufsize=1)
            ready = json.loads(self.worker.stdout.readline())
            if ready.get('opencv') != '4.11.0' or ready.get('dictionary_sha256') != hashlib.sha256(dictionary.bytesList.tobytes()).hexdigest():
                self.close()
                raise RuntimeError('Unverified ArUco compatibility worker: ' + str(ready))
            print('ArUco isolated worker: ' + json.dumps(ready), flush=True)
            atexit.register(self.close)
        self.detectors = []
        # Prefer the positive threshold: constant zero can extend a sharp marker's
        # black contour into a flat white margin. Retain zero only as fallback.
        for constant in (3, 0):
            parameters = cv2.aruco.DetectorParameters()
            parameters.adaptiveThreshWinSizeMax = 101
            parameters.adaptiveThreshWinSizeStep = 4
            parameters.adaptiveThreshConstant = constant
            parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            self.detectors.append(cv2.aruco.ArucoDetector(dictionary, parameters))

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                self.worker.wait(timeout=5)
            self.worker = None

    def detect(self, gray, mask, boxes=()):
        regions = []
        if mask is not None and mask.shape == gray.shape:
            ys, xs = np.where(mask == 2)
            if len(xs) > 16:
                regions.append([xs.min(), ys.min(), xs.max()+1, ys.max()+1])
        regions.extend(boxes)
        # Crop boundaries affect adaptive thresholding. Try both the historical
        # padded mask/box and its exact bounds without changing decoding limits.
        regions = [(box, padding) for box in regions for padding in (.3, 0)]
        for box, padding_ratio in regions:
            x0, y0, x1, y1 = np.asarray(box, dtype=float)
            if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
                raise ValueError('Invalid verified cube ROI')
            padding = max(x1-x0, y1-y0) * padding_ratio + (8 if padding_ratio else 0)
            left, top = max(0, int(x0-padding)), max(0, int(y0-padding))
            right, bottom = min(gray.shape[1], int(x1+padding)), min(gray.shape[0], int(y1+padding))
            if right <= left or bottom <= top:
                continue
            region = gray[top:bottom, left:right]
            if self.worker is not None:
                request = {'shape': region.shape, 'gray': base64.b64encode(region.tobytes()).decode('ascii')}
                self.worker.stdin.write(json.dumps(request)+'\n')
                self.worker.stdin.flush()
                response = json.loads(self.worker.stdout.readline())
                ids = np.asarray(response['ids'], dtype=np.int32).reshape(-1, 1) if response['ids'] else None
                corners = [np.asarray(c, dtype=np.float32) for c in response['corners']]
            else:
                corners, ids = self.detect_region(region)
            if ids is not None:
                return [c + np.array([left, top], dtype=np.float32) for c in corners], ids
        return None, None

    def detect_region(self, region):
        for scale in (1, 2):
            enlarged = cv2.resize(region, None, fx=scale, fy=scale,
                                  interpolation=cv2.INTER_CUBIC)
            for detector in self.detectors:
                corners, ids, _ = detector.detectMarkers(enlarged)
                if ids is not None and np.isin(ids, (0, 1)).any():
                    corners = [corner / scale for corner in corners]
                    return corners, ids
        return None, None


def verified_boxes(scene):
    """Only same-frame DINO selections from this scene; never borrow a nearby box.

    The selected-frame hash binds the report to the current undistorted images.
    These boxes extend detector search only; masks and geometry are unchanged.
    """
    scene = Path(scene)
    report_path = scene / 'auto_rubik_keyframe.json'
    if not report_path.is_file():
        return {}
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if Path(report['scene']).name != scene.name:
        raise ValueError('Cube ROI report belongs to another scene')
    selected = scene / 'images' / report['selected_frame']
    if hashlib.sha256(selected.read_bytes()).hexdigest() != report['selected_frame_sha256']:
        raise ValueError('Stale cube ROI report')
    return {r['image_name']: [obj['box'] for obj in r['selected_rubiks']]
            for r in report['ranking'] if r.get('selected_rubiks')}


if __name__ == '__main__':
    if sys.argv[1:] != ['--worker'] or cv2.__version__ != '4.11.0':
        raise RuntimeError('Worker requires isolated OpenCV 4.11.0')
    # Prevent recursive worker creation. No detector acceptance parameter changes.
    os.environ.pop('ARUCO_COMPAT_PATH', None)
    cv2.setNumThreads(1)
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = LowContrastDetector(dictionary)
    print(json.dumps({'opencv': cv2.__version__, 'module': cv2.__file__,
                      'dictionary_sha256': hashlib.sha256(dictionary.bytesList.tobytes()).hexdigest(),
                      'error_correction_rate': cv2.aruco.DetectorParameters().errorCorrectionRate}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        gray = np.frombuffer(base64.b64decode(request['gray']), dtype=np.uint8).reshape(request['shape'])
        corners, ids = detector.detect_region(gray)
        print(json.dumps({'ids': [] if ids is None else ids.ravel().tolist(),
                          'corners': [] if ids is None else [c.tolist() for c in corners]}), flush=True)
