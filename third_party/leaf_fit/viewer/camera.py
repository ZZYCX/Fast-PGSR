import os

if 'PYOPENGL_ACCELERATE' not in os.environ:
    os.environ['PYOPENGL_ACCELERATE'] = 'False'

import numpy as np
import glm

class Camera:
    def __init__(self, h, w):
        self.znear = 0.01
        self.zfar = 100
        self.h = h
        self.w = w
        self.fovy = np.pi / 2
        self.position = np.array([0.0, 0.0, 3.88]).astype(np.float32)
        self.target = np.array([0.0, 0.0, 0.0]).astype(np.float32)
        self.up = np.array([0.0, -1.0, 0.0]).astype(np.float32)
        self.yaw = -np.pi / 2
        self.pitch = 0
        
        self.is_pose_dirty = True
        self.is_intrin_dirty = True
        
        self.last_x = 640
        self.last_y = 360
        self.first_mouse = True
        
        self.is_leftmouse_pressed = False
        self.is_rightmouse_pressed = False
        self.is_ctrl_pressed = False
        self.is_middlemouse_pressed = False
        self.is_drag = False
        
        self.rot_sensitivity = 0.02
        self.trans_sensitivity = 0.01
        self.zoom_sensitivity = 0.08
        self.roll_sensitivity = 0.03
        self.target_dist = 3.88
    
    def _global_rot_mat(self):
        x = np.array([1, 0, 0])
        z = np.cross(x, self.up)
        z = z / np.linalg.norm(z)
        x = np.cross(self.up, z)
        return np.stack([x, self.up, z], axis=-1)

    def get_view_matrix(self):
        return np.array(glm.lookAt(self.position, self.target, self.up))

    def get_project_matrix(self):
        project_mat = glm.perspective(
            self.fovy,
            self.w / self.h,
            self.znear,
            self.zfar
        )
        return np.array(project_mat).astype(np.float32)

    def get_htanfovxy_focal(self):
        htany = np.tan(self.fovy / 2)
        htanx = htany / self.h * self.w
        focal = self.h / (2 * htany)
        return [htanx, htany, focal]

    def get_focal(self):
        return self.h / (2 * np.tan(self.fovy / 2))

    def process_mouse(self, xpos, ypos):
        if self.first_mouse:
            self.last_x = xpos
            self.last_y = ypos
            self.first_mouse = False

        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x = xpos
        self.last_y = ypos

        if self.is_leftmouse_pressed:
            self.yaw += xoffset * self.rot_sensitivity
            self.pitch += yoffset * self.rot_sensitivity

            self.pitch = np.clip(self.pitch, -np.pi / 2, np.pi / 2)

            front = np.array([np.cos(self.yaw) * np.cos(self.pitch), 
                            np.sin(self.pitch), np.sin(self.yaw) * 
                            np.cos(self.pitch)])
            front = self._global_rot_mat() @ front.reshape(3, 1)
            front = front[:, 0]
            self.position[:] = - front * np.linalg.norm(self.position - self.target) + self.target
            
            self.is_pose_dirty = True
        
        if self.is_rightmouse_pressed:
            front = self.target - self.position
            front = front / np.linalg.norm(front)
            right = np.cross(self.up, front)
            self.position += right * xoffset * self.trans_sensitivity
            self.target += right * xoffset * self.trans_sensitivity
            cam_up = np.cross(right, front)
            self.position += cam_up * yoffset * self.trans_sensitivity
            self.target += cam_up * yoffset * self.trans_sensitivity
            
            self.is_pose_dirty = True
        
    def process_wheel(self, dx, dy):
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        self.position += front * dy * self.zoom_sensitivity
        self.target += front * dy * self.zoom_sensitivity
        self.is_pose_dirty = True
        
    def process_roll_key(self, d):
        front = self.target - self.position
        right = np.cross(front, self.up)
        new_up = self.up + right * (d * self.roll_sensitivity / np.linalg.norm(right))
        self.up = new_up / np.linalg.norm(new_up)
        self.is_pose_dirty = True

    def flip_ground(self):
        self.up = -self.up
        self.is_pose_dirty = True

    def update_target_distance(self):
        _dir = self.target - self.position
        _dir = _dir / np.linalg.norm(_dir)
        self.target = self.position + _dir * self.target_dist
        
    def update_resolution(self, height, width):
        self.h = max(height, 1)
        self.w = max(width, 1)
        self.is_intrin_dirty = True
        
    def print_camera_pose(self):
        print(f"Position: [{self.position[0]:.6f}, {self.position[1]:.6f}, {self.position[2]:.6f}]")
        print(f"Target:   [{self.target[0]:.6f}, {self.target[1]:.6f}, {self.target[2]:.6f}]")
        print(f"Up:       [{self.up[0]:.6f}, {self.up[1]:.6f}, {self.up[2]:.6f}]")
        
        print(f"Yaw:      {self.yaw:.6f}")
        print(f"Pitch:    {self.pitch:.6f}")
        
        front = self.target - self.position
        front = front / np.linalg.norm(front)
        print(f"Direction: [{front[0]:.6f}, {front[1]:.6f}, {front[2]:.6f}]")
        
        print("-" * 30)
        print(f"camera.set_camera_pose(")
        print(f"    position=[{self.position[0]:.6f}, {self.position[1]:.6f}, {self.position[2]:.6f}],")
        print(f"    target=[{self.target[0]:.6f}, {self.target[1]:.6f}, {self.target[2]:.6f}],")
        print(f"    up=[{self.up[0]:.6f}, {self.up[1]:.6f}, {self.up[2]:.6f}],")
        print(f"    yaw={self.yaw:.6f}, pitch={self.pitch:.6f}")
        print(f")")
        print("="*50 + "\n")
    
    def set_camera_pose(self, position=None, target=None, up=None, yaw=None, pitch=None):
        if position is not None:
            self.position = np.array(position, dtype=np.float32)
        if target is not None:
            self.target = np.array(target, dtype=np.float32)
        if up is not None:
            self.up = np.array(up, dtype=np.float32)
        if yaw is not None:
            self.yaw = yaw
        if pitch is not None:
            self.pitch = pitch
            
        self.is_pose_dirty = True
SimpleCamera = Camera