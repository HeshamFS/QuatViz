import numpy as np

class Quaternion:
    def __init__(self, w=1, x=0, y=0, z=0):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, q):
        return Quaternion(self.w + q.w, self.x + q.x, self.y + q.y, self.z + q.z)

    def __sub__(self, q):
        return Quaternion(self.w - q.w, self.x - q.x, self.y - q.y, self.z - q.z)

    def __mul__(self, q):
        return Quaternion(
            self.w * q.w - self.x * q.x - self.y * q.y - self.z * q.z,
            self.w * q.x + self.x * q.w + self.y * q.z - self.z * q.y,
            self.w * q.y - self.x * q.z + self.y * q.w + self.z * q.x,
            self.w * q.z + self.x * q.y - self.y * q.x + self.z * q.w
        )

    def conjugate(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def norm(self):
        return (self.w**2 + self.x**2 + self.y**2 + self.z**2)**0.5

    def normalize(self):
        n = self.norm()
        if n == 0:
            raise ValueError("Cannot normalize a zero quaternion.")
        self.w /= n
        self.x /= n
        self.y /= n
        self.z /= n

    def inverse(self):
        conj = self.conjugate()
        norm_sq = self.norm()**2
        if norm_sq == 0:
            raise ValueError("Cannot compute inverse of a zero quaternion.")
        return Quaternion(conj.w/norm_sq, conj.x/norm_sq, conj.y/norm_sq, conj.z/norm_sq)

    def to_rotation_matrix(self):
        """Convert the quaternion to a 3x3 rotation matrix."""
        R = np.array([
            [1 - 2*self.y**2 - 2*self.z**2, 2*self.x*self.y - 2*self.z*self.w, 2*self.x*self.z + 2*self.y*self.w],
            [2*self.x*self.y + 2*self.z*self.w, 1 - 2*self.x**2 - 2*self.z**2, 2*self.y*self.z - 2*self.x*self.w],
            [2*self.x*self.z - 2*self.y*self.w, 2*self.y*self.z + 2*self.x*self.w, 1 - 2*self.x**2 - 2*self.y**2]
        ])
        return R

    def from_euler_angles(pitch, yaw, roll):
        """Initialize a Quaternion from Euler angles."""
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)

        w = cy * cr * cp + sy * sr * sp
        x = cy * sr * cp - sy * cr * sp
        y = cy * cr * sp + sy * sr * cp
        z = sy * cr * cp - cy * sr * sp

        return Quaternion(w, x, y, z)

    def rotate_vector(self, v):
        """Rotate a 3D vector using this quaternion."""
        # Treating the vector as a quaternion with w=0
        vec_quaternion = Quaternion(0, v[0], v[1], v[2])
        
        # Performing the rotation: v' = q * v * q^-1
        rotated_vector = self * vec_quaternion * self.inverse()
        
        # Extract the vector part of the quaternion
        return np.array([rotated_vector.x, rotated_vector.y, rotated_vector.z])
    
    def inverse(self):
        conj = self.conjugate()
        norm_sq = self.norm()**2
        if norm_sq == 0:
            raise ValueError("Cannot compute inverse of a zero quaternion.")
        return Quaternion(conj.w/norm_sq, conj.x/norm_sq, conj.y/norm_sq, conj.z/norm_sq)
    
    def to_euler_angles(self):
        """Convert the quaternion to Euler angles."""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (self.w * self.x + self.y * self.z)
        cosr_cosp = 1 - 2 * (self.x**2 + self.y**2)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (self.w * self.y - self.z * self.x)
        if np.abs(sinp) >= 1:
            pitch = np.sign(sinp) * np.pi / 2
        else:
            pitch = np.arcsin(sinp)
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (self.w * self.z + self.x * self.y)
        cosy_cosp = 1 - 2 * (self.y**2 + self.z**2)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return pitch, yaw, roll

    def __str__(self):
        return f"({self.w}, {self.x}i, {self.y}j, {self.z}k)"
