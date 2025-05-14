from adafruit_servokit import ServoKit
import math
import time
from picamera2 import Picamera2
import cv2
import numpy as np
import threading


class Orient:
    def __init__(self):
        # Initialize servo kit (16 channels)
        self.kit = ServoKit(channels=16)

        # Set pulse width range for MG995
        self.kit.servo[0].set_pulse_width_range(500, 2500)
        self.kit.servo[4].set_pulse_width_range(500, 2500)
        self.kit.servo[8].set_pulse_width_range(500, 2500)

        # Map servo indexes
        self.s1_index = 0
        self.s2_index = 4
        self.s3_index = 8

        # Link length L = [bottom, lower link, upper link, ceiling]
        self.L = [0.046, 0.04, 0.065, 0.065]

        # Initial position (theta, phi, pz)
        self.ini_pos = [0, 0, 0.0732]
        self.pz_max = 0.0732
        self.pz_min = 0.0532
        self.phi_max = 20

    def set_up(self):
        pass  # No setup needed for MG995 using ServoKit

    def clean_up(self):
        pass  # No specific cleanup for MG995, unless you want to reset angles

    def kinema_inv(self, n, Pz):
        L = self.L
        A = (L[0]+L[1])/Pz
        B = (Pz**2+L[2]**2-(L[0]+L[1])**2-L[3]**2)/(2*Pz)
        C = A**2+1
        D = 2*(A*B-(L[0]+L[1]))
        E = B**2+(L[0]+L[1])**2-L[2]**2
        Pmx = (-D+math.sqrt(D**2-4*C*E))/(2*C)
        Pmz = math.sqrt(L[2]**2-Pmx**2+2*(L[0]+L[1])*Pmx-(L[0]+L[1])**2)

        a_m_x = (L[3]/(math.sqrt(n[0]**2 + n[2]**2)))*(n[2])
        a_m_y = 0
        a_m_z = Pz + (L[3]/(math.sqrt(n[0]**2 + n[2]**2)))*(-n[0])
        A_m = [a_m_x, a_m_y, a_m_z]
        A = (L[0]-A_m[0])/A_m[2]
        B = (A_m[0]**2+A_m[1]**2+A_m[2]**2-L[2]**2-L[0]**2+L[1]**2)/(2*A_m[2])
        C = A**2+1
        D = 2*(A*B-L[0])
        E = B**2+L[0]**2-L[1]**2
        ax = (-D+math.sqrt(D**2-4*C*E))/(2*C)
        ay = 0
        az = math.sqrt(L[1]**2-ax**2+2*L[0]*ax-L[0]**2)
        if (a_m_z < Pmz):
            az = -az
        A_2 = [ax, ay, az]
        theta_a = 90 - math.degrees(math.atan2(A_2[0]-L[0], A_2[2]))

        b_m_x = (L[3]/(math.sqrt(n[0]**2+3*n[1]**2+4*n[2]
                 ** 2+2*math.sqrt(3)*n[0]*n[1])))*(-n[2])
        b_m_y = (L[3]/(math.sqrt(n[0]**2+3*n[1]**2+4*n[2]**2 +
                 2*math.sqrt(3)*n[0]*n[1])))*(-math.sqrt(3)*n[2])
        b_m_z = Pz + (L[3]/(math.sqrt(n[0]**2+3*n[1]**2+4*n[2] **
                      2+2*math.sqrt(3)*n[0]*n[1])))*(math.sqrt(3)*n[1]+n[0])
        B_m = [b_m_x, b_m_y, b_m_z]

        A = -(B_m[0]+math.sqrt(3)*B_m[1]+2*L[0])/B_m[2]
        B = (B_m[0]**2+B_m[1]**2+B_m[2]**2+L[1]**2-L[2]**2-L[0]**2)/(2*B_m[2])
        C = A**2+4
        D = 2*A*B+4*L[0]
        E = B**2+L[0]**2-L[1]**2
        x = (-D-math.sqrt(D**2-4*C*E))/(2*C)
        y = math.sqrt(3)*x
        z = math.sqrt(L[1]**2-4*x**2-4*L[0]*x-L[0]**2)
        if (b_m_z < Pmz):
            z = -z
        B_2 = [x, y, z]
        theta_b = 90 - \
            math.degrees(math.atan2(
                math.sqrt(B_2[0]**2+B_2[1]**2)-L[0], B_2[2]))

        c_m_x = (L[3]/(math.sqrt(n[0]**2+3*n[1]**2+4*n[2]
                 ** 2-2*math.sqrt(3)*n[0]*n[1])))*(-n[2])
        c_m_y = (L[3]/(math.sqrt(n[0]**2+3*n[1]**2+4*n[2]**2 -
                 2*math.sqrt(3)*n[0]*n[1])))*(math.sqrt(3)*n[2])
        c_m_z = Pz + (L[3]/(math.sqrt(n[0]**2+3*n[1]**2+4*n[2] **
                      2-2*math.sqrt(3)*n[0]*n[1])))*(-math.sqrt(3)*n[1]+n[0])
        C_m = [c_m_x, c_m_y, c_m_z]

        A = -(C_m[0]-math.sqrt(3)*C_m[1]+2*L[0])/C_m[2]
        B = (C_m[0]**2+C_m[1]**2+C_m[2]**2+L[1]**2-L[2]**2-L[0]**2)/(2*C_m[2])
        C = A**2+4
        D = 2*A*B+4*L[0]
        E = B**2+L[0]**2-L[1]**2
        x = (-D-math.sqrt(D**2-4*C*E))/(2*C)
        y = -math.sqrt(3)*x
        z = math.sqrt(L[1]**2-4*x**2-4*L[0]*x-L[0]**2)
        if (c_m_z < Pmz):
            z = -z
        C_2 = [x, y, z]
        theta_c = 90 - \
            math.degrees(math.atan2(
                math.sqrt(C_2[0]**2+C_2[1]**2)-L[0], C_2[2]))
        thetas = [theta_a, theta_b-7, theta_c]
        return thetas

    def control_t_posture(self, pos, t):
        theta = pos[0]
        phi = pos[1]
        if phi > self.phi_max:
            phi = self.phi_max
        Pz = pos[2]
        if Pz > self.pz_max:
            Pz = self.pz_max
        elif Pz < self.pz_min:
            Pz = self.pz_min

        z = math.cos(math.radians(phi))
        r = math.sin(math.radians(phi))
        x = r * math.cos(math.radians(theta))
        y = r * math.sin(math.radians(theta))
        n = [x, y, z]
        angles = self.kinema_inv(n, Pz)

        print(
            f"Servo angles: S1 = {angles[0]:.2f}°, S2 = {angles[1]:.2f}°, S3 = {angles[2]:.2f}°")

        # Move the servos
        self.kit.servo[self.s1_index].angle = max(0, min(180, angles[0]))
        self.kit.servo[self.s2_index].angle = max(0, min(180, angles[1]))
        self.kit.servo[self.s3_index].angle = max(0, min(180, angles[2]))
        time.sleep(t)

    def Initialize_posture(self):
        self.control_t_posture(self.ini_pos, 1)


class PID:
    def __init__(self, K_PID, k, alpha):
        self.kp = K_PID[0]
        self.ki = K_PID[1]
        self.kd = K_PID[2]
        self.k = k
        self.alpha = alpha  # Coefficient of the low-pass filter
        self.last_output_x = 0
        self.last_output_y = 0
        self.last_error_x = 0
        self.integral_x = 0
        self.last_error_y = 0
        self.integral_y = 0
        self.last_time = None
        self.count = 0
        self.F = 0

    def compute(self, Goal, Current_value):
        current_time = time.perf_counter()
        if self.last_time is None:
            self.last_time = current_time
            return 0, 0
        # Calculate error
        error_x = Goal[0] - Current_value[0]
        error_y = Goal[1] - Current_value[1]
        # Calculate integral
        self.integral_x += error_x * (current_time - self.last_time)
        self.integral_y += error_y * (current_time - self.last_time)
        # Calculate derivative
        derivative_x = (error_x - self.last_error_x) / \
                        (current_time - self.last_time)
        derivative_y = (error_y - self.last_error_y) / \
                        (current_time - self.last_time)
        # Calculate PID output
        output_x = self.kp * error_x + self.ki * \
            self.integral_x + self.kd * derivative_x
        output_y = self.kp * error_y + self.ki * \
            self.integral_y + self.kd * derivative_y
        # Apply low-pass filter
        output_x = self.alpha * output_x + \
            (1 - self.alpha) * self.last_output_x
        output_y = self.alpha * output_y + \
            (1 - self.alpha) * self.last_output_y
        # Calculate theta and phi
        theta = math.degrees(math.atan2(output_y, output_x))
        if theta < 0:
            theta += 360
        phi = self.k * math.sqrt(output_x**2 + output_y**2)

        self.last_error_x = error_x
        self.last_error_y = error_y
        self.last_output_x = output_x
        self.last_output_y = output_y
        self.last_time = current_time

        return theta, phi


class Camera:
    def __init__(self):
        self.picam2 = Picamera2()
        self.height = 480
        self.width = 480
        self.config = self.picam2.create_video_configuration(
            main={"format": 'RGB888', "size": (self.height, self.width)},

        )
        self.picam2.configure(self.config)

        # yellow
        # self.lower_pink = np.array([0, 173, 159])  
        # self.upper_pink = np.array([46, 255, 255])  

        # Pink
        self.lower_pink = np.array([165, 150, 50]) 
        self.upper_pink = np.array([180, 255, 255])

        # Kancha
        # self.lower_pink = np.array([50, 0, 0])  
        # self.upper_pink = np.array([90, 255, 110])
        
        
        # Orange
        #self.lower_pink = np.array([0, 155, 155])  
        #self.upper_pink = np.array([30, 255, 255])

        self.picam2.start()

    def take_pic(self):
        image = self.picam2.capture_array()
        return image

    def show_video(self, image):
        img_align = image.copy()  # Create a copy to draw axes
        cv2.line(img_align, (0, 240), (480, 240), (0, 0, 0), thickness=2)
        cv2.line(img_align, (240, 0), (240, 480), (0, 0, 0), thickness=2)
        cv2.circle(img_align, (240, 240), radius=5,
                   color=(0, 255, 255), thickness=-1)

        cv2.imshow("Align", img_align)
        # cv2.imshow("Live", image)
        cv2.waitKey(1)

    def find_ball(self, image):
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(image_hsv, self.lower_pink, self.upper_pink)
   
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            area = cv2.contourArea(largest_contour) 

            if area > 200:  
                cv2.circle(image, (int(x), int(y)),
                           int(radius), (0, 255, 0), 2)
                self.show_video(image)
                d = radius*2
                h = 10000/d
                x -= self.height / 2
                y -= self.width / 2
                x, y = -y, -x
                return int(x), int(y), int(area)  
        self.show_video(image)
        return -1, -1, 0  

    def clean_up_cam(self):
        self.picam2.stop()
        self.picam2.close()
        cv2.destroyAllWindows()


height = 480
width = 480
channels = 3

# Create an empty image (all pixels are 0)
image = np.zeros((height, width, channels), dtype=np.uint8)
# Trial1 (0.015, 0.000152, 0.0055) k=1 a=0.9
# Trial2 (0.0155, 0.000152, 0.0055) k=1 a=0.9
# Trial3 (0.0153, 0.000152, 0.0055) k=1 a=0.9
# Trial4 (0.0153, 0.00016, 0.0055) k=1 a=0.9
K_PID = [0.0153, 0.00016, 0.0055]  # 0.015, 0.0001, 0.0051
k = 1
a = 0.9

Robot = Orient()
camera = Camera()
pid = PID(K_PID, k, a)

Robot.set_up()
Robot.Initialize_posture()
pz_ini = Robot.ini_pos[2]

frame_count = 0
start_time = time.time()
img_start_time = time.time()
rob_start_time = time.time()
fps = 0  # Set the initial value to 0
img_fps = 0
rob_fps = 0

# ball coordinates
x = -1
y = -1
area = -1

goal = [0, 0]


def main():
    def get_img():
        global image, img_fps, img_start_time
        img_frame_count = 0
        while (1):
            image = camera.take_pic()
            # img_fps calculation
            img_frame_count += 1
            if img_frame_count == 100:
                img_end_time = time.time()
                img_elapsed_time = img_end_time - img_start_time
                img_fps = 100 / img_elapsed_time
                img_start_time = img_end_time
                img_frame_count = 0

    def cont_rob():
        global x, y, area, rob_fps, rob_start_time
        rob_frame_count = 0
        while (1):
            x, y, area = camera.find_ball(image)
            # rob_fps calculation 
            rob_frame_count += 1
            if rob_frame_count == 100:
                rob_end_time = time.time()
                rob_elapsed_time = rob_end_time - rob_start_time
                rob_fps = 100 / rob_elapsed_time
                rob_start_time = rob_end_time
                rob_frame_count = 0

    try:
        camera_thread = threading.Thread(target=get_img)
        rob_thread = threading.Thread(target=cont_rob)
        camera_thread.start()
        rob_thread.start()

        while(1):
            Current_value = [x, y, area]
            if x != -1:
                theta, phi = pid.compute(goal, Current_value)
                pos = [theta, phi, pz_ini]
                Robot.control_t_posture(pos, 0.01)
                
                print(Current_value)
                
            print(f"img_fps: {img_fps}, rob_fps: {rob_fps}")
     

    finally:
        Robot.clean_up()
    

if __name__ == "__main__":
    main()
