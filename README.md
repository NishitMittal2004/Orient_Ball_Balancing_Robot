# Orient Ball Balancing Robot
![alt text](https://github.com/NishitMittal2004/Orient_Ball_Balancing_Robot/blob/main/Ball%20Balancing%20Robot%20-%20Poster.png)

---

## 📌 Overview

 **ORIENT** is a ball-balancing **3-RRS (Revolute-Revolute-Spherical)** Platform. It can dynamically balance a ball in real time and perform motion tasks such as moving in straight lines, quadrants, and even circles. The robot uses real-time computer vision, inverse kinematics, and PID control for its operation.

---

## 🔧 Tools & Concepts Used

- Python
- OpenCV (Contour Detection, HSV Tuning)
- Inverse Kinematics (for 3-RRS)
- PID Control

---

## Demo Video of robot balancing the ball
![](Balance.gif)

---


## 🧰 List of Components [(Link)](https://github.com/NishitMittal2004/Orient_Ball_Balancing_Robot/blob/main/List%20of%20Components.pdf)

| S.No | Component | Quantity | Purpose | 
|------|-----------|----------|---------|
| 1. | **Raspberry Pi 5** | 1 | Main processing unit |
| 2. | **Raspberry Pi Active Cooler** | 1 | Heat management for Pi | 
| 3. | **5MP Camera Module (for Pi 5)** | 1 | Vision (ball detection) | 
| 4. | **Camera Cable** | 1 | Connects camera to Pi | 
| 5. | **MG995 Servo Motors** | 3 | Actuation of platform | 
| 6. | **PCA9685 Servo Driver** | 1 | Servo PWM control |
| 7. | **5V/2A DC Adapter** | 1 | Powers servo driver | 
| 8. | **Female DC Jack** | 1 | Connects adapter to servo driver|
| 9. | **Ball Joint (M6)** | 3 | Platform joints (RRS design) | 
| 10. | **Ball Bearings (ID - 6mm, OD - 16mm)** | 3 | For joint rotation |

---

## 🧮 Inverse Kinematics [(Notes)](https://github.com/NishitMittal2004/Orient_Ball_Balancing_Robot/blob/main/Inverse%20Kinematics%20Maths.pdf)

We derived and implemented the IK for the 3-RRS platform to calculate the exact servo angles required to achieve desired platform tilt and height. This allowed precise control over the ball's motion.

---

## 👁️ Ball Detection with OpenCV

- HSV Trackbar for color tuning
- `cv2.findContours` and `cv2.minEnclosingCircle` to detect the ball
- Real-time position and area returned for control logic

---

## 🎛️ PID Tuning

After exhaustive tuning and testing, a finely tuned PID controller was implemented, keeping the ball at centre.

---

## 👨‍💻 Authors

- **Nishit Mittal**  
- **Sahil Sharma**


