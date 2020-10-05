from controller import Robot, Keyboard, Display, Motion, Motor, Camera
import numpy as np
import cv2


class NaoRobot(Robot):
    def __init__(self):
        super(NaoRobot, self).__init__()
        print('> Starting robot controller')

        self.timeStepInMilli = 32
        self.state = 0

        # Sensors init
        self.gps = self.getGPS('gps')
        self.gps.enable(self.timeStepInMilli)

        self.step(self.timeStepInMilli)  # Execute one step to get the initial position

        self.topCam: Camera = self.getCamera("CameraTop")
        self.topCam.enable(self.timeStepInMilli)
        self.topDisplay: Display = self.getDisplay("CameraTop")

        # Actuators init
        self.headYaw: Motor = self.getMotor("HeadYaw")
        self.headYaw.getPositionSensor().enable(self.timeStepInMilli)
        self.leftShoulderPitch: Motor = self.getMotor("LShoulderPitch")
        self.leftShoulderPitch.getPositionSensor().enable(self.timeStepInMilli)
        self.leftShoulderRoll: Motor = self.getMotor("LShoulderRoll")
        self.leftShoulderRoll.getPositionSensor().enable(self.timeStepInMilli)

    @staticmethod
    def image_to_display(img, display):
        height, width, channels = img.shape
        imageRef = display.imageNew(cv2.transpose(img).tolist(), Display.RGB, width, height)
        display.imagePaste(imageRef, 0, 0)

    def collect_data(self):
        """
        This method takes in keyboard input and either turns the head from left to right or walks forwards/backwards.
        """
        self.enable_keyboard()
        step_size_pitch = (self.leftShoulderPitch.getMaxPosition() + abs(self.leftShoulderPitch.getMinPosition())) / 50
        step_size_roll = (self.leftShoulderRoll.getMaxPosition() + abs(self.leftShoulderRoll.getMinPosition())) / 50
        step_size_yaw = (self.headYaw.getMaxPosition() + abs(self.headYaw.getMinPosition())) / 50
        self.initiate_motors(0.0, 0.0, 1.0, 1.0)
        while self.step(self.timeStepInMilli) != -1:
            k = self.keyboard.getKey()
            pitchPosition = self.leftShoulderPitch.getPositionSensor().getValue()
            rollPosition = self.leftShoulderPitch.getPositionSensor().getValue()
            headYawPosition = self.headYaw.getPositionSensor().getValue()
            if k == Keyboard.DOWN:
                self.leftShoulderPitch.setPosition(pitchPosition + step_size_pitch)
            elif k == Keyboard.UP:
                self.leftShoulderPitch.setPosition(pitchPosition - step_size_pitch)
            if k == Keyboard.RIGHT:
                self.leftShoulderRoll.setPosition(rollPosition + step_size_roll)
            elif k == Keyboard.LEFT:
                self.leftShoulderRoll.setPosition(rollPosition - step_size_roll)
            elif k == ord('A'):
                self.headYaw.setPosition(headYawPosition - step_size_yaw)
            elif k == ord('D'):
                self.headYaw.setPosition(headYawPosition + step_size_yaw)
            elif k == ord('E'):
                self.safe_data()
            elif k == ord('S'):
                self.leftShoulderRoll.setVelocity(0)
                self.leftShoulderRoll.setPosition(float('inf'))
                self.leftShoulderPitch.setVelocity(0)
                self.leftShoulderPitch.setPosition(float('inf'))

    def initiate_motors(self, pos_roll: float, pos_pitch: float, vel_roll: float, vel_pitch: float):
        """
        The head motors need to be set to an initial value in order to work.
        """
        self.leftShoulderPitch.setPosition(pos_pitch)
        self.leftShoulderPitch.setVelocity(vel_pitch)
        self.leftShoulderRoll.setPosition(pos_roll)
        self.leftShoulderRoll.setVelocity(vel_roll)

    def enable_keyboard(self):
        self.keyboard.enable(self.timeStepInMilli)
        self.keyboard = self.getKeyboard()

    @staticmethod
    def find_ball_in_img(img):
        """
        Using the HSV values described in this article https://bit.ly/3kvcg9v in order to
        find the green sphere in the image.
        """
        greenLower = (29, 86, 6)
        greenUpper = (64, 255, 255)
        blurred = cv2.GaussianBlur(img, (11, 11), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, greenLower, greenUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        im2, contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours):
            m = cv2.moments(np.array(contours[0]))
            cy, cx = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
            return cx, cy
        return None, None

    def safe_data(self):
        img = np.array(self.topCam.getImageArray(), dtype=np.uint8)
        ball_x, ball_y = self.find_ball_in_img(img)
        positionLeftShoulderRoll = self.leftShoulderRoll.getPositionSensor().getValue()
        positionLeftShoulderPitch = self.leftShoulderPitch.getPositionSensor().getValue()
        with open('data_points.csv', 'a') as f:
            f.write(",".join(map(str, [ball_x, ball_y, positionLeftShoulderRoll, positionLeftShoulderPitch])))
            f.write('\n')


robot = NaoRobot()
robot.collect_data()
