## coding:utf-8 ##
import RPi.GPIO as GPIO
import time
import cv2
import numpy as np
import skfuzzy as fuzz


######Initialization#####

#sign function
def sign(x):
    if x > 0:
        return 1.0
    else:
        return -1.0

# define pins of the board
EA, I2, I1, EB, I4, I3 = (13, 19, 26, 16, 20, 21)
FREQUENCY = 50

# set GPIO configuration
GPIO.setmode(GPIO.BCM)

# set GPIO output
GPIO.setup([EA, I2, I1, EB, I4, I3], GPIO.OUT)
GPIO.output([EA, I2, EB, I3], GPIO.LOW)
GPIO.output([I1, I4], GPIO.HIGH)

pwma = GPIO.PWM(EA, FREQUENCY)
pwmb = GPIO.PWM(EB, FREQUENCY)

# pwm Initialization
pwma.start(0)
pwmb.start(0)

# define position center
center_now = 320

# activate the camera，image size640*480（length*height），in OpenCV it is stored as 480*640（line*column）
cap = cv2.VideoCapture(0)

# PID deifne and initialize the three errors and adjust
error = [0, 0, 0]
delta_error = [0, 0, 0]
adjust = [0, 0, 0]

# PID parameters and target settings, standard pwm value and allowed derivation interval
kp = 1.3
ki = 0.3
kd = 0.1
target = 320
lspeed = 60
rspeed = 55
control = 35
ret, frame = cap.read()

#initialization complete
print("Initialization Complete！Press ENTER to Start！")
input()

try:
    while True:
        ret, frame = cap.read()
        # transfer the image into HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        upper_orange = np.array([11, 43, 46])
        lower_orange = np.array([25, 255, 255])
        # binarization
        ret, dst = cv2.inRange(hsv, lower_orange, upper_orange)
        # sharpen the image
        dst = cv2.dilate(dst, None, iterations=2)
        cv2.imshow("Camera view", dst)

        # analyze line 320 as the benchmark
        color = dst[320]
        # count black pixels
        black_count = np.sum(color == 0)

        # avoid empty black_count error
        if black_count == 0:
            continue
        else:
            black_index = np.where(color == 0)
        # find the black pixel center point
        center_now = (black_index[0][black_count - 1] + black_index[0][0]) / 2

        # calculate the offset value
        direction = center_now - 320

        print("offset：", direction)
        if abs(direction) > 240:
            pwma.ChangeDutyCycle(0)
            pwmb.ChangeDutyCycle(0)

        ########################################
        ##Start of the fuzzy inference system ##
        ########################################

        # define inputs and outputs of the fuzzy controller
        error = fuzz.Antecedent(np.arange(-10, 10, 1), 'error')
        delta_error = fuzz.Antecedent(np.arange(-5, 5, 1), 'delta_error')
        output = fuzz.Consequent(np.arange(-10, 10, 1), 'output')

        # define input and output membership functions
        error['NB'] = fuzz.trimf(error.universe, [-10, -10, -5])
        error['NM'] = fuzz.trimf(error.universe, [-10, -5, 0])
        error['NS'] = fuzz.trimf(error.universe, [-5, 0, 5])
        error['ZO'] = fuzz.trimf(error.universe, [0, 0, 0])
        error['PS'] = fuzz.trimf(error.universe, [5, 0, -5])
        error['PM'] = fuzz.trimf(error.universe, [10, 5, 0])
        error['PB'] = fuzz.trimf(error.universe, [10, 10, 5])
        delta_error['NB'] = fuzz.trimf(delta_error.universe, [-5, -5, -2])
        delta_error['NM'] = fuzz.trimf(delta_error.universe, [-5, -2, 0])
        delta_error['NS'] = fuzz.trimf(delta_error.universe, [-2, 0, 2])
        delta_error['ZO'] = fuzz.trimf(delta_error.universe, [0, 0, 0])
        delta_error['PS'] = fuzz.trimf(delta_error.universe, [2, 0, -2])
        delta_error['PM'] = fuzz.trimf(delta_error.universe, [5, 2, 0])
        delta_error['PB'] = fuzz.trimf(delta_error.universe, [5, 5, 2])
        output['NB'] = fuzz.trimf(output.universe, [-10, -10, -5])
        output['NM'] = fuzz.trimf(output.universe, [-10, -5, 0])
        output['NS'] = fuzz.trimf(output.universe, [-5, 0, 5])
        output['ZO'] = fuzz.trimf(output.universe, [0, 0, 0])
        output['PS'] = fuzz.trimf(output.universe, [5, 0, -5])
        output['PM'] = fuzz.trimf(output.universe, [10, 5, 0])
        output['PB'] = fuzz.trimf(output.universe, [10, 10, 5])

        # define fuzzy rule base
        rule1 = fuzz.Rule(error['NB'] & delta_error['NB'], output['PB'])
        rule2 = fuzz.Rule(error['NB'] & delta_error['NM'], output['PB'])
        rule3 = fuzz.Rule(error['NB'] & delta_error['NS'], output['PB'])
        rule4 = fuzz.Rule(error['NB'] & delta_error['ZO'], output['PB'])
        rule5 = fuzz.Rule(error['NB'] & delta_error['PS'], output['PM'])
        rule6 = fuzz.Rule(error['NB'] & delta_error['PM'], output['ZO'])
        rule7 = fuzz.Rule(error['NB'] & delta_error['PB'], output['ZO'])
        rule8 = fuzz.Rule(error['NM'] & delta_error['NB'], output['PB'])
        rule9 = fuzz.Rule(error['NM'] & delta_error['NM'], output['PB'])
        rule10 = fuzz.Rule(error['NM'] & delta_error['NS'], output['PB'])
        rule11 = fuzz.Rule(error['NM'] & delta_error['ZO'], output['PM'])
        rule12 = fuzz.Rule(error['NM'] & delta_error['PS'], output['PM'])
        rule13 = fuzz.Rule(error['NM'] & delta_error['PM'], output['ZO'])
        rule14 = fuzz.Rule(error['NM'] & delta_error['PB'], output['ZO'])
        rule15 = fuzz.Rule(error['NS'] & delta_error['NB'], output['PB'])
        rule16 = fuzz.Rule(error['NS'] & delta_error['NM'], output['PM'])
        rule17 = fuzz.Rule(error['NS'] & delta_error['NS'], output['PM'])
        rule18 = fuzz.Rule(error['NS'] & delta_error['ZO'], output['PS'])
        rule19 = fuzz.Rule(error['NS'] & delta_error['PS'], output['ZO'])
        rule20 = fuzz.Rule(error['NS'] & delta_error['PM'], output['NS'])
        rule21 = fuzz.Rule(error['NS'] & delta_error['PB'], output['NM'])
        rule22 = fuzz.Rule(error['ZO'] & delta_error['NB'], output['PM'])
        rule23 = fuzz.Rule(error['ZO'] & delta_error['NM'], output['PM'])
        rule24 = fuzz.Rule(error['ZO'] & delta_error['NS'], output['PS'])
        rule25 = fuzz.Rule(error['ZO'] & delta_error['ZO'], output['ZO'])
        rule26 = fuzz.Rule(error['ZO'] & delta_error['PS'], output['NS'])
        rule27 = fuzz.Rule(error['ZO'] & delta_error['PM'], output['NM'])
        rule28 = fuzz.Rule(error['ZO'] & delta_error['PB'], output['NM'])
        rule29 = fuzz.Rule(error['PS'] & delta_error['NB'], output['PS'])
        rule30 = fuzz.Rule(error['PS'] & delta_error['NM'], output['PS'])
        rule31 = fuzz.Rule(error['PS'] & delta_error['NS'], output['ZO'])
        rule32 = fuzz.Rule(error['PS'] & delta_error['ZO'], output['NM'])
        rule33 = fuzz.Rule(error['PS'] & delta_error['PS'], output['NM'])
        rule34 = fuzz.Rule(error['PS'] & delta_error['PM'], output['NM'])
        rule35 = fuzz.Rule(error['PS'] & delta_error['PB'], output['NB'])
        rule36 = fuzz.Rule(error['PM'] & delta_error['NB'], output['ZO'])
        rule37 = fuzz.Rule(error['PM'] & delta_error['NM'], output['ZO'])
        rule38 = fuzz.Rule(error['PM'] & delta_error['NS'], output['ZO'])
        rule39 = fuzz.Rule(error['PM'] & delta_error['ZO'], output['NM'])
        rule40 = fuzz.Rule(error['PM'] & delta_error['PS'], output['NB'])
        rule41 = fuzz.Rule(error['PM'] & delta_error['PM'], output['NB'])
        rule42 = fuzz.Rule(error['PM'] & delta_error['PB'], output['NB'])
        rule43 = fuzz.Rule(error['PB'] & delta_error['NB'], output['ZO'])
        rule44 = fuzz.Rule(error['PB'] & delta_error['NM'], output['NS'])
        rule45 = fuzz.Rule(error['PB'] & delta_error['NS'], output['NB'])
        rule46 = fuzz.Rule(error['PB'] & delta_error['ZO'], output['NB'])
        rule47 = fuzz.Rule(error['PB'] & delta_error['PS'], output['NB'])
        rule48 = fuzz.Rule(error['PB'] & delta_error['PM'], output['NB'])
        rule49 = fuzz.Rule(error['PB'] & delta_error['PB'], output['NB'])
        # organize the fuzzy controller
        controller = fuzz.ControlSystem([rule1, rule2, rule3, rule4, rule5, 
                                         rule6, rule7, rule8, rule9, rule10, 
                                         rule11, rule12, rule13, rule14, rule15, 
                                         rule16, rule17, rule18, rule19, rule20, 
                                         rule21, rule22, rule23, rule24, rule25,
                                         rule26, rule27, rule28, rule29, rule30, 
                                         rule31, rule32, rule33, rule34, rule35, 
                                         rule36, rule37, rule38, rule39, rule40, 
                                         rule41, rule42, rule43, rule44, rule45, 
                                         rule46, rule47, rule48, rule49])

        # initializtion
        output_value = 0

        # retrieve center_now and target, calculate errors and error changes, denoted as delta_error
        error[0] = error[1]
        error[1] = error[2]
        error[2] = center_now - target
        delta_error[0] = delta_error[1]
        delta_error[1] = delta_error[2]
        delta_error[2] = error[2] - error[1]

        # inference according to the rule base
        output_ctrl = fuzz.ControlSystemSimulation(controller)
        output_ctrl.input['error'] = error[2]
        output_ctrl.input['delta_error'] = delta_error[2]
        output_ctrl.compute()
        output_value = output_ctrl.output['output']

        # update PID values
        adjust[0] = adjust[1]
        adjust[1] = adjust[2]
        adjust[2] = adjust[1] + kp * (output_value - adjust[1]) + ki * output_value + kd * (output_value - 2 * adjust[1] + adjust[0])

        #####################################
        ##End of the fuzzy inference system##
        #####################################

        # constrain saturated values to the value control
        if abs(adjust[2]) > control:
            adjust[2] = sign(adjust[2]) * control

        # execute PID adjustments

        # right turn
        if adjust[2] > 0:
            pwma.ChangeDutyCycle(rspeed - adjust[2]*1.3)
            pwmb.ChangeDutyCycle(lspeed)

        # left turn
        else:
            pwma.ChangeDutyCycle(rspeed)
            pwmb.ChangeDutyCycle(lspeed + adjust[2]*1.3)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            time.sleep(0.05)
except KeyboardInterrupt:
    print("END！")
    pass
# free and release

cap.release()
cv2.destroyAllWindows()
pwma.stop()
pwmb.stop()
GPIO.cleanup()
