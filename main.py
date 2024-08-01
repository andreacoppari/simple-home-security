import time
import cv2
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import logging


# Configure logging
logging.basicConfig(filename='home_security.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# Load pre-trained MobileNet SSD model and the class labels
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def send_email(image_path):
    port = 587
    smtp_server = "live.smtp.mailtrap.io"
    login = "api"
    password = "9ed04dc0442fa97f4929b59572b6d9e4"
    fromaddr = "mailtrap@demomailtrap.com"
    toaddr = "andrea.coppari@proton.me"
    msg = MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = toaddr
    msg['Subject'] = "Alert: Multiple People Detected"

    body = "Multiple people have been detected in your house. See the attached image."
    msg.attach(MIMEText(body, 'plain'))

    attachment = open(image_path, "rb")
    part = MIMEBase('application', 'octet-stream')
    part.set_payload((attachment).read())
    encoders.encode_base64(part)
    part.add_header('Content-Disposition', "attachment; filename= %s" % image_path)
    msg.attach(part)

    server = smtplib.SMTP(smtp_server, port)
    server.starttls()
    server.login(login, password)
    text = msg.as_string()
    server.sendmail(fromaddr, toaddr, text)
    server.quit()

# Set up the webcam
cap = cv2.VideoCapture(1)

# Initialize the timer for the email alert
last_email_time = 0
email_timeout = 10 * 60  # 10 minutes

if not cap.isOpened():
    logging.error("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Error: Could not read frame.")
        break

    # Prepare the frame for the model
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    count = 0

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            if CLASSES[idx] == "person":
                count += 1
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # Display the number of people detected
    cv2.putText(frame, f'People count: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)

    # Log the detected count
    # logging.info(f'People count: {count}')

    current_time = time.time()
    # If more than one person is detected, save a snapshot, send an email, and break
    if count >= 1 and ((current_time - last_email_time > email_timeout) or count >= 2):
        snapshot_path = 'alert_snapshot.jpg'
        cv2.imwrite(snapshot_path, frame)
        logging.warning("Multiple people detected. Saving snapshot and sending email.")
        send_email(snapshot_path)
        last_email_time = current_time

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logging.info("Exiting the application.")
        break

cap.release()
cv2.destroyAllWindows()
