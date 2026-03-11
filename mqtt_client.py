import paho.mqtt.client as mqtt
import ssl
import time

# MQTT Config
BROKER = "raspberrypi"
PORT = 8883
USERNAME = "smartgate"
PASSWORD = "smartgate123"
TOPIC_DECISION = "smartgate/decision"
TOPIC_DISTANCE = "smartgate/distance"
TOPIC_PLATE = "smartgate/plate"

CA_CERT = "/etc/mosquitto/certs/ca.crt"
CLIENT_CERT = "/etc/mosquitto/certs/client.crt"
CLIENT_KEY = "/etc/mosquitto/certs/client.key"

def create_client(client_id="smartgate-pi"):
    client = mqtt.Client(client_id=client_id)
    client.username_pw_set(USERNAME, PASSWORD)
    client.tls_set(
        ca_certs=CA_CERT,
        certfile=CLIENT_CERT,
        keyfile=CLIENT_KEY,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.tls_insecure_set(False)
    return client

def publish_decision(decision, plate, risk_score):
    """Publish gate decision over encrypted MQTT"""
    try:
        client = create_client("smartgate-publisher")
        client.connect(BROKER, PORT, 60)
        
        message = f"{decision}|{plate}|{risk_score}"
        client.publish(TOPIC_DECISION, message)
        client.publish(TOPIC_PLATE, plate)
        time.sleep(0.2)
        client.disconnect()
        print(f"📡 MQTT published: {message}")
    except Exception as e:
        print(f"MQTT error: {e}")

def publish_distance(distance):
    """Publish distance reading over encrypted MQTT"""
    try:
        client = create_client("smartgate-distance")
        client.connect(BROKER, PORT, 60)
        client.publish(TOPIC_DISTANCE, str(distance))
        time.sleep(0.1)
        client.disconnect()
    except Exception as e:
        pass  # Silent fail for distance updates
