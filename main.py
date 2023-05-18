import sentry_sdk
import json
import logging
from kafka import KafkaConsumer
from kafka import KafkaProducer
from model.tip import Model
from model.tip import TripValidationError

BOOTSTRAP_SERVER = 'kafka-service.kafka.svc.cluster.local:9092'
TRIPS = 'trips'
TIPS = 'tips'

logger = logging.getLogger('mariotte')

sentry_sdk.init(
  dsn = "https://89376a19f9d244c3b3e64f0bd599821c@sentry.cauchy.link/5",
  traces_sample_rate = 1.0
)

consumer = KafkaConsumer(
  bootstrap_servers = [BOOTSTRAP_SERVER],
  value_deserializer = lambda x: json.loads(x)
)

producer = KafkaProducer(
  bootstrap_servers = [BOOTSTRAP_SERVER],
  value_serializer = lambda x: json.dumps(x).encode('utf-8')
)

model = Model()

if __name__ == "__main__":
  consumer.subscribe(TRIPS)
  for message in consumer:
    trip = message.value
    try:
      tip = model.predict(trip)
      trip['predicted_tip'] = tip
      producer.send(TIPS, trip)
    except TripValidationError as e:
      logger.warning("Skipping invalid record")
      continue
    except Exception as e:
      raise e
