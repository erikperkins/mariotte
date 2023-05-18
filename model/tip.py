from pandas import DataFrame
from pandas import Timestamp
from math import floor
import mlflow
import os


MODEL_URI = "models:/GPUTipPipeline/Production"


class TripValidationError(Exception):
  def __init__(self, message):
    super().__init__(message)

class Model():
  """Predict tip from trip data."""
  def __init__(self):
    mlflow.set_tracking_uri("https://mlflow.cauchy.link")
    self.regressor = mlflow.sklearn.load_model(MODEL_URI)

  def predict(self, message):
    """Predict tip using loaded model."""
    self.validate(message)

    data = DataFrame([message]).astype({
      'pickup_datetime': 'datetime64[ns]',
      'dropoff_datetime': 'datetime64[ns]',
      'pickup_location_id': 'category',
      'dropoff_location_id': 'category',
      'payment_type': 'category'
    })

    tip, = self.regressor.predict(data)
    return floor(100 * tip) / 100.

  def validate(self, message):
    """
    Validate message structure. Ensure all expected features are present,
    and all types are correct.
    """
    try:
      keys = set(message.keys())
      features = set(self.regressor.feature_names_in_)
      assert features.issubset(keys)

      assert type(message['pickup_location_id']) in [int, type(None)]
      assert type(message['dropoff_location_id']) in [int, type(None)]
      assert type(message['payment_type']) in [int, type(None)]

      assert type(message['passenger_count']) in [float, type(None)]
      assert type(message['trip_distance']) in [float, type(None)]
      assert type(message['fare_amount']) in [float, type(None)]
      assert type(message['extra']) in [float, type(None)]
      assert type(message['mta_tax']) in [float, type(None)]
      assert type(message['tolls_amount']) in [float, type(None)]
      assert type(message['improvement_surcharge']) in [float, type(None)]
      assert type(message['congestion_surcharge']) in [float, type(None)]
      assert type(message['airport_fee']) in [float, type(None)]
      assert type(message['tip_amount']) in [float, type(None)]

      assert type(message['pickup_datetime']) in [str, type(None)]
      assert type(message['dropoff_datetime']) in [str, type(None)]

      Timestamp(message['pickup_datetime'])
      Timestamp(message['dropoff_datetime'])
    except Exception as e:
      raise TripValidationError(e)
