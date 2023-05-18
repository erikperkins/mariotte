import unittest
from unittest.mock import Mock
from unittest.mock import patch
from model.tip import Model
from model.tip import TripValidationError
from numpy import array

class TestModel(unittest.TestCase):
  @patch('model.tip.mlflow', Mock())
  def setUp(self):
    self.model = Model()
    self.model.regressor = Mock()
    self.model.regressor.feature_names_in_ = {'pickup_datetime': '2022-05-17 12:11:03'}
    self.model.regressor.predict.return_value = array([3.0700364])

    self.message = {
      'pickup_datetime': '2022-05-17 12:11:03',
      'dropoff_datetime': '2022-05-17 12:30:10',
      'pickup_location_id': 237,
      'dropoff_location_id': 79,
      'passenger_count': 2.0,
      'trip_distance': 2.39,
      'payment_type': 1,
      'fare_amount': 13.5,
      'extra': 0.0,
      'mta_tax': 0.5,
      'tolls_amount': 0.0,
      'improvement_surcharge': 0.3,
      'congestion_surcharge': 2.5,
      'airport_fee': 0.0,
      'tip_amount': 3.36
    }

  def tearDown(self):
    self.model.regressor.reset_mock()

  @patch('model.tip.mlflow.sklearn.load_model')
  def test_init_loads_model(self, mock_load_model):
    model = Model()
    mock_load_model.assert_called()

  def test_predict(self):
    tip = self.model.predict(self.message)
    self.assertEqual(tip, 3.07)

    with self.assertRaises(TripValidationError):
      bad_message = self.message.copy()
      del(bad_message['pickup_datetime'])
      self.model.predict(bad_message)

  def test_validate(self):
    try:
      self.model.validate(self.message)
    except TripValidationError:
      raise self.failureException('TripValidationError raised on valid input')

    try:
      incomplete_message = self.message.copy()
      incomplete_message['passenger_count'] = None
      self.model.validate(incomplete_message)
    except TripValidationError:
      raise self.failureException('TripValidationError raised on valid input')

    with self.assertRaises(TripValidationError, msg = 'missing covariate'):
      bad_message = self.message.copy()
      del(bad_message['pickup_datetime'])
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid pickup_location_id'):
      bad_message = self.message.copy()
      bad_message['pickup_location_id'] = '1'
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid dropoff_location_id'):
      bad_message = self.message.copy()
      bad_message['dropoff_location_id'] = '1'
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid payment_type'):
      bad_message = self.message.copy()
      bad_message['payment_type'] = '1'
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid passenger_count'):
      bad_message = self.message.copy()
      bad_message['passenger_count'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid trip_distance'):
      bad_message = self.message.copy()
      bad_message['trip_distance'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid fare_amount'):
      bad_message = self.message.copy()
      bad_message['fare_amount'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid extra'):
      bad_message = self.message.copy()
      bad_message['extra'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid mta_tax'):
      bad_message = self.message.copy()
      bad_message['mta_tax'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid tolls_amount'):
      bad_message = self.message.copy()
      bad_message['tolls_amount'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid improvement_surcharge'):
      bad_message = self.message.copy()
      bad_message['improvement_surcharge'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid congestion_surcharge'):
      bad_message = self.message.copy()
      bad_message['congestion_surcharge'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid airport_fee'):
      bad_message = self.message.copy()
      bad_message['airport_fee'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid tip_amount'):
      bad_message = self.message.copy()
      bad_message['tip_amount'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid pickup_datetime'):
      bad_message = self.message.copy()
      bad_message['pickup_datetime'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid dropoff_datetime'):
      bad_message = self.message.copy()
      bad_message['dropoff_datetime'] = 0
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid pickup_datetime'):
      bad_message = self.message.copy()
      bad_message['pickup_datetime'] = 'Monday at noon'
      self.model.validate(bad_message)

    with self.assertRaises(TripValidationError, msg = 'invalid dropoff_datetime'):
      bad_message = self.message.copy()
      bad_message['dropoff_datetime'] = 'Monday at noon'
      self.model.validate(bad_message)
