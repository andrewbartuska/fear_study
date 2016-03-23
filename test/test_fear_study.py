from fear_study import *
from nose.tools import *

def test_import_data():
	

def test_load_tobii_data():
	df = load_tobii_data("test/test_data/tobii_test.csv")
	assert_equal(type(df), pd.DataFrame)

def test_load_sync_pulses_data():
	tobii_sync = load_sync_pulses_data("test/test_data/test_sync_pulse_real.json")
	assert_equal(type(tobii_sync), dict)
	keys = tobii_sync.keys()
	assert_equal(all(isinstance(x, float) for x in keys), True)

def test_load_task_data():
	task_data = load_task_data("test/test_data/test_faces_run.json")
	assert_equal(type(task_data), dict)