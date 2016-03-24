from fear_study import *
from nose.tools import *

# Questions for John- Using outputs from other functions in tests

def test_import_data():
	df, tobii_sync, task_data = import_data("test/test_data/tobii_test.csv", \
						 	"test/test_data/test_sync_pulse_real.json", \
							"test/test_data/test_faces_run.json")
	data = []
	if (df.empty == False and bool(tobii_sync) == True and bool(task_data) == True):
		data.append(df)
		data.append(tobii_sync)
		data.append(task_data)
	else:
		raise TypeError("Some of the data is empty")
	assert_equal(len(data), 3)

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
	
def test_fix_table():
	df, tobii_sync, task_data = import_data("test/test_data/tobii_test.csv", \
						 	"test/test_data/test_sync_pulse_real.json", \
							"test/test_data/test_faces_run.json")	
	df = fix_tables(df, tobii_sync, task_data)
	assert_equal(df.task_time.dtypes, float)
	assert_equal(df.pupil_average.dtypes, float)
	
def test_split_target():
	df, tobii_sync, task_data = import_data("test/test_data/tobii_test.csv", \
						 	"test/test_data/test_sync_pulse_real.json", \
							"test/test_data/test_faces_run.json")	
	df = fix_tables(df, tobii_sync, task_data)
	fear_trials, notfear_trials = split_target(df, task_data)
	
	assert_equal(type(fear_trials), np.ndarray)
	assert_equal(type(fear_trials), np.ndarray)
	
	assert_equal(fear_trials.size == 0, False)
	assert_equal(notfear_trials.size == 0, False)
	
	assert_equal(all(len(x) == len(fear_trials[0]) for x in fear_trials), True)
	assert_equal(all(len(x) == len(notfear_trials[0]) for x in notfear_trials), True)
	
def test_baseline_norm_sub():
	pass

def test_baseline_norm_div():
	pass

def test_plot_comparisons():
	pass

