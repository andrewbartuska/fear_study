
# coding: utf-8

# In[2]:
# Original

import pandas as pd
import json
import matplotlib as mpl
import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
import seaborn as sns
import numpy as np


def import_data(csv_file, sync_pulses_file, faces_file):
	# Import data from separate files
	
	df = load_tobii_data(csv_file)
	tobii_sync = load_sync_pulses_data(sync_pulses_file)
	task_data = load_task_data(faces_file) #long json file
	
	return df, tobii_sync, task_data

def load_tobii_data(csv_file):
	df = pd.read_csv(csv_file) # run#.csv file
	return df

def load_sync_pulses_data(sync_pulses_file):
	tobii_sync = json.load(open(sync_pulses_file)) # Run's sync pulses
	tobii_sync = {int(k)/1000000.: v for k, v in tobii_sync.iteritems()}
	return tobii_sync

def load_task_data(faces_file):
	task_data = json.load(open(faces_file)) #long json file
	return task_data

	
def fix_tables(df, tobii_sync, task_data):
	tobii_pulses = []
	
	i = 0
	for k, v in sorted(tobii_sync.iteritems()):
		if v == 0 and i > 1:
			tobii_pulses.append(k)
		i += 1
   	
   	# Convert to seconds
	df['index_secs'] = df['index']/1000000.
   	
   	#Connect task time to sync pulses
	sync_i = 0
	for i, row in df.iterrows():
		curr_time = row.index_secs
		if curr_time > tobii_pulses[sync_i]:
			sync_i += 1
		try:
			df.loc[i, 'task_time'] = curr_time - tobii_pulses[sync_i] + task_data['sync_pulses'][sync_i]
		except IndexError:
			sync_i -= 1
			df.loc[i, 'task_time'] = curr_time - tobii_pulses[sync_i] + task_data['sync_pulses'][sync_i]
	
	# Calculate Pupil average between two eyes
	df['pupil_average'] = (df.l_pup_diam + df.r_pup_diam)/2.

	return df, task_data, tobii_sync

def split_target(df, task_data, tobii_sync):
	# Split trials between fear and non fear trials
	
	fear_trials = []
	notfear_trials = []
	
	for i in range(len(task_data['is_fear'])):
		if task_data['is_fear'][i] == 1:
			fear_trials.append(df.pupil_average[(df.task_time > task_data['stim_start'][i] - 0.25) & (df.task_time < task_data['stim_start'][i] + 2.5)].values)
		else:
			notfear_trials.append(df.pupil_average[(df.task_time > task_data['stim_start'][i] - 0.25) & (df.task_time < task_data['stim_start'][i] + 2.5)].values)
	
	lengths = []

	for i in range(len(fear_trials)):
  		x = len(fear_trials[i])
   		lengths.append(x)
    
	for i in range(len(notfear_trials)):
		x = len(notfear_trials[i])
		lengths.append(x)
	
	minval = min(filter(None, lengths))

	to_delete = []
	
	for i in range(len(fear_trials)):
		array = fear_trials[i]
		if len(array) > minval:
			fear_trials[i] = array[:minval]
		if len(array) < minval:
			to_delete.append(i)

	for val in to_delete:
		del fear_trials[val]

	to_delete = []
	for i in range(len(notfear_trials)):
		array = notfear_trials[i]
		if len(array) > minval:
			notfear_trials[i] = array[:minval]
		if len(array) < minval:
			to_delete.append(i)

	for val in to_delete:
		del notfear_trials[val]
    
	fear_trials = np.array(fear_trials)
	notfear_trials = np.array(notfear_trials)

	return fear_trials, notfear_trials


def baseline_norm_sub(trials):
    #Take mean of first 50 columns and subtract that from all of the rows
    trials_norm = trials - np.tile(trials[:, :50].mean(axis=1).reshape(
                (trials.shape[0], 1)), trials.shape[1])
    
    # trials[0:15].mean(axis=0, keepdims=True) - trials
    return trials_norm

def baseline_norm_div(trials):
    #Take mean of first 15 columns and subtract that from all of the rows
    trials_norm = trials / np.tile(trials[:, :15].mean(axis=1).reshape(
                (trials.shape[0], 1)), trials.shape[1])
    
    # trials[0:15].mean(axis=0, keepdims=True) - trials
    return trials_norm
    
        
def plot_comparisons(fear_trials, notfear_trials):
	# Plot graphs
	
	fig1  = plt.figure()
	ax = plt.axes()
	
	plt.plot(np.nanmean(fear_trials, axis=0), label="Fear")
	plt.plot(np.nanmean(notfear_trials, axis=0), label="Other")
	leg = ax.legend()
	plt.title("Pupillary Response for Faces")
	
	fig2 = plt.figure()
	ax1 = fig2.add_axes([0, 0, 0.5, 0.5])
	ax2 = fig2.add_axes([0.5, 0, 0.5, 0.5])
	
	ax1.plot(np.nanmean(fear_trials, axis=0), color="Yellow", label="Fear Mean")
	ax2.plot(np.nanmean(notfear_trials, axis=0), color="Yellow", label="Other Mean")
	leg1 = ax1.legend()
	leg2 = ax2.legend()
	
	for i in range(len(fear_trials)):
		ax1.plot(fear_trials[i], color='blue')
    
	for i in range(len(notfear_trials)):
		ax2.plot(notfear_trials[i], color='red')    

def run_all(*args):
	# Run all the functions in this file to produce graphs with data modified
	
	file1, file2, file3 = args
	a, b, c = import_data(file1, file2, file3)
	d, e, f = fix_tables(a, b, c)
	g, h = split_target(d, e, f)
	return g, h

def run_all_sub(*args):
	g, h = run_all(*args)
	i = baseline_norm_sub(g)
	j = baseline_norm_sub(h)
	plot_comparisons(i, j)

def run_all_div(*args):
	g, h = run_all(*args)
	i = baseline_norm_div(g)
	j = baseline_norm_div(h)
	plot_comparisons(i, j)
	
'''	
	
# In[3]:

# Import data from pupil responses
df = pd.read_csv("run1.csv")


# In[4]:

# Import sync pulses from trial
tobii_sync = json.load(open("run1_sync_pulses.json"))


# In[5]:

# Convert Tobii_sync to seconds
tobii_sync = {int(k)/1000000.: v for k, v in tobii_sync.iteritems()}


# In[6]:

#Import task_data with experimental data (fear, stimulation times)
task_data = json.load(open('faces_run1_projID-aaqnbxq_recID-3s2wja6.json'))

'''

'''
# In[7]:

# Convert tobii_sync from dictionary to list with the first points cut out.
tobii_pulses = []
i = 0
for k, v in sorted(tobii_sync.iteritems()):
    if v == 0 and i > 1:
        tobii_pulses.append(k)
    i += 1


# In[8]:

tobii_pulses


# In[9]:

task_data # Is a dictionary- Referring keys to values (that are lists- each of 30 items = TRIALS)


# In[10]:

task_data['sync_pulses']


# In[11]:

# Convert index_secs to seconds instead of xxxseconds also
df['index_secs'] = df['index']/1000000.


# In[12]:

# Attach the task times to df index_secs
sync_i = 0
for i, row in df.iterrows():
    curr_time = row.index_secs
    if curr_time > tobii_pulses[sync_i]:
        sync_i += 1
    try:
        df.loc[i, 'task_time'] = curr_time - tobii_pulses[sync_i] + task_data['sync_pulses'][sync_i]
    except IndexError:
        sync_i -= 1
        df.loc[i, 'task_time'] = curr_time - tobii_pulses[sync_i] + task_data['sync_pulses'][sync_i]


# In[13]:

# Example of finding a section of pupil diameter for a set period of time
df.l_pup_diam[(df.task_time > 50) & (df.task_time<60)]


# WHAT AM I TRYING TO DO:
# 
# Taking an average of the pupil diameter (pupil_average)
# Splitting time into chunks between stim start + 2/3 seconds => Trials
#     -This is the x-axis? 0 being start of trial
# Splitting them into fear or not (is_fear = 0 or 1)
#     Put them in two arrays for fear or not
# Averaging between the trials (all points from trials at each time point / len(task_data['stim_end'])
#     -Do they have to be the same size exactly for this to work?
# Plotting trial averages versus time

# In[14]:

df['pupil_average'] = (df.l_pup_diam + df.r_pup_diam)/2.
df # Is dataframe - indices referring to multiple arrays. Columns are attributes of df. 
'''

'''
# In[15]:

# Example of plotting a section of pupil averages to a set period of time
fig = plt.figure()
axs = plt.axes()
plt.plot(df.pupil_average[(df.task_time > 50) & (df.task_time<60)])


# In[16]:
'''

'''
# Split data from fear trials and not-fear trials into two buckets to plot separately.
# Time frame is 1/4 second before trial start to 2.5 seconds after stim.
fear_trials = []
notfear_trials = []
for i in range(len(task_data['is_fear'])):
    if task_data['is_fear'][i] == 1:
        fear_trials.append(df.pupil_average[(df.task_time > task_data['stim_start'][i] - 0.25) & (df.task_time < task_data['stim_start'][i] + 2.5)].values)
    else:
        notfear_trials.append(df.pupil_average[(df.task_time > task_data['stim_start'][i] - 0.25) & (df.task_time < task_data['stim_start'][i] + 2.5)].values)


# In[43]:

# Cut arrays down to the same length
lengths = []

for i in range(len(fear_trials)):
    x = len(fear_trials[i])
    lengths.append(x)
    
for i in range(len(notfear_trials)):
    x = len(notfear_trials[i])
    lengths.append(x)

print min(lengths)

for i in range(len(fear_trials)):
    array = fear_trials[i]
    if len(array) > min(lengths):
        fear_trials[i] = array[:min(lengths)]

for i in range(len(notfear_trials)):
    array = notfear_trials[i]
    if len(array) > min(lengths):
        notfear_trials[i] = array[:min(lengths)]


# In[44]:

# Turn lists into arrays

fear_trials = np.array(fear_trials)
notfear_trials = np.array(notfear_trials)

'''

'''
# In[48]:

# Plot the means of the two fear and not fear groups
plt.plot(np.nanmean(fear_trials, axis=0))
plt.plot(np.nanmean(notfear_trials, axis=0))

# Add Titles, Axes, Legend, Vertical Line showing axis start


# In[101]:

fig = plt.figure()
ax1 = fig.add_axes([0, 0, 0.5, 0.5],
                      ylim=(2.9, 3.8))
ax2 = fig.add_axes([0.5, 0, 0.5, 0.5],
                       ylim=(2.9, 3.8))


# ax1 = plt.axes([0, 0, .5, .5]) # standard axes
# ax2 = plt.axes([0.5, 0, 0.5, 0.5])

for i in range(len(fear_trials)):
    ax1.plot(fear_trials[i], color='blue')
    
for i in range(len(notfear_trials)):
    ax2.plot(notfear_trials[i], color='red')

'''