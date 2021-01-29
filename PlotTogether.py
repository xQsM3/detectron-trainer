import json
import matplotlib.pyplot as plt
import sys
import numpy as np
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

experiment_folder = str(sys.argv[1])
def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

## TRAINING AND VALIDATION LOSS + learning rate
experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')
#print(experiment_metrics)
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)
ax2=ax.twinx()

# loss 
ax.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
ax.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])

# learning rate
ax2.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['lr'] for x in experiment_metrics if 'total_loss' in x],color="C6")


l = []
[l.append(x['iteration']) for x in experiment_metrics if 'total_loss' in x]

plt.xlim(0,max(l))
ax.grid()

# y-axis label
ybox1 = TextArea("training loss ", textprops=dict(color="C0", size=14,rotation='vertical'))
ybox2 = TextArea("and ", textprops=dict(color="k", size=15,rotation='vertical'))
ybox3 = TextArea("                 validation loss ", textprops=dict(color="C1", size=14,rotation='vertical'))

ybox = VPacker(children=[ybox1, ybox2, ybox3],
                  align="center", pad=0, sep=5)

anchored_ybox = AnchoredOffsetbox(loc=7,child=ybox, pad=0., frameon=False,
                                      bbox_to_anchor=(-0.08, 0.4),
                                      bbox_transform=ax.transAxes, borderpad=0.)
ax.add_artist(anchored_ybox)                                      
ax.set_ylim(0, 1.2)                                     
ax2.set_ylabel("learning rate",fontsize=14,color="C6")
ax.set_xlabel("iterations",fontsize=14)
plt.show()





## AP AP75 AP50
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)     
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x], 
    [x['bbox/AP'] for x in experiment_metrics if 'bbox/AP' in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x], 
    [x['bbox/AP50'] for x in experiment_metrics if 'bbox/AP' in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x], 
    [x['bbox/AP75'] for x in experiment_metrics if 'bbox/AP' in x])        
plt.legend(['AP', 'AP50','AP75'], loc='upper left')
plt.xlim(0,max(l))
plt.ylim(0, 100)
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 101, 20)
minor_ticks = np.arange(0, 101, 5)

#ax.set_xticks(major_ticks)
#ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')
#plt.grid()
ax.set_xlabel("iterations",fontsize=14)	
plt.show()

## APs APm APl
fig = plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(1, 1, 1)
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x], 
    [x['bbox/APs'] for x in experiment_metrics if 'bbox/AP' in x])     
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x], 
    [x['bbox/APm'] for x in experiment_metrics if 'bbox/AP' in x]) 
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'bbox/AP' in x], 
    [x['bbox/APl'] for x in experiment_metrics if 'bbox/AP' in x]) 
plt.legend(['APs','APm','APl'], loc='upper left')
plt.xlim(0,max(l))
plt.ylim(0, 100)
# Major ticks every 20, minor ticks every 5
major_ticks = np.arange(0, 101, 20)
minor_ticks = np.arange(0, 101, 5)

#ax.set_xticks(major_ticks)
#ax.set_xticks(minor_ticks, minor=True)
ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

ax.grid(which='both')
ax.set_xlabel("iterations",fontsize=14)
plt.show()

## LEARNING RATE
plt.figure(num=None, figsize=(8, 6), dpi=80, facecolor='w', edgecolor='k')
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['lr'] for x in experiment_metrics if 'total_loss' in x])
plt.legend(['learning rate'], loc='upper left')

l = []
[l.append(x['iteration']) for x in experiment_metrics if 'total_loss' in x]

plt.xlim(0,max(l))
plt.ylim(0, 0.021)
plt.grid()
plt.xlabel("iterations",fontsize=14)
plt.show()
