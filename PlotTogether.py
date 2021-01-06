import json
import matplotlib.pyplot as plt

experiment_folder = './output'
def load_json_arr(json_path):
    lines = []
    with open(json_path, 'r') as f:
        for line in f:
            lines.append(json.loads(line))
    return lines

experiment_metrics = load_json_arr(experiment_folder + '/metrics.json')   

plt.plot(
    [x['iteration'] for x in experiment_metrics if 'total_loss' in x], 
    [x['total_loss'] for x in experiment_metrics if 'total_loss' in x])
plt.plot(
    [x['iteration'] for x in experiment_metrics if 'validation_loss' in x], 
    [x['validation_loss'] for x in experiment_metrics if 'validation_loss' in x])
plt.legend(['train_loss', 'validation_loss'], loc='upper left')
plt.xlim(0, 1000)
plt.ylim(0, 1)
plt.show()

     
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
plt.xlim(0, 40000)
plt.ylim(0, 100)
plt.show()
