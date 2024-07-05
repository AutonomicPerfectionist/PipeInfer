import matplotlib.pyplot as plt
import csv

import numpy as np

models = ["Dolphin", "Goliath", "Falcon"]
impls = ["Sequential", "Speculative", "PipeInfer"]

x = np.arange(len(impls))

_, speed_axes = plt.subplots(layout='constrained')
_, itl_axes = plt.subplots(layout='constrained')
_, ttft_axes = plt.subplots(layout='constrained')


width = 0.3  # the width of the bars
multiplier = 0
with open('results.csv', 'r') as csvfile:
    data = iter(csv.reader(csvfile, delimiter=','))
    grouped_data = zip(data, data, data)

    for index, row in enumerate(grouped_data):
        offset = width * multiplier
        rects = speed_axes.bar(x + offset, [float(elem[1]) for elem in row], width, label=models[index % len(models)])
        speed_axes.bar_label(rects, padding=5)
        rects = itl_axes.bar(x + offset, [float(elem[2]) for elem in row], width, label=models[index % len(models)])
        itl_axes.bar_label(rects, padding=5)
        rects = ttft_axes.bar(x + offset, [float(elem[3]) for elem in row], width, label=models[index % len(models)])
        ttft_axes.bar_label(rects, padding=5)
        multiplier += 1


speed_axes.set_ylabel('Speed (tokens / sec)')
speed_axes.set_title('Generation Speed')
speed_axes.set_xticks(x + width, impls)
speed_axes.legend(loc='upper left', ncols=3)

itl_axes.set_ylabel('Inter-token Latency (seconds)')
itl_axes.set_title('ITL')
itl_axes.set_xticks(x + width, impls)
itl_axes.legend(loc='upper left', ncols=3)

ttft_axes.set_ylabel('Time to First Token (seconds)')
ttft_axes.set_title('TTFT')
ttft_axes.set_xticks(x + width, impls)
ttft_axes.legend(loc='upper left', ncols=3)

plt.show()
