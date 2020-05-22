import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def auto_label(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height), # put the detail data
                    xy=(rect.get_x() + rect.get_width() / 2, height), # get the center location.
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def auto_text(rects):
    for rect in rects:
        ax.text(rect.get_x(), rect.get_height(), rect.get_height(), ha='left', va='bottom')

#bar chart
labels = ['0.50:0.95','0.50','0.75']
imap = [0.278,0.564,0.242]
apperson = [0.267,0.616,0.180]
apbicycle=[0.122,0.346,0.067]
apcar=[0.445,0.729,0.479]

index = np.arange(len(labels))
width = 0.2

fig, ax = plt.subplots()
rect1 = ax.bar(index, imap,  width=width, label ='MAP')
rect2 = ax.bar(index+0.2, apperson,  width=width, label ='person')
rect3 = ax.bar(index+0.4, apbicycle, width=width, label ='bicycle')
rect4 = ax.bar(index+0.6, apcar,  width=width, label ='car')


ax.set_title('EVAL')
ax.set_xticks(ticks=index+0.3)
ax.set_xticklabels(labels)
ax.set_ylabel('AP')

ax.set_ylim(0,1)
# auto_label(rect1)
# auto_label(rect2)
auto_text(rect1)
auto_text(rect2)
auto_text(rect3)
auto_text(rect4)


ax.legend(loc='upper right', frameon=False)
fig.tight_layout()
# plt.savefig('2.tif', dpi=300)
plt.show()

