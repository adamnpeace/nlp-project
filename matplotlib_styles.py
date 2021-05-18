import matplotlib.pyplot as plt
import seaborn as sns

sns.set(rc={
 'axes.axisbelow': False,
 'axes.edgecolor': 'lightgrey',
 'axes.facecolor': 'None',
 'axes.grid': False,
 'axes.labelcolor': 'dimgrey',
 'axes.spines.right': False,
 'axes.spines.top': False,
 'figure.facecolor': 'white',
 'lines.solid_capstyle': 'round',
 'patch.edgecolor': 'w',
 'patch.force_edgecolor': True,
 'text.color': 'dimgrey',
 'xtick.bottom': False,
 'xtick.color': 'dimgrey',
 'xtick.direction': 'out',
 'xtick.top': False,
 'ytick.color': 'dimgrey',
 'ytick.direction': 'out',
 'ytick.left': False,
 'ytick.right': False})

sns.set_context("notebook", rc={"font.size":12,
                                "axes.titlesize":15,
                                "axes.labelsize":12})

blue = '#2979FF'
light_blue = "#00B0FF"
green = '#00BFA5'
pink = '#FF4081'
red = "#F44336"
purple = '#AA00FF'
violet = '#6200EA'
amber = '#FFAB00'

color_list = [blue, pink, green, amber,
              purple, violet]

plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
plt.rcParams['figure.figsize'] = (12, 8)
