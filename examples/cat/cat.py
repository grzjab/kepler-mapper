import km
from mst_clustering import MSTClustering

data = km.np.genfromtxt('examples/cat/cat-reference.csv',delimiter=',')

mapper = km.KeplerMapper( verbose=2)

projected_data = mapper.fit_transform(data, projection = [1] )

# Create the graph
complex = mapper.map(projected_data, data, nr_cubes=10, overlap_perc=0.7, clusterer = MSTClustering(cutoff_scale=0.025)) #clusterer=km.cluster.DBSCAN(eps=0.1, min_samples=5), )

# Tooltips with the target y-labels for every cluster member
mapper.visualize(complex, path_html="keplermapper_cat_ylabel_tooltips.html", title="Cat", graph_gravity=0.05)

# You may want to visualize the original point cloud data in 3D scatter too
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:,0],data[:,1],data[:,2])
plt.savefig("cat-reference.csv.png")
plt.show()
"""
