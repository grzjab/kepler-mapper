import km
import numpy as np
from mst_clustering import MSTClustering

infile = file('data.npy','r')
data = np.load(infile)
infile.close()
infile = file('labels.npy','r')
labels = np.load(infile)
infile.close()

mapper = km.KeplerMapper( verbose=2)

projected_data = mapper.fit_transform(data, projection = km.manifold.TSNE(metric='jaccard') )
print projected_data

# Create the graph
complex = mapper.map(projected_data, data, nr_cubes=10, overlap_perc=0.5, clusterer=km.cluster.DBSCAN(eps=0.3, min_samples=5, metric='jaccard'),) 

# Tooltips with the target y-labels for every cluster member
mapper.visualize(complex, path_html="keplermapper_votes_ylabel_tooltips.html", title="Votes", graph_gravity=0.15, custom_tooltips=labels)

# You may want to visualize the original point cloud data in 3D scatter too
import matplotlib.pyplot as plt

plt.plot(projected_data[:,0],projected_data[:,1],'.')
#plt.savefig("cat-reference.csv.png")
plt.show()
