import km

# Load digits data
from sklearn import datasets
data, labels = datasets.load_breast_cancer().data, datasets.load_breast_cancer().target

# Create images for a custom tooltip array
import StringIO
from scipy.misc import imsave, toimage
import base64

# Initialize to use t-SNE with 2 components (reduces data to 2 dimensions). Also note high overlap_percentage.
mapper = km.KeplerMapper( verbose=2)

# Fit and transform data
#data = mapper.fit_transform(data, projection = km.manifold.TSNE())
data = mapper.fit_transform(data, projection = km.manifold.TSNE())

# Create the graph
complex = mapper.map(data, nr_cubes=25, overlap_perc=0.4, clusterer=km.cluster.DBSCAN(eps=0.3, min_samples=10), )

# Create the visualizations (increased the graph_gravity for a tighter graph-look.)

# Tooltips with the target y-labels for every cluster member
mapper.visualize(complex, path_html="keplermapper_breast_cancer_ylabel_tooltips.html", title="Digits", graph_gravity=0.25, custom_tooltips=labels)
