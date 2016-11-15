# Import the class
import km

# Some sample data
from sklearn import datasets
data, labels = datasets.make_circles(n_samples=5000, noise=0.03, factor=0.3)

# Initialize
mapper = km.KeplerMapper(verbose=1)

# Fit to and transform the data
projected_data = mapper.fit_transform(data, projection=[0]) # X-Y axis

# Create dictionary called 'complex' with nodes, edges and meta-information
complex = mapper.map(projected_data, data, nr_cubes=10)

# Visualize it
mapper.visualize(complex, path_html="make_circles_keplermapper_output.html", 
                 title="make_circles(n_samples=5000, noise=0.03, factor=0.3)")
