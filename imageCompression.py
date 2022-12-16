import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import imageio

def read_image():
	# loading the png image as a 3d matrix
	img = imageio.imread('dog.png')

	plt.imshow(img) # plotting the image
	plt.show()
	
	img = img / 255
	return img

def initialize_means(img, clusters=15):
	# reshaping it or flattening it into a 2d matrix
	points = np.reshape(img, (img.shape[0] * img.shape[1], img.shape[2]))
	_, n = points.shape
	
	# means is the array of assumed means or centroids.
	means = np.zeros((clusters, n))

	# random initialization of means.
	for i in range(clusters):
		rand1 = int(np.random.random(1)*10)
		rand2 = int(np.random.random(1)*8)
		means[i, 0] = points[rand1, 0]
		means[i, 1] = points[rand2, 1]

	return points, means

# Function to measure the euclidean
# distance (distance formula)
def distance(x1, y1, x2, y2):
	d = ((x1-x2)**2+(y1-y2)**2)**0.5
	return d


def k_means(points, means, clusters):
	iterations = 10 # the number of iterations
	m, _ = points.shape
	
	index = np.zeros(m)

	# k-means algorithm.
	while(iterations > 0):
		for j in range(len(points)):
			minv = 1000
			temp = None
			for k in range(clusters):
				x1 = points[j, 0]
				y1 = points[j, 1]
				x2 = means[k, 0]
				y2 = means[k, 1]
				if(distance(x1, y1, x2, y2) < minv):		
					minv = distance(x1, y1, x2, y2)
					temp = k
					index[j] = k
		
		for k in range(clusters):
			sumx = 0
			sumy = 0
			count = 0
			for j in range(len(points)):
				if(index[j] == k):
					sumx += points[j, 0]
					sumy += points[j, 1]
					count += 1
			
			if(count == 0):
				count = 1	
			
			means[k, 0] = float(sumx / count)
			means[k, 1] = float(sumy / count)	
			
		iterations -= 1

	return means, index


def compress_image(means, index, img):
	centroid = np.array(means)
	recovered = centroid[index.astype(int), :]
	
	# getting back the 3d matrix (row, col, rgb(3))
	recovered = np.reshape(recovered, (img.shape[0], img.shape[1], img.shape[2]))

	# plotting the compressed image.
	plt.imshow(recovered)
	plt.show()

	# saving the compressed image.
	imageio.imwrite('compressed_' + str(clusters) +'_colors.png', recovered)

img = read_image()

clusters = 16
clusters = int(input('Enter the number of clusters : '))
points, means = initialize_means(img, clusters)
means, index = k_means(points, means, clusters)
compress_image(means, index, img)
