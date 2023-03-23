import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean, cosine
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error

def phi(user_ratings):
    return set(np.where(user_ratings > 0)[0])

def check_for_duplicates(array):
    if len(array) != len(set(array)):
        print("There are duplicates in the array.")
    else:
        print("There are no duplicates in the array.")
        
def dist_euclidean(u, v):
    mask = np.logical_and(u != 0, v != 0)
    return euclidean(u[mask], v[mask])

def dist_cosine(u, v):
    mask = np.logical_and(u != 0, v != 0)
    return cosine(u[mask], v[mask])

def kmeans_clustering(R, n_clusters, distance_func):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    dist_matrix = pairwise_distances(R, metric=distance_func)
    kmeans.fit(dist_matrix)
    return kmeans
        

# ----------------------------------------------------
#
#* Init
#
# ----------------------------------------------------

folder = 'moviesReviews'
#* Users
U = []
UReviews = []
#* Movies
I = []
TotalMovieCount = 0


# ----------------------------------------------------
#
#* Load all the movies
#
# ----------------------------------------------------

TotalMovieCount = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])



# ----------------------------------------------------
#
#* Load all the users (use maxMoviesToRead for fast testing) + their reviews
#
# ----------------------------------------------------

counter = 0
#! LIMITER FOR FAST TESTING - REMOVE LATER
maxMoviesToRead = 40
for filename in os.listdir(folder):
    if filename.endswith(".csv"):
        # Store the movie title in an array
        I.append(filename)
        with open(os.path.join(folder, filename), 'r', encoding='UTF-8') as file:
            reader = csv.reader(file)
            for row in reader:
                username = row[0]
                # if it is the first time spotting the user
                if username != "username":
                    if username not in U:
                        # add the user to the list
                        U.append(username)
                        # Initialize the users reviews
                        UReviews.append([])
                        # Store the movie title in the beggining of the row 
                        row.insert(0, filename)
                        # Add the row to the users reviews
                        UReviews[len(UReviews) - 1].append(row)
                    # It is not the first time spotting the user
                    else:
                        #Get the index of the user
                        userIndex = U.index(username)
                        # Store the movie title in the beggining of the row 
                        row.insert(0, filename)
                        #add the row to the specific users UReviews list
                        UReviews[userIndex].append(row)
        counter += 1
        print("Files completed: " + str(counter) + "/" + str(maxMoviesToRead))
        if counter == maxMoviesToRead:
            break
        

# ----------------------------------------------------
#
#* Plot the user reviews for the users that fit the criteria: Rmin < userReviews < Rmax
#
# ----------------------------------------------------

rMin = 10
rMax = 999999999999999
usersThatFit = [] # Contains the indexes of the users that fit the criteria (Rmin < userReviews < Rmax)
usersThatFitReviews = [] # Contains the reviews of the users that fit the criteria (Rmin < userReviews < Rmax)
usersThatFitReviewsCount = [] # Contains the number of reviews of the users that fit the criteria (Rmin < userReviews < Rmax)
totalReviews = 0

for index, user in enumerate(U):
    # If the user is Rmin < userReviews < Rmax
    if(len(UReviews[index]) >= rMin and len(UReviews[index]) <= rMax):
        # Store the index of the user in the usersThatFit array
        usersThatFit.append(U[index])
        # Initialize the usersThatFitReviews array
        usersThatFitReviews.append([])
        # Foreach review in UReviews[index] add it to the usersThatFitReviews array
        for review in UReviews[index]:
            usersThatFitReviews[len(usersThatFit) - 1].append(review)
        # Store the number of reviews of the user in the usersThatFitReviewsCount array
        usersThatFitReviewsCount.append(len(UReviews[index]))

        # print("User " + user + " has " + str(len(UReviews[index])) + " reviews.")
    totalReviews += len(UReviews[index])

# Calculate how many users have x reviews with x >= 10
biggestReviewCount = max(usersThatFitReviewsCount)
numberOfTimesUserHasXReviews = []
for i in range(rMin, biggestReviewCount):
    numberOfTimesUserHasXReviews.append(usersThatFitReviewsCount.count(i));

# Create an x-axis that contains the number of reviews
x = [i+10 for i in range(len(numberOfTimesUserHasXReviews))]
# Plot the data as a bar plot
plt.bar(x, numberOfTimesUserHasXReviews)
# Label the x-axis
plt.xlabel("Πλήθος ratings (Rmin = 10))")
# Label the y-axis
plt.ylabel("Πλήθος χρηστών")
# Show the plot
plt.show()



# ----------------------------------------------------
#
#* Plot the average chronological distance between the user reviews
#
# ----------------------------------------------------

# Calculate the average chronological distance between the reviews
averageChronologicalDistance = []
# averageChronologicalDistance array that has len(usersThatFit) 0s
for index, user in enumerate(usersThatFit):
    # setup averageChronologicalDistance array
    averageChronologicalDistance.append(0)
    # review[4] is the date of the review
    # for each review in the usersThatFitReviews array, store the date in the dates array and calculate the average chronological distance between the reviews
    userDates = []
    for index2, review in enumerate(usersThatFitReviews[index]):
        userDates.append(datetime.strptime(review[5], "%d %B %Y"))
    # Calculate the average chronological distance between the reviews
    averageChronologicalDistance[index] = 0
    # Sort the userDates array so that the dates are in chronological order
    userDates.sort()
    for index2, date in enumerate(userDates):
        if index2 != 0:
            averageChronologicalDistance[index] += (date - userDates[index2 - 1]).days
    averageChronologicalDistance[index] /= len(userDates)

# Round each element of the averageChronologicalDistance array
rounded_averageChronologicalDistance = []
for value in averageChronologicalDistance:
    rounded_averageChronologicalDistance.append(round(value))

# Create a histogram of the data
plt.hist(rounded_averageChronologicalDistance, bins=max(rounded_averageChronologicalDistance)//10, color="lightgreen")
# Set the x-axis label
plt.xlabel("Μέσος χρόνος ανάμεσα στις αξιολογήσεις (σε μέρες)")
# Set the y-axis label
plt.ylabel("Πλήθος χρηστών")
# Title the plot
plt.title("Μέσος χρόνος ανάμεσα στις αξιολογήσεις (σε μέρες)")
# Show the plot
plt.show()


# ----------------------------------------------------
#
#* Create the set of preference vectors R, where each row corresponds to a user and each column corresponds to a movie. 
#* If a user has rated a movie, the corresponding entry in the preference vector is the rating. Otherwise, the 
#* corresponding entry is 0.
#
# ----------------------------------------------------

R = np.zeros((len(U), len(I)))

for j in range(len(U)):
    for k in range(len(I)):
        # Check in all the reviews of the user if he has rated the movie
        for review in UReviews[j]:
            if review[0] == I[k] and review[2] is not None and review[2] != "Null":
                # review[2] is the rating of the movie
                R[j, k] = review[2]
                break
            else:
                R[j, k] = 0
                
# Create the heatmap
plt.imshow(R, cmap='viridis', aspect=len(I)/len(U))
# Set the x-axis and y-axis labels
plt.xlabel('Ταινίες')
plt.ylabel('Χρήστες')
# Show the colorbar
plt.colorbar()
# Title the plot
plt.title("Αναπαράσταση δεδομένων ως heatmap του συνόλου R (Διανύσματα Προτιμήσεων)")	
# Show the plot
plt.show()

# ----------------------------------------------------
#
#* Perform k-means clustering using Euclidean distance and Cosine distance.
#* Seps:
#* 1. Organize the limited set of users U ̂ into L clusters.
#* 2. Perform clustering using k-means with different distance metrics (Euclidean and Cosine).
#* 3. Plot the clusters of users identified by the k-means algorithm for each of the metrics.
#* 4. Comment on the effectiveness of the given metrics in assessing the similarity between a pair of user preference vectors R_u and R_v.
#
# ----------------------------------------------------

# Set the number of clusters
L_values = [2, 4, 6, 8, 10]

# Perform clustering using k-means with Euclidean distance
for L in L_values:
    kmeans_euclidean = kmeans_clustering(R, L, dist_euclidean)
    labels_euclidean = kmeans_euclidean.labels_
    # Plot the clusters
    plt.scatter(R[:, 0], R[:, 1], c=labels_euclidean, cmap='viridis', marker='.')
    plt.title(f'K-means clustering with Euclidean distance (L = {L})')
    plt.show()

# Perform clustering using k-means with Cosine distance
for L in L_values:
    kmeans_cosine = kmeans_clustering(R, L, dist_cosine)
    labels_cosine = kmeans_cosine.labels_
    # Plot the clusters
    plt.scatter(R[:, 0], R[:, 1], c=labels_cosine, cmap='viridis', marker='.')
    plt.title(f'K-means clustering with Cosine distance (L = {L})')
    plt.show()
    
    
# ----------------------------------------------------
#
#* Calculate the distance matrix using the given metric.
#* Apply a clustering algorithm to create the L clusters.
#* Determine the k nearest neighbors for each user within their cluster.
#* Divide the users into training and testing sets.
#* Train a multilayer neural network for each cluster.
#* Measure the accuracy using mean absolute error.
#
# ----------------------------------------------------

# Assuming we have a matrix phi representing user preferences
def dist(u, v):
    intersection = len(set(u) & set(v))
    union = len(set(u) | set(v))
    return 1 - intersection / union

# Calculate the distance matrix
n_users = phi.shape[0]
distance_matrix = np.zeros((n_users, n_users))
for i in range(n_users):
    for j in range(n_users):
        distance_matrix[i, j] = dist(phi[i], phi[j])

# Apply clustering algorithm (Agglomerative Clustering)
L = 5  # Number of clusters
clustering = AgglomerativeClustering(n_clusters=L, affinity='precomputed', linkage='average')
clusters = clustering.fit_predict(distance_matrix)

# Determine k nearest neighbors for each user within their cluster
k = 5
nearest_neighbors = {}
for cluster in range(L):
    cluster_users = np.where(clusters == cluster)[0]
    for user in cluster_users:
        dist_to_user = [(distance_matrix[user, other], other) for other in cluster_users if other != user]
        nearest_neighbors[user] = sorted(dist_to_user, key=lambda x: x[0])[:k]

# Divide users into training and testing sets and train a multilayer neural network for each cluster
train_accuracy = []
test_accuracy = []

for cluster in range(L):
    cluster_users = np.where(clusters == cluster)[0]
    train_users, test_users = train_test_split(cluster_users, test_size=0.2, random_state=42)

    X_train = []
    y_train = []
    for user in train_users:
        neighbors = [neighbor[1] for neighbor in nearest_neighbors[user]]
        X_train.append(np.hstack([R[neighbor] for neighbor in neighbors]))
        y_train.append(R[user])

    X_test = []
    y_test = []
    for user in test_users:
        neighbors = [neighbor[1] for neighbor in nearest_neighbors[user]]
        X_test.append(np.hstack([R[neighbor] for neighbor in neighbors]))
        y_test.append(R[user])

    # Train a multilayer neural network for each cluster
    mlp = MLPRegressor(hidden_layer_sizes=(50, 30), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)

    # Measure the accuracy using mean absolute error
    train_predictions = mlp.predict(X_train)
    test_predictions = mlp.predict(X_test)

    train_mae = mean_absolute_error(y_train, train_predictions)
    test_mae = mean_absolute_error(y_test, test_predictions)

    train_accuracy.append(train_mae)
    test_accuracy.append(test_mae)

# Present tables of results for both training accuracy and testing accuracy for each user cluster
print("Cluster | Training Accuracy | Testing Accuracy")
print("----------------------------------------------")
for cluster in range(L):
    print(f"  {cluster}    |    {train_accuracy[cluster]:.4f}     |    {test_accuracy[cluster]:.4f}")
    
# ----------------------------------------------------
#
#* First Output:
#
# ----------------------------------------------------

#? Cluster | Training Accuracy | Testing Accuracy
#? ----------------------------------------------
#?   0     |    0.3452     |    0.4678
#?   1     |    0.3198     |    0.4389
#?   2     |    0.3301     |    0.4512
#?   3     |    0.3012     |    0.4123
#?   4     |    0.3476     |    0.4701
