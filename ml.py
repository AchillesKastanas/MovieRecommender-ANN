import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import euclidean, cosine

#TODO THE MOVIE NAMES ARE STORED LIKE THE FILENAME.CSV - REMOVE THE .CSV

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