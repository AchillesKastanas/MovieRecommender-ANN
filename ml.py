import csv
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def check_for_duplicates(array):
    if len(array) != len(set(array)):
        print("There are duplicates in the array.")
    else:
        print("There are no duplicates in the array.")
        
        

# ----------------------------------------------------
#
#* Init
#
# ----------------------------------------------------

folder = 'moviesReviews'
U = []
UReviews = []
I = 0
folder = 'moviesReviews'


# ----------------------------------------------------
#
#* Load all the movies
#
# ----------------------------------------------------

I = len([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])



# ----------------------------------------------------
#
#* Load all the users (use maxMoviesToRead for fast testing)
#
# ----------------------------------------------------

counter = 0
#! LIMITER FOR FAST TESTING - REMOVE LATER
maxMoviesToRead = 40
for filename in os.listdir(folder):
    if filename.endswith(".csv"):
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
                        # Add the row to the users reviews
                        UReviews[len(UReviews) - 1].append(row)
                    # It is not the first time spotting the user
                    else:
                        # print("USER SPOTTED AGAIN" + username)
                        #Get the index of the user
                        userIndex = U.index(username)
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
        userDates.append(datetime.strptime(review[4], "%d %B %Y"))
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