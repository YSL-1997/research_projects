# Import necessary packages
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from sklearn.utils import resample
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

# Read in source data
df = pd.read_excel("supp4.xls", header=0)
dataset = df.iloc[:, :]

# Random sampling 
# Randomly select 1% source data for ANN model construction; set seed to be 1
data_train = []
random.seed(7)
ini_train_percent = 0.1
#train_size = int(ini_train_percent*len(dataset))
#test_size = len(dataset) - train_size
rand_idxs = random.sample(range(0, len(dataset)),int(ini_train_percent*len(dataset)))
for idx in rand_idxs:
    data_train.append(np.array(dataset.iloc[idx, :]))
data_train = np.array(data_train).squeeze()
X_train = data_train[:, 0:2].astype(np.float)
Y_train = data_train[:, 4:].astype(np.float)

# Use the remaining 70% source data to test the model
data_test =[]
for idx in range (0, len(dataset)):
    if idx not in rand_idxs:
        data_test.append(np.array(dataset.iloc[idx, :]))
data_test = np.array(data_test).squeeze()
X_test = data_test[:, 0:2].astype(np.float)
Y_test = data_test[:, 4:].astype(np.float)


# Preprocessing the training data and reduce dimesion by PCA
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#scaler = StandardScaler(with_mean=False, with_std=False)
#Y_train = scaler.fit_transform(Y_train)
num_components = 20
pca = PCA(n_components=num_components)
Y_train = pca.fit(Y_train).transform(Y_train)
Y_test = pca.fit(Y_test).transform(Y_test)


# Define the baseline model
def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(
        8,
        input_dim=X_train.shape[1],
        kernel_initializer='normal',
        activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(
        16,
        kernel_initializer='normal',
        activation='relu'))
    model.add(Dropout(0.20))
    model.add(Dense(
        num_components,
        kernel_initializer='normal'))
    # Compile model
    #model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='mean_absolute_error', optimizer='adam')
    return model
    
num_iteration = 20
for i in range(num_iteration):
    # Parameter: number of folds for cross-validation
    k = 5;
    # Parameter: number of ensembled models
    numB = 10;
    
    train_size = len(X_train)
    test_size = len(X_test)
    
    cv = KFold(n_splits=k)
    cv_split = cv.split(X_train, Y_train)
    
    X_trs = []
    Y_trs = []
    X_cv_val = [] # test here is used for validation
    Y_cv_val = []
    for train, val in cv_split:
        X_cv_val.append(X_train[val])
        Y_cv_val.append(Y_train[val])
        X_temp = X_train[train]
        Y_temp = Y_train[train]
        X_cv_train = []
        Y_cv_train = []
        for i in range(numB):
            X_temp1, Y_temp1 = resample(X_temp, Y_temp)
            X_cv_train.append(X_temp1)
            Y_cv_train.append(Y_temp1)
        # three layer list kfold*numB*data
        X_trs.append(X_cv_train)
        Y_trs.append(Y_cv_train)
    
    # Train all numB models with bootstrap training sets
    #Bmodels = []  # list to store all numB models
    #histories = []
    #results = []
    pre_results = []
    for i in range(k):
        print('/////////////////   ITR ', i, '   ///////////////////')
        #cv_model = []
        for j in range(numB):
            b_model = baseline_model()
            b_model.fit(X_trs[i][j], Y_trs[i][j],
                        batch_size=5,
                        epochs=200)
            #cv_model.append(b_model)
            # use the data we have to predict the experiments that we have not done
            pre_result = b_model.predict(X_test)
        # two layers
        #Bmodels.append(cv_model)
        pre_results.append(pre_result)
    
    # Calculate the standard deviation of the predicted values of each point across all models
    # Create the list containing (index, std) for each point in testing set
    std_list = []
    for i in range(test_size):
        pre_vector_cur_point = []
        for j in range(len(pre_results)):
            pre_vector_cur_point.append(pre_results[j][i, :])
        std_cur_point = np.std(pre_vector_cur_point)
        std_list.append((i, std_cur_point))
    # Sort the std list
    std_list = sorted(std_list,key=(lambda x:x[1]), reverse = True)
    
    # Convert testing data into a dictionary, with key = X, value = Y
    test_data_dict = {}
    for i in range(test_size):
        temp_tuple = (X_test[i][0], X_test[i][1])
        test_data_dict[temp_tuple] = Y_test[i]
    
    # When it is not the last round of iteration:
    if i < num_iteration:
        # Select the top 1% predicted points with the highest std
        # Update training data for next iteration
        new_training_X = []
        select_percent = 0.01
        for i in range(int(select_percent*test_size)):
            temp_index = std_list[i][0] # get the index of test data to be retrieved
            new_training_X.append([X_test[temp_index][0], X_test[temp_index][1]])
        new_training_Y = []
        for e in new_training_X:
            #temp_tuple = (e[0], e[1])
            new_training_Y.append((test_data_dict[(e[0], e[1])]).tolist())
    
        X_train = np.row_stack((X_train, np.array(new_training_X)))
        Y_train = np.row_stack((Y_train, np.array(new_training_Y)))
        
        # Update testing data for next iteration
        X_test = X_test.tolist()
        Y_test = Y_test.tolist()
        new_test_X =[]
        new_test_Y = []
        for i in range(len(X_test)):
            if X_test[i] not in new_training_X:
                new_test_X.append(X_test[i])
                new_test_Y.append(Y_test[i])
        # for i in range(len(Y_test)):
        #     if Y_test[i] not in new_training_Y:
        #         new_test_Y.append(Y_test[i])
        X_test = np.array(new_test_X)
        Y_test = np.array(new_test_Y)
    
    #if i == num_iteration-1:# Update train_size and test_size before exiting the loop
    #train_size = len(X_train)
    #test_size = len(X_test)

# Calculate the final prdicted result
temp_sum = np.zeros((test_size, num_components))
for i in range(len(pre_results)):
    temp_sum = temp_sum + pre_results[i]
final_predicted = temp_sum / len(pre_results)

# Use training data and TSNE to find the number of clusters
tsne_temp_Y = Y_train
from sklearn.manifold import TSNE
ts_iteration = 4
for i in range(ts_iteration):
    ts = TSNE(n_components=2, learning_rate = 200)
    ts.fit_transform(tsne_temp_Y)
    tsne_temp_Y = ts.embedding_
    print(ts.embedding_)
fig = plt.figure(figsize=(8, 8))
plt.scatter(tsne_temp_Y[:, 0], tsne_temp_Y[:, 1], c="blue", cmap=plt.cm.Spectral)
plt.show()

# Use training data and Elbow Method to find the number of clusters
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,40))
visualizer.fit(Y_train)        # Fit the data to the visualizer
visualizer.show() 


# Get the classification result for the actual Y values (Y_test)
clf1=KMeans(n_clusters=9)
clf1=clf1.fit(Y_test)
clf1.cluster_centers_
classify_actual_result = clf1.labels_

# get the classification result for the predicted Y values (predicted_results)
#for i in range(len(predict_each_model_point)):
clf2=KMeans(n_clusters=9)
clf2=clf2.fit(final_predicted)
clf2.cluster_centers_
classify_predicted_result = clf2.labels_


# function to get unique values 
def unique(list1): 
      
    # insert the list to the set 
    list_set = set(list1) 
    # convert the set to the list 
    unique_list = list(list_set)
    return unique_list
      
unique_clusters = unique(classify_actual_result)

def get_clusters(lst: list, unique_cluster_values: list):
    cluster = list() # to be returned
    for i in unique_cluster_values:
        sub_cluster = list()
        for j in range(len(lst)): # j is the index of the experiments
            if lst[j]==i:
                sub_cluster.append(j)
        cluster.append(sub_cluster)
    return cluster

cluster_actual = get_clusters(classify_actual_result, unique_clusters)
cluster_predict = get_clusters(classify_predicted_result, unique_clusters)


# overlap number of experiments between two clusters
# one cluster from predicted, the other one from real
def overlap_percentage(c1:list, c2: list):
    count = 0 # the value to be returned
    length = min(len(c1), len(c2))
    
    if length == len(c1):
        for i in c1:
            if i in c2:
                count += 1
        return count/length
    
    else:
        for i in c2:
            if i in c1:
                count += 1
        return count/length

def overlap_num(c1:list, c2: list):
    count = 0 # the value to be returned
    length = min(len(c1), len(c2))
    
    if length == len(c1):
        for i in c1:
            if i in c2:
                count += 1
        return count
    
    else:
        for i in c2:
            if i in c1:
                count += 1
        return count

    
# Given two lists, each list consists of the same number of clusters
# input: predicted_clusters: (p1, p2, ... pn), real_clusters: (r1, r2, ... rn)
#        each pi and ri are all list()
#
# output: a list of pairs [(pi, ri), ...] representing the result of the pairs of clusters
def accuracy_cluster(predicted_clusters: list, real_clusters: list):
    num_clusters = len(predicted_clusters)

    d = dict() 
    # a dict to store overlap_percentage of all pairs of clusters (predict, real)
    
    return_list = [] 
    # list of cluster pairs to be returned
    
    for i,p in enumerate(predicted_clusters):
        for j,r in enumerate(real_clusters):
            d[(i,j)] = overlap_percentage(p, r)

    # now, we have a dictionary that stores overlap_percentage of all pairs of clusters {(predict, real): overlap_percentage}
    
    all_values = list(d.values())
    all_keys = list(d.keys())

    
    loop_num = 1
    while len(return_list) != len(predicted_clusters):
        print("this is the ", loop_num, "loop")
        loop_num += 1
        
        # position is (i,j) - (pi, rj) - the pair of clusters with the largest overlap percentage
        position = all_keys[all_values.index(max(all_values))]
        print("the pair", position, "has the largest overlap_percentage", max(all_values))
        if predicted_clusters[position[0]] == 0 or real_clusters[position[1]] == 0:
            # one of or both clusters have been paired
            # need to remove this value and continue with the second largest value
            del all_keys[all_values.index(max(all_values))]
            del all_values[all_values.index(max(all_values))]
            
            continue
        else:
            # mark the two clusters in predicted_clusters and real_clusters as 0
            predicted_clusters[position[0]] = 0
            real_clusters[position[1]] = 0
            
            # add the two clusters to return_list
            return_list.append(position)
            
            # remove the largest value so far and go on with the next loop
            del all_keys[all_values.index(max(all_values))]
            del all_values[all_values.index(max(all_values))]
            
    
    # now, return_list is the list of pairs [(pi, ri), ...] representing the result of the pairs of clusters
    # need to calculate accuracy
    accuracy = 0
    up = 0 # numerator
    down = 0 # denominator
    
    for pair in return_list:
        # calculate # of inaccurate assignments in each pair of cluster
        num_overlap = overlap_num(predicted_clusters[pair[0]], real_clusters[pair[1]])
        up += num_overlap
        down += len(predicted_clusters[pair[0]])
    accuracy = up/down

    return accuracy

accuracy_cluster(cluster_actual,cluster_predict)
