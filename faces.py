from get_data import *
import re
import os
import numpy as np

'''------------------------------------------------------------------------------------------------------'''
'''                                       Global Variables                                               '''
'''------------------------------------------------------------------------------------------------------'''

subset_act = ['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act_test_subset = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']
test = ['Bill Hader', 'Steve Carell']
    
'''------------------------------------------------------------------------------------------------------'''
'''                                       Linear Regression                                              '''
'''------------------------------------------------------------------------------------------------------'''

def f(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return sum( (y - dot(theta.T,x)) ** 2)

def df(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return -2*sum((y-dot(theta.T, x))*x, 1)

def dfm(x, y, theta):
    x = vstack( (ones((1, x.shape[1])), x))
    return 2*dot(x, (dot(theta.T,x)-y.T).T)

def grad_descent(f, df, x, y, init_t, alpha, max_i, mult=0):
    EPS = 1e-5   #EPS = 10**(-5)
    prev_t = init_t-10*EPS
    t = init_t.copy()
    max_iter = max_i
    iter  = 0 
    while norm(t - prev_t) >  EPS and iter < max_iter:
        prev_t = t.copy()
        if mult:
            t -= alpha*dfm(x, y, t)
            if iter % 500 == 0:
                print "Iter", iter
                print "df = ", alpha*dfm(x, y, t)
                print "t = ", t 
                print "Gradient: ", dfm(x, y, t), "\n"
            iter += 1
        else:
            t -= alpha*df(x, y, t).reshape(1025, 1)
            if iter % 500 == 0:
                print "Iter", iter
                print "df = ", alpha*df(x, y, t)
                print "t = ", t 
                print "Gradient: ", df(x, y, t), "\n"
            iter += 1
        
    return t

def classify(train, train_y, alpha, max_i):
    theta = np.random.rand(1025, 1)*(1E-9)
    t = grad_descent(f, df, train.T, train_y, theta, alpha, max_i)
    return t

def mclassify(train, train_y, alpha, max_i):
    theta = np.random.rand(1025, 6)*(1E-9)
    t = grad_descent(f, dfm, train.T, train_y, theta, alpha, max_i, 1)
    return t

def measure_accuracy(x, y, theta, size, mult=0):
    if mult == 1:
        h = dot(theta.T, vstack( (ones((1, x.shape[1])), x))).T
        accuracy = 0

        for i in range(len(y)):
            if h[i].argmax() == y[i].argmax():
                accuracy+=1
        return accuracy/float(size)
    else:
        h = dot(theta.T, vstack( (ones((1, x.shape[1])), x)))
        given = []
        
        for i in range(len(y[0])):
            if h[:,i] > 0.5:
                given.append(1)
            else:
                given.append(0)
            
        accuracy = 0
        for i in range(len(y[0])):
            if y[0][i] == given[i]:
                accuracy+=1
        return accuracy/float(size) 

def gradient_check(x, y, theta):
    sample_point = [(1, 5), (765, 0)]
    
    for point in sample_point:        
        h = np.zeros([1025, 6])
        h[point[0],point[1]] = 0.000001
        
        t1 = theta+h
        t2 = theta-h
        
        x = x.T.reshape(1024, 1)
        y = y.reshape(1,6)
        
        fd = ( sum(np.square((y - dot(t1.T,vstack( (ones((1, x.shape[1])), x))).T))) - sum(np.square((y - dot(t2.T,vstack( (ones((1, x.shape[1])), x))).T))) ) /(0.000002)
        df_result =  dfm(x, y, theta)[point[0], point[1]]
        
        if round(fd, 7) != round(df_result, 7):
            print "Error: point value is different."
            break
    
        print "Gradient value is the same:"
        print "At point " + str(point)
        print "Finite Difference: " +  str(round(fd, 7)) 
        print "Gradient Descent: " + str(round(df_result, 7))    
    
'''------------------------------------------------------------------------------------------------------'''
'''                                          Execution                                                   '''
'''------------------------------------------------------------------------------------------------------'''

if __name__ == "__main__":
    
    print "----------------Preparing folders----------------"
    
    #Create uncropped folders
    if not(os.path.exists("uncropped/")):
        os.makedirs("uncropped/")
        if not(os.path.exists("uncropped/female")):
            os.makedirs("uncropped/female")
        if not(os.path.exists("uncropped/male/")):
            os.makedirs("uncropped/male/")
            
    #Create copped folders
    if not os.path.exists("cropped/"):
        os.makedirs("cropped/")
        if not(os.path.exists("cropped/part1/")):
            os.makedirs("cropped/part1")
        if not(os.path.exists("cropped/part5_act")):
            os.makedirs("cropped/part5_act")
        if not(os.path.exists("cropped/part5_acttest")):
            os.makedirs("cropped/part5_acttest")

    print " "
    
    print "----------------Folders Prepared----------------"
    
    print " "
    
    print "----------------Executing Part 1: Getting Actor Images----------------"
    get_data(test, "cropped/part1/")
    get_data(subset_act, "cropped/part5_act/")
    get_data(act_test_subset, "cropped/part5_acttest/")
    
    print "----------------Part 1 Finished: Images Fetched----------------"
    
    print " "
    
    print "----------------Executing Part 2: Creating Sets----------------"
    image_data, labels, count = get_image_data(test, "cropped/part1/")
    image_data /=225. 
    
    train, train_y, valid, valid_y, test, test_y = get_sets(image_data, labels, count, 100, 10, 10)
    
    print "----------------Part 2 Finished: Sets Created----------------"
    
    print " "
    
    print "----------------Executing Part 3: Distinguishing Hader and Carell----------------"
    
    train_y = reshape(train_y.T[1], (1, len(train)))
    valid_y = reshape(valid_y.T[1], (1, len(valid)))
    test_y = reshape(test_y.T[1], (1, len(test)))
    
    theta = classify(train, train_y, 5E-6, 30000)

    print("Training Set Performance: %", measure_accuracy(train.T, train_y, theta, len(train))*100)
    print("Validation Set Performance: %", measure_accuracy(valid.T, valid_y, theta, len(valid))*100)
    print("Test Set Performance: %", measure_accuracy(test.T, test_y, theta, len(test))*100)
    
    print "----------------Part 3 Finished----------------"
    
    print "----------------Executing Part 4: Presenting Thetas----------------"
    
    #Theta for the full dataset
    plt.figure()
    plt.title("Full Dataset")
    plt.imshow(np.resize(theta[1:], (32,32)), interpolation='nearest')
    show()
    plt.savefig('theta.png')

    #Theta for two samples
    train2, train_y2, valid2, valid_y2, test2, test_y2 = get_sets(image_data, labels, count, 2, 10, 10)
    train_y2 = reshape(train_y2.T[1], (1, len(train2)))
    theta2 = classify(train2, train_y2, 5E-6, 30000)
    
    plt.figure()
    plt.title("Two Samples")
    plt.imshow(np.resize(theta2[1:], (32,32)), interpolation='nearest')
    show()
    plt.savefig('theta2.png')
    
    print "----------------Part 4 Finished----------------"
    
    print "----------------Executing Part 5: Overfitting----------------"

    perform = []
    samples = [5, 10, 25, 50, 100]

    act_train_per = []
    act_valid_per = []
    
    for set in samples:
        #For act
        subset_male = subset_act[2:]
        subset_female = subset_act[:2]
        image_data_male, labels_male, count_male = get_image_data_gender(subset_male, "m", "cropped/part5_act/")
        image_data_male /=225. 
            
        image_data_female, labels_female, count_female = get_image_data_gender(subset_female, "f", "cropped/part5_act/")
        image_data_female /=225. 
            
        act_count = np.append(count_male, count_female)
        
        act_data = []
        for i in image_data_male:
            act_data.extend([i])
            
        for i in image_data_female:
            act_data.extend([i])
        
        act_labels = []
        for i in labels_male:
            act_labels.extend([i])
        
        for i in labels_female:
            act_labels.extend([i])
        
        act_train, act_train_y, act_valid, act_valid_y, act_test, act_test_y = get_sets(act_data, act_labels, act_count, set, 10, 10)
        
        act_train_y = reshape(act_train_y.T[1], (1, len(act_train_y)))
        act_valid_y = reshape(act_valid_y.T[1], (1, len(act_valid)))
        act_test_y = reshape(act_test_y.T[1], (1, len(act_test)))
        
        act_theta = classify(act_train, act_train_y, 5E-7, 30000)
        
        act_train_perm = measure_accuracy(act_train.T, act_train_y, act_theta, len(act_train))
        act_valid_perm = measure_accuracy(act_valid.T, act_valid_y, act_theta, len(act_valid))
        
        print("Training Set Performance: %", act_train_perm*100)
        print("Validation Set Performance: %", act_valid_perm*100)
        print("Test Set Performance: %", measure_accuracy(act_test.T, act_test_y, act_theta, len(act_test))*100)
        
        act_train_per.append(act_train_perm) 
        act_valid_per.append(act_valid_perm)
        
    perform.append(act_train_per)
    perform.append(act_valid_per)
    
    plt.figure()
    plt.title("Performance vs Size")
    plt.xlabel("Size of Training Set")
    plt.ylabel("Performance Accuracy")
    plt.plot(samples, perform[1], label="Validation Set")
    plt.plot(samples, perform[0], label="Training Set")
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.show()
    plt.savefig('performance_plot.png')
    
    #For act_test
    subset_male2 = act_test_subset[:2]
    subset_female2 = act_test_subset[2:]
    
    image_data_male2, labels_male2, count_male2 = get_image_data_gender(subset_male2, "m", "cropped/part5_acttest/")
    image_data_male2 /=225. 
    
    image_data_female2, labels_female2, count_female2 = get_image_data_gender(subset_female2, "f", "cropped/part5_acttest/")
    image_data_female2 /=225. 
        
    act_count2 = np.append(count_male2, count_female2)
    
    act_data2 = []
    for i in image_data_male2:
        act_data2.extend([i])
        
    for i in image_data_female2:
        act_data2.extend([i])
    
    act_labels2 = []
    for i in labels_male2:
        act_labels2.extend([i])
    
    for i in labels_female2:
        act_labels2.extend([i])
    
    test_act_train, test_act_train_y, test_act_valid, test_act_valid_y, test_act_test, test_act_test_y = get_sets(act_data2, act_labels2, act_count2, 100, 10, 10)
    
    test_act_train_y = reshape(test_act_train_y.T[1], (1, len(test_act_train_y)))
    test_act_valid_y = reshape(test_act_valid_y.T[1], (1, len(test_act_valid)))
    test_act_test_y = reshape(test_act_test_y.T[1], (1, len(test_act_test)))
    
    test_act_theta = classify(test_act_train, test_act_train_y, 5E-7, 30000)
    
    print("Training Set Performance: %", measure_accuracy(test_act_train.T, test_act_train_y, test_act_theta, len(test_act_train))*100)
    print("Validation Set Performance: %", measure_accuracy(test_act_valid.T, test_act_valid_y, test_act_theta, len(test_act_valid))*100)
    print("Test Set Performance: %",measure_accuracy(test_act_test.T, test_act_test_y, test_act_theta, len(test_act_test))*100)
    
    print "----------------Part 5 Finished----------------"
    
    print " "
    
    print "----------------Part 7: Six actors----------------"
    
    single_image_data, single_labels, single_image_count = get_image_data(subset_act, "cropped/part5_act/")
    single_image_data /=250.
    
    single_train, single_train_y, single_valid, single_valid_y, single_test, single_test_y = get_sets(single_image_data, single_labels, single_image_count, 100, 10, 10)

    single_theta = mclassify(single_train, single_train_y, 1E-6, 5000)
    
    gradient_check(single_train[1], single_labels[1], single_theta)

    print("Training Set Performance: %",measure_accuracy(single_train.T, single_train_y, single_theta, len(single_train), 1)*100)
    print("Validation Set Performance: %", measure_accuracy(single_valid.T, single_valid_y, single_theta, len(single_valid), 1)*100)
    print("Test Set Performance: %",measure_accuracy(single_test.T, single_test_y, single_theta, len(single_test), 1)
*100)
    
    print "----------------Part 7 Finished----------------"
    
    print " "
    
    print "----------------Part 8: Presenting Six Actors Thetas----------------"
    plt.figure()
    for i in range(single_theta.shape[1]):
        plt.subplot(2, 3, i+1)
        plt.title(subset_act[i])
        plt.imshow(np.resize(single_theta[1:,i], (32,32)), interpolation='nearest')
        show()
    plt.savefig("all.png")
    
    print "----------------Part 8 Finished----------------"
    