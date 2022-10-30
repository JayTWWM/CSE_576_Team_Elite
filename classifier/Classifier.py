'''
This file contains code for avengers ensemble.

'''
# importing libraries
import pickle
import sys
from sklearn.svm import SVC
import os 

# Setting the parameter
try:
    database_name = str(sys.argv[0])
    required_authors = str(sys.argv[1])
except:
    database_name, required_authors= 'amt', 5

# main function
if __name__ == '__main__':
    
