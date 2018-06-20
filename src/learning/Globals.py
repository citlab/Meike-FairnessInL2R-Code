#global variables declaration

#number of training iterations
global T
T = 1000

#learning rate
global e
e = 0.00001

#regularization constant
global LAMDA
LAMDA = 0.001

#number of cores for parallel processing
global CORES
CORES = 8

#range of values for initialization of weights
global INIT_VAR
INIT_VAR = 0.01

global ONLY_U
ONLY_U = 0

global ONLY_L
ONLY_L = 0

global L_AND_U
L_AND_U = 1

#index of column that contains protected attribute
global PROT_COL
PROT_COL = 1

#define what is the protected attribute
global PROT_ATTR
PROT_ATTR = 1

#debug
global DEBUG
DEBUG = 0
global DEBUG_PRINT
DEBUG_PRINT = 0

