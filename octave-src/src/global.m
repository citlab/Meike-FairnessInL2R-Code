% global variables declaration

% number of training iterations
global T = 50;

% learning rate
global e = 0.00005;

% regularization constant
global LAMBDA = 0.001;

% number of cores for parallel processing
global CORES = 4;

% range of values for initialization of weights
global INIT_VAR = 0.01;

% factor to weight the importance of the fairness component in the loss function
global GAMMA = 0;

% index of column that contains protected attribute
global PROT_COL = 1;
