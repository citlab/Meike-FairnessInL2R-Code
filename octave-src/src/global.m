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
% TODO: IN COST FUNCTION VALUES FOR EXPOSURE AND COST DIFFERENTIATE BY 12 ORDERS OF MAGNITUDE!! WHAT TO DO ABOUT THAT?
% IN GRADIENT FUNCTION THEY ONLY DIFFER BY 6 ORERS OF MAGNITUDE...WHAT NOW?
global GAMMA = 1;

% index of column that contains protected attribute
global PROT_COL = 1;

% wanna debug?
global DEBUG = 0;
