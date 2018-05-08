function [omega, avg_J] = trainNN(list_id, X, y, T, e, quiet=false)
    % load constants
    source globals.m;

    m = size(X,1);
    n_features = size(X,2);
    n_lists = size(unique(list_id),1);
    
    prot_idx = ( X(:,PROT_COL)==PROT_ATTR ); 
    
    % linear neural network parameter initialization
    %omega = rand(n_features,1)*INIT_VAR;
    omega = [0.0069304; 0.0084614];

    for t = 1:T
        if quiet == false
            fprintf("iteration %d: ", t)
        end
        
        % forward propagation
        z =  X * omega;
        
        % cost
        if quiet == false
            fprintf("computing cost... ")
        end

        % with regularization
        cost = listwise_cost(y,z, list_id, prot_idx);
        fprintf("cost=%f\n", sum(cost));
        %fprintf("cost: %f\n", J(1));
        J = cost + ((z.*z)'.*LAMBDA);
        % without regularization
        %J = listwise_cost(y,z, list_id, prot_idx);
        
        % gradient
        if quiet == false
            fprintf("computing gradient...\n")
        end

        grad = listnet_gradient(X, y, z, list_id, prot_idx);
        fprintf("grad: %f\n", grad(1));

        % parameter update
        omega = omega - (e .* sum(grad',2));
        %fprintf("summation of gradient: %f\n", sum(grad',2));

        
        if quiet == false
            fprintf("\n")
        end
    end
end

