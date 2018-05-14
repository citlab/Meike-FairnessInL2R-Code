function [omega, avg_J] = trainNN(list_id, X, y, T, e, quiet=false)
    % load constants
    source globals.m;

    m = size(X,1);
    n_features = size(X,2);
    n_lists = size(unique(list_id),1);
    
    prot_idx = ( X(:,PROT_COL)==PROT_ATTR ); 
    
    % linear neural network parameter initialization
    omega = rand(n_features,1)*INIT_VAR;
    %omega = [0.0069304; 0.0084614];
    
    cost_converge_J = zeros(T, 1);
    cost_converge_L = zeros(T, 1);
    cost_converge_U = zeros(T, 1);
    omega_converge = zeros(T, n_features);

    for t = 1:T
        if quiet == false
            fprintf("iteration %d: ", t)
        end
        
        % forward propagation
        z =  X * omega;
        
        % cost
        if quiet == false
            fprintf("computing cost... \n")
        end

        % with regularization
        [cost, L, U] = listwise_cost(y,z, list_id, prot_idx);
        %fprintf("cost=%f\n", sum(cost));
        %cost_converge(t) = sum(cost);
        %fprintf("cost: %f\n", J(1));
        J = cost + ((z.*z)'.*LAMBDA);
        cost_converge_J(t) = sum(J);
        cost_converge_L(t) = sum(L);
        cost_converge_U(t) = sum(U);

        % without regularization
        %J = listwise_cost(y,z, list_id, prot_idx);
        
        % gradient
        if quiet == false
            fprintf("computing gradient...\n")
        end

        grad = listnet_gradient(X, y, z, list_id, prot_idx);
        %fprintf("grad: %f\n", grad(1));

        % parameter update
        omega = omega - (e .* sum(grad',2));
        %fprintf("omega: %f\n", omega);
        omega_converge(t, :) = omega(:);

        
        if quiet == false
            fprintf("\n")
        end
    end
    figure(); plot(cost_converge_J);
    figure(); plot(cost_converge_U);
    figure(); plot(cost_converge_L);
    figure(); plot(omega_converge);
end

