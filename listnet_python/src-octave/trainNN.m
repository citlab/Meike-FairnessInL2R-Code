function [omega, avg_J] = trainNN(list_id, X, y, T, e, quiet=false)
    % load constants
    source global.m;

    m = size(X,1);
    n_features = size(X,2);
    n_lists = size(unique(list_id),1);

    % linear neural network parameter initialization
    omega = rand(n_features,1)*INIT_VAR;

    % STATIC OMEGA initial
    omega(1) = 0.00537264;
    omega(2) = 0.00471433;
    omega(3) = 0.00205609;
    omega(4) = 0.00398706;
    omega(5) = 0.0074851;
    omega(6) = 0.00637906;
    omega(7) = 0.00903121;
    omega(8) = 0.00930291;
    omega(9) = 0.00883843;
    omega(10) = 0.0048150;

    omega
    size(omega)
    % /STATIC

    last_cost = -1

    for t = 1:T
        if quiet == false
            fprintf("\niteration %d: ", t)
        end
        
        % forward propagation
        z =  X * omega;
         
        % cost
        if quiet == false
            fprintf("computing cost...\n")
        end

        % with regularization
	J_listwise_cost = listwise_cost(y, z, list_id);
	J_regularizer_cost = ((z.*z)'.*LAMBDA);
        J = J_listwise_cost + J_regularizer_cost;
	fprintf(" cost_listwise=%f, cost_regularizer=%f, ", sum(J_listwise_cost), sum(J_regularizer_cost) );
        % without regularization
        %J = listwise_cost(y,z, list_id);
	fprintf('cost_total=%f ', sum(J));
	if last_cost != -1
		fprintf('delta cost=%f ', sum(J) - last_cost);
	end
	last_cost = sum(J);
        
        % gradient
        if quiet == false
            fprintf("\n\ncomputing gradient...")
        end

        grad = listnet_gradient(X, y, z, list_id);
        
        % parameter update
        omega = omega - (e .* sum(grad',2));
        
        if quiet == false
            fprintf("\n")
        end
    end
end

