function grad = listnet_gradient (X, y, z, list_id, prot_idx)
    % prot_idx : array for logical indexing that tells in which lines of the data are
    %            protected items 
    global CORES
    global GAMMA
    global DEBUG
    global DEBUG_PRINT
    global ONLY_L
    global ONLY_U
    global L_AND_U
    
    m = size(X,1); %number of documents
    p = size(X,2); %number of features
    
    %find all data points that belong to one query
    lx = @(i) X(find(list_id == list_id(i)),:);
    ly = @(i) y(find(list_id == list_id(i)),:);
    lz = @(i) z(find(list_id == list_id(i)),:);
    
    % get idx of protected candidates per query, otherwise dimensions don't fit
    prot_idx_per_query = @(i) prot_idx(find(list_id == list_id(i)),:);

    % returns only those lines in which the logical array idx contains 1
    l_group_vec = @(preds, idx) preds(idx);
    l_group_mat = @(data, idx) data(idx, :);
    
    % Derivative for topp_prot in pieces:    
    % t = feature vector of (non-) protected group
    % u = predicted scores for (non-) protected group
    % v = predictions of all data points
    % w = feature vectors of all data points    
    tp1 = @(t, u) t .* repmat(exp(u), 1, columns(t));
    tp2 = @(v) sum(exp(v));      
    tp3 = @(w, v) sum(w' * exp(v));  
    tp4 = @(v) sum(exp(v))^2;  
    
    % collect data structures
    group_features_p = @(i) l_group_mat(lx(i), prot_idx_per_query(i));
    group_preds_p = @(i) l_group_vec(lz(i), prot_idx_per_query(i));
    group_features_np = @(i) l_group_mat(lx(i), !prot_idx_per_query(i));
    group_preds_np = @(i) l_group_vec(lz(i), !prot_idx_per_query(i));
    
    % derivative calculated for protected group and non-protected
    % contains weight adjustments for each feature (each column is a feature) 
    % of each document (each row is a document) (hence a matrix)
    % in order to get overall weight adjustment vector sum up elements in each column
    tp_p = @(i) sum(-1*(tp1(group_features_p(i), group_preds_p(i)) * tp2(lz(i)) - exp(group_preds_p(i)) * tp3(lx(i), lz(i)))) / tp4(lz(i));    
    tp_np = @(i) sum(-1*(tp1(group_features_np(i), group_preds_np(i)) * tp2(lz(i)) - exp(group_preds_np(i)) * tp3(lx(i), lz(i)))) / tp4(lz(i));
        
    group_size_p = @(i) size(group_preds_p(i), 1);
    group_size_np = @(i) size(group_preds_np(i), 1);
 
    % Exposure in Rankings for the protected and non-protected group
    exposure_prot = @(i) (sum(topp_prot(group_preds_p(i), lz(i)))) ./ log(2);    
    exposure_nprot = @(i) (sum(topp_prot(group_preds_np(i), lz(i)))) ./ log(2);
   
    % normalize exposures
    exposure_prot_normalized = @(i) exposure_prot(i) / group_size_p(i); 
    exposure_nprot_normalized = @(i) exposure_nprot(i) / group_size_np(i); 
    
    if DEBUG_PRINT
      fprintf("exposure_prot_normalized: %f\n", exposure_prot_normalized(1)); 
      fprintf("exposure_nprot_normalized: %f\n", exposure_nprot_normalized(1)); 
    end
    
    % make sure u1 is not NaN
    u1 = @(i) 2 * max((exposure_nprot_normalized(i) - exposure_prot_normalized(i)), 0);
    u2 = @(i) (tp_np(i) ./ log(2)) / group_size_np(i);
    u3 = @(i) (tp_p(i) ./ log(2)) / group_size_p(i);
    U = @(i) u1(i) * (u2(i) - u3(i)); % should be vector of size n(q)
    
    l1 = @(i) -(lx(i)' * topp(ly(i)));
    l2 = @(i) (1 / sum(exp(lz(i))));
    l3 = @(i) (lx(i)' * exp(lz(i)));

    L = @(i) (l1(i) + l2(i) * l3(i)); % should be vector of size n(q)
    
    if ONLY_L
      f = @(i) L(i)';
    end
    
    if ONLY_U
      f = @(i) GAMMA * U(i);
    end 
    
    if L_AND_U
      f = @(i) GAMMA * U(i) + L(i)';
    end 
    
    if DEBUG_PRINT
      %fprintf("cost in gradient 1: %f\n", f(1));
    end
    
    if DEBUG
      iter = 1:m;

      prot_idx_q1 = prot_idx_per_query(iter);
      
      lz1 = lz(iter);
      lx1 = lx(iter);
      
      z_prot = l_group_vec(lz(iter), prot_idx_per_query(iter));
      z_nprot = l_group_vec(lz(iter), !prot_idx_per_query(iter));
      x_prot = l_group_mat(lx(iter), prot_idx_per_query(iter));
      
      exposure_p = exposure_prot(iter);
      exposure_p_norm = exposure_prot_normalized(iter);
      
      exposure_np = exposure_nprot(iter);
      exposure_np_norm = exposure_nprot_normalized(iter);
      
      tp1p = tp1(x_prot, z_prot);
      tp2p = tp2(lz(iter));
      tp3p = tp3(lx(iter), lz(iter));
      twoMinusThree = tp2(lz(iter)) - tp3(lx(iter), lz(iter));
      tp4p = tp4(lz(iter));
 
      tp_prot = tp_p(iter);
      
      u1expdiff = u1(iter);
      u2np = u2(iter);
      u3p = u3(iter);
      
      fair_w = U(iter);
      fair_w_gamma = GAMMA * U(iter);
      acc_w = L(iter);
      fval = fair_w_gamma + acc_w';
    end

    grad = reshape(pararrayfun(CORES, f, 1:m, "VerboseLevel", 0),p,m)';
end
