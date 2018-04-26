function J = listwise_cost(y, z, list_id, prot_idx)
    global CORES
    global GAMMA
    global DEBUG
    
    ly = @(i) y(find(list_id == list_id(i)),:);
    lz = @(i) z(find(list_id == list_id(i)),:);
    
    % get idx of protected candidates per query, otherwise dimensions don't fit
    prot_idx_per_query = @(i) prot_idx(find(list_id == list_id(i)),:);
    
    l_prot_vec = @(preds, idx) preds(idx);
    
    group_size_p = @(i) size(l_prot_vec(lz(i), prot_idx_per_query(i)), 1);
    group_size_np = @(i) size(l_prot_vec(lz(i), !prot_idx_per_query(i)), 1);
 
    % Exposure in Rankings for the protected and non-protected group
    exposure_prot = @(i) sum(topp_prot(l_prot_vec(lz(i), prot_idx_per_query(i)), lz(i)) ./ log(2)); 
    exposure_prot_normalized = @(i) exposure_prot(i) / group_size_p(i); 
    
    exposure_nprot = @(i) sum(topp_prot(l_prot_vec(lz(i), !prot_idx_per_query(i)), lz(i)) ./ log(2));
    exposure_nprot_normalized = @(i) exposure_nprot(i) / group_size_np(i); 
   
    % calculate difference of exposure between the two groups
    exposure_diff = @(i) (exposure_prot_normalized(i) - exposure_nprot_normalized(i))^2;
    
    % calculate accuracy wrt training data
    accuracy = @(i) (-sum(topp(ly(i)) .* log( topp(lz(i)) )));
    
    % make sure exposure is not NaN, but 0 instead 
    % can be NaN if either protected or non-protected group has size 0
    exposure_non_nan = @(i) max(exposure_diff(i), 0);
    
    if DEBUG
      iter = 1:size(z,1)
      idx = prot_idx_per_query(iter);
      z_prot = l_prot_vec(lz(iter), prot_idx_per_query(iter));
      top1_prot = topp_prot(l_prot_vec(lz(iter), prot_idx_per_query(iter)), lz(iter));
      top1_prot_times_v = topp_prot(l_prot_vec(lz(iter), prot_idx_per_query(iter)), lz(iter)) ./ log(2);
      
      group_size_prot = group_size_p(iter);
      group_size_nprot = group_size_np(iter);
      
      exposure_p = exposure_prot(iter);
      exposure_p_norm = exposure_prot_normalized(iter);
      
      exposure_np = exposure_nprot(iter);
      exposure_np_norm = exposure_nprot_normalized(iter);
      
      exposure2 = (exposure_p_norm - exposure_np_norm)^2;
      accuracy2 = accuracy(iter);
      
      exposure_not_nan = exposure_non_nan(iter);
      cost = GAMMA * exposure_non_nan(iter) .+ accuracy(iter);
    end
    
    j = @(i) GAMMA * exposure_non_nan(i) .+ accuracy(i);
    J = pararrayfun(CORES, j,1:size(z,1), "VerboseLevel", 0);
end