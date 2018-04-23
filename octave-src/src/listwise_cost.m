function J = listwise_cost(y, z, list_id, prot_idx)
    global CORES
    ly = @(i) y(find(list_id == list_id(i)),:);
    lz = @(i) z(find(list_id == list_id(i)),:);
    
    lz_prot = @(preds, prot_idx) preds(prot_idx);
    lz_nprot = @(preds, prot_idx) preds(!prot_idx);
    
    % Exposure in Rankings for the protected and non-protected group
    exposure_prot = @(i) sum(topp_prot(lz_prot(lz(i), prot_idx), lz(i)) ./ log(2)) / size(lz_prot(lz(i), prot_idx), 1); 
    exposure_nprot = @(i) sum(topp_prot(lz_nprot(lz(i), prot_idx), lz(i)) ./ log(2)) / size(lz_nprot(lz(i), prot_idx), 1); 
    
    % no nead for element-wise operation, because this is supposed to be a number
    exposure = @(i) (exposure_prot(i) - exposure_nprot(i))^2
    accuracy = @(i) (-sum(topp(ly(i)) .* log( topp(lz(i)) )));
    
    j = @(i) GAMMA * exposure(i) + accuracy(i)
    J = pararrayfun(CORES, j,1:size(z,1), "VerboseLevel", 0);
end
