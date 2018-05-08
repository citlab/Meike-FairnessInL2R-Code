function J = listwise_cost(y,z, list_id)
    ly = @(i) y(find(list_id == list_id(i)),:);
    lz = @(i) z(find(list_id == list_id(i)),:);
    
    j = @(i) (-sum(topp(ly(i)) .* log( topp(lz(i)) )));
 


    J = arrayfun(j,1:size(z,1));
    
    fprintf("listwise_cost(1) = %f\n", j(1));
    fprintf("listwise_cost(2) = %f\n", j(2));
    fprintf("listwise_cost(10) = %f\n", j(10));
    fprintf("listwise_cost(16) = %f\n", j(16));
    fprintf("listwise_cost(17) = %f\n", j(17));
    fprintf("len(listwise_cost) = %d\n", size(J,2));
end
