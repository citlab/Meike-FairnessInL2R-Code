function plot_rankings_positions_only(data, output_file, k, title1)
  global PROT_COL
  global PROT_ATTR
  
  hf = figure ();
  output_file_with_extension = strcat(output_file, ".png");
  
  % Generate dummy info for plot handles "h"
  h = zeros(2,1);
  h(1,1) = plot(1,1,'ro', 'markersize', 5, 'markerfacecolor', 'r');hold on;
  h(2,1) = plot(1,1,'b+', 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);
  
  for i = 1:2:k
    if (data(i, PROT_COL) == PROT_ATTR)
      %plot(i, data(i, plot_col), 'ro');
      plot(i, i, 'ro', 'markersize', 5, 'markerfacecolor', 'r');
    else 
      %plot(i, data(i, plot_col), 'bo');
      plot(i, i, 'b+', 'markersize', 5, 'markerfacecolor', 'b', 'LineWidth', 2);
    end
  end
  hold off;
  set(gca, "ydir", "reverse") 
  set(gca, 'xtick', [])
  legend(h, 'protected', 'non-protected');
  
  ylabel ("ranking position");
  title (title1);
  print(hf, output_file_with_extension);
  %print (hf, output_file_with_extension, "-dpng");
  %system (sprintf("pdflatex %s", output_file));
  open(output_file_with_extension);
end