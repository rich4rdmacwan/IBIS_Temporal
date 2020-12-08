% Read the label string into two arrays. The first array is the sp_label.
% The second array is the index uptil which the previous Sp was repeated
% e.g. 0 50, 1, 90, 2, 120 means the label 0 repeats uptil index 50, 1
% repeats uptil index 90, etc.
% Both the indices and labels are zero-index based
function [sp_labels, cum_indices]=str2double_label(label)
    arr = sscanf(label,'%d %d,',[2,Inf]);
    sp_labels=arr(1,:);
    cum_indices =arr(2,:);
end

