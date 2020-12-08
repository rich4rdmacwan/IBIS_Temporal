% Convert the coded array into an image
% sp_labels contains the labels and the cum_indices represents the indices
% of changing labels-1.
% e.g. 000011111222 is represented as [0 1 2],[3 8 11]
% Optional argument: 'size' - 2 element vector to indicate output image
% size [width height]
function img=lbl2img(labels,varargin)
    [sp_labels,cum_indices] = str2double_label(labels);
    sz=[640, 480];
    %Optional size argument
    i=1;
    
    if nargin>2
        if nargin<4 
            error('lbl2img needs key value pairs for optional arguments');
        end
        while i+2<nargin
            if strcmp(varargin{i},'size')==1
                sz=varargin{i+1};
            end
        end
    end
    
    diffs =[cum_indices(1)+1 diff(cum_indices)];
    img = repelem(sp_labels,diffs);
    img = reshape(img,sz);
    img = img';    
end