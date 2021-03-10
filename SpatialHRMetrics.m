function [] = SpatialHRMetrics(varargin)
%HR Metrics for spatial rPPG 
db = struct;
db.idx = 2; % CasmeSq=1, FixedData=2, MMSE=3;
close all;
db = getNextVid(db);
roisize = [128 128];
while db.res
    % Load HR metricds
    load([db.tracesFolder 'HRmetrics-' num2str(roisize(1)) 'x' num2str(roisize(2)) '.mat']);
    plot(HR_PPG,HR_rPPG,'*');
    %Get next video
    db = getNextVid(db);
end


end

