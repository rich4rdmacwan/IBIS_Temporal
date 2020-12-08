function [] = analyseCasme(varargin)
%analyseCasme Iterate through Casme db videos and analyse the .seg files
%from IBIS_Temporal
%   Detailed explanation goes here
tracesFolder = '/mnt/nvmedisk/CasmeSq/traces_500_50/';
vidFolder = '/mnt/nvmedisk/CasmeSq/rawvideo/';
METHOD_SrPDE = 0; % SP=0 or SrPDE=1
close all;
while nargin >1
    i=1;
    if strcmp(varargin{i},'tracesFolder')==1
        tracesFolder = varargin{i+1};
    end
    if strcmp(varargin{i},'vidFolder')==1
        vidFolder = varargin{i+1};
    end
    if(strcmp(varargin{i},'WINDOW_LENGTH_SEC')==1)
        WINDOW_LENGTH_SEC=varargin{i+1};
    end
    if(strcmp(varargin{i},'STEP_SEC')==1)
        STEP_SEC=varargin{i+1};
    end
    if strcmp(varargin{i},'method')==1
        if strcmp(varargin{i+1},'SP')
            METHOD_SrPDE = 0;
        elseif strcmp(varargin{i+1},'SrPDE')
            METHOD_SrPDE = 1;
        end
    end
    i = i+2;
end

%% Face Landmarks init
%setenv('LD_LIBRARY_PATH', ['/usr/local/MATLAB/R2018a/extern/bin/glnxa64:' getenv('LD_LIBRARY_PATH') ]);
%Face Landmark detection init
%Might need prepend this to matlab to get dlib to work:
%LD_PRELOAD=/usr/lib/libstdc++.so.6:/usr/lib/liblapack.so:/usr/lib/libblas.so LD_LIBRARY_PATH=/usr/lib:/usr/local/Polyspace/R2020a/extern/bin/glnxa64:/usr/local/Polyspace/R2020a/bin/glnxa64:/usr/local/lib:
% Mismatch in the blas and lapack libraries being used

addpath('/usr/local/interfaces/matlab/');
%setenv('LD_LIBRARY_PATH', '');
useFaceLandmarks = true;
modelFile='/home/richard/src/IBIS_Temporal/shape_predictor_68_face_landmarks.dat';
pointTracker = vision.PointTracker;

%% Expressions GT for CasmeSq
load([vidFolder '../codefinal.mat']);
%Function handle to extract expression name from second column of
%tcodefinal (e.g. anger1_1, anger1_2, etc.). Shall be used in cellfun, due
%to the tabular structure of the ground truth
cellsubstr=@(str) str(1:end-2);


%% Access the CasmeSq directory structure
dirs = dir([tracesFolder 'rawvideo']);
%Iterate through each dir. Start from 3 to skip . and .. listing
for didx=3:numel(dirs)
    %We have subject directories here: s15, s16,etc.
    viddir = dir([dirs(didx).folder '/' dirs(didx).name]);
    %We have the video directories here, we can access the .seg files now
    
    %Find all the expressions for this subject. Each subject has multiple
    %videos
    sid = str2num(dirs(didx).name(2:end));
    all_expr_gt = tcodefinal(tnaming_rule1{:,1}==sid,:);
    
    for vidx=4:numel(viddir)
        %Read expressions
        disp(viddir(vidx).name);
        %Get expression list for current video from all_exp_gt
        videocode=viddir(vidx).name(4:7);
        %Get expression name from videocode
        li = ismember(tnaming_rule2{:,1},videocode);
        expr_name = tnaming_rule2{li,2}{:};
        %Use this expr_name to filter all_exp_gt
        all_exprs = cellfun(cellsubstr,all_expr_gt{:,2},'UniformOutput',false);
        filter = ismember(all_exprs,expr_name);
        gtrows = all_expr_gt(filter,:);
        %gtrows contains all expressions for this video
        expr_frame_indices = gtrows{:,3:5}; %int32 matrix
        expr_start = 0;
        %TODO: Add a summary of expressions somewhere in the window
        
        %Read original video
        vid = VideoReader([vidFolder dirs(didx).name '/' viddir(vidx).name ]);
        %Read labels video
        %lblVid = VideoReader([viddir(j).folder '/' viddir(j).name '/labels.avi']);
        vidPlayer = vision.VideoPlayer;
        STEP_SEC = .2;
        Fs=30; %Casme2 database fps.
        %winLength = pow2(floor(log2(WINDOW_LENGTH_SEC*Fs))); % length of the sliding window for FFT
        winLength = 300; % %This is the n° of frames used in ibis for HR calculation. Can be changed
        
        data = struct; %Structure to encompass all processing and data
        
        %% Read SP contours
        if ~METHOD_SrPDE
            data.segfiles = dir([viddir(vidx).folder '/' viddir(vidx).name '/traces*.seg']);
            data.nFrames = numel(data.segfiles);
            
            fid=fopen([viddir(vidx).folder '/' viddir(vidx).name  '/contours.seg']);
            %Each block of values separated by ',' represents the indices of
            %contours in the 640x480 image
            disp('Reading contour indices for all frames...');
            data.mask=textscan(fid,'%s','delimiter',','); %Heavy operation
            data.mask=data.mask{1};
            fclose(fid);
            disp('Reading labels for all frames...');
            fid=fopen([viddir(vidx).folder '/' viddir(vidx).name  '/labels_px.seg']);
            data.all_sp_labels = textscan(fid,'%s','delimiter',';');
            data.all_sp_labels = data.all_sp_labels{1};
            fclose(fid);
        end
        
        %% SrPDE variables
        traceSize = int32(vid.FrameRate*vid.Duration);
        %Sanity check:
        if data.nFrames ~=traceSize
            error('Trace size does not match the number of frames in the video!');
        end
        vidDuration = double(traceSize/Fs);
        timeTrace = linspace(0,vidDuration,traceSize);
        pulseTrace = zeros(1,traceSize);
        maxLags = round(1.5*Fs);
        stepSize = round(STEP_SEC*Fs);
        maxLags = 60/40; %Corresponding to 40bpm
        halfWin = (winLength/2);
        
        
        %for i = 0:nFrames-1
        ind = 1; %Temporal window index
        for i=halfWin:stepSize:traceSize-halfWin-maxLags
            fprintf('Pass %u of %.0f\n',ind,numel(halfWin:stepSize:traceSize-halfWin-maxLags));
            if(ind==1)
                fprintf('Accumulating %d frames for the first window\n',winLength);
            end
            startInd = i-halfWin+1;
            endInd = i+halfWin;
            
            %% TODO for SP
            % 1. Apply facial landmarks to extract face
            % 2. Find SPs encompassing the face
            % 3. Use RGB traces from those SPs to perform SrPDE
            %    Before applying SrPDE, we need X and Xt traces of size
            %    3xnFramesxnSP (nSP is number of superpixels)
            %Get indices of the seed centroids
            %xs=data(4,:); ys=data(5,:);
            %inds=sub2ind([640,480],round(xs),round(ys));
            
            %% Expressions groundtruth visualisation
            %Check if current frame is in exp_frame_indices
            is_expr_frame =find(expr_frame_indices(:,1)==startInd);
            if is_expr_frame
                expr_start = expr_frame_indices(is_expr_frame,1);
                expr_peak = expr_frame_indices(is_expr_frame,3);
                expr_end = expr_frame_indices(is_expr_frame,3);
            end
            %If we are in the middle of an expression, check if we have
            %reached the apex or end frames
            if expr_start~=0
                if startInd == expr_end
                    %Reset variables
                    expr_start =0; expr_peak = 0; expr_end = 0;
                else
                    %TODO:
                    %Logic for displaying expressions text in the plots
                    %Display a text annotation expr_start to expr_end frame
                    %Add gradient with a peak at expr_peak frame
                end
                
            end
            
            
            %% Accumulate frames for the first window
            if ind == 1 %First temporal window
                % Accumulate frames into tensor upto endInd-1
                for j = startInd:endInd
                    %Read rgb frame from video
                    data.frame = readFrame(vid);
                    %Landmarks and tracker
                    if useFaceLandmarks && j==1
                        % Find face landmarks and tracker
                        lmd = find_face_landmarks(modelFile, data.frame);
                        Xs = double(lmd.faces.landmarks(:,1)');
                        Ys = double(lmd.faces.landmarks(:,2)');
                        %Forehead
                        minX = min(Xs); minY = min(Ys);
                        maxX = max(Xs); maxY = max(Ys);
                        foreheadExtent = (maxY - minY)/4;
                        % Bounds to include the forehead. [x;y]
                        bounds = [ minX, minX+(maxX-minX)/5, ...
                            Xs(28), maxX-(maxX-minX)/5,...
                            maxX; minY,foreheadExtent,foreheadExtent-20,foreheadExtent,minY];
                        
                        Xs = [Xs, bounds(1,:)]; Ys = [Ys, bounds(2,:)];
                        %Convex hull
                        k = convhull(Xs,Ys);
                        data.face_landmarks=[Xs(k)',Ys(k)'];
                        initialize(pointTracker,data.face_landmarks,data.frame);
                 
                    elseif useFaceLandmarks
                        [data.face_landmarks, validity] = pointTracker(data.frame);
                        skinmask = roipoly(data.frame,data.face_landmarks(:,1), data.face_landmarks(:,2));
                    end
                    %inCentroids = inpolygon(cX,cY,face_landmarks(:,1),face_landmarks(:,2));
                    %% SP
                    if ~METHOD_SrPDE
                        [data] = stepSP(didx,j,data);
                        if(j==data.nFrames-1)
                            %Save traces
                            save([segfiles(didx).folder '/traces' ],'SP.traces');
                        end
                    end
                end
            else
                %Temporal windows 2 to last. Read frames worth step seconds and update the structures
                for j = endInd-stepSize+1:endInd
                    data.frame = readFrame(vid);
                    if ~METHOD_SrPDE
                        [data] = stepSP(didx,j,data);
                    end
                end
            end
            ind = ind + 1;
        end
        
        
    end
    
end
end


function [SP] ...
    = stepSP(didx, j, SP)
% Read SP trace values for this frame
fname = sprintf('%s/traces_%04d.seg', SP.segfiles(didx).folder,j);
fileID = fopen(fname,'r');
spdata = fscanf(fileID,'%f %f %f %f %f %f',[6 Inf]);
fclose(fileID);

if ~exist('SP.SPNumber','var')
    SP.SPNumber = size(spdata,2);
    SP.traces = zeros(3,SP.SPNumber,SP.nFrames);
end
%Get the contour indices
cntr_idx = sscanf(SP.mask{j},'%d');
%[cntr_idx_x, cntr_idx_y] = ind2sub([640,480],cntr_idx);
bmask = zeros(640,480);

%Turn on contour indices
bmask(cntr_idx) = 1;
%imagesc(bmask');

%Overlat contours on frame
contourI=labeloverlay(SP.frame,bmask');
SP.traces(:,:,j) = spdata(1:3,:);

cX = round(spdata(4,:)); cY = round(spdata(5,:));
% Calculate centroids only for the first frame, and then reuse
if j==1
    % Find SPs falling closest to and inside 'face_landmarks'
    
    
    [dilatedLMX, dilatedLMY] = dilatePoints(SP.face_landmarks(:,1),SP.face_landmarks(:,2),SP.SPNumber,SP.frame);
    SP.inCentroids = inpolygon(cX,cY,dilatedLMX,dilatedLMY);
    
    %Get pixels corresponding to each SP from labelframes
    sp_labels_txt = SP.all_sp_labels{j};
    lFR = lbl2img(sp_labels_txt);
    lFG = lFR; lFB = lFR;
    %             lFrame = readFrame(lblVid);
    %             lFR = lFrame(:,:,1); lFG=lFrame(:,:,2); lFB=lFrame(:,:,3);
    %
    for spi = find(SP.inCentroids)
        %lFrame has 0 indexed SPs
        lFR(lFR==spi-1) = spdata(1,spi);
        lFG(lFG==spi-1) = spdata(2,spi);
        lFB(lFB==spi-1) = spdata(3,spi);
    end
    lFrame = uint8(cat(3,lFR,lFG,lFB));
    
    subplot(2,2,1);
    SP.h221 = image(lFrame);
    title('Superpixel RGB');
    hold on;
    %hlm = plot(face_landmarks(:,1),face_landmarks(:,2),'w.','MarkerSize',10);
    SP.hlm = plot(SP.face_landmarks(:,1),SP.face_landmarks(:,2),'w.','MarkerSize',10);
    SP.hC = plot(cX(SP.inCentroids), cY(SP.inCentroids),'x');
    
    axis equal; axis tight;
    subplot(2,2,2);
    title('Superpixel contours');
    SP.h222 = image(contourI);
    axis equal; axis tight;
else
    %Get pixels corresponding to each SP from labelframes
    sp_labels_txt = SP.all_sp_labels{j};
    lFR = lbl2img(sp_labels_txt);
    lFG = lFR; lFB = lFR;
    %             lFrame = readFrame(lblVid);
    %             lFR = lFrame(:,:,1); lFG=lFrame(:,:,2); lFB=lFrame(:,:,3);
    %
    for spi = find(SP.inCentroids)
        %lFrame has 0 indexed SPs
        lFR(lFR==spi-1) = spdata(1,spi);
        lFG(lFG==spi-1) = spdata(2,spi);
        lFB(lFB==spi-1) = spdata(3,spi);
    end
    lFrame = uint8(cat(3,lFR,lFG,lFB));
    
    set(SP.h221,'Cdata',lFrame);
    set(SP.h222,'Cdata',contourI);
    set(SP.hlm,'Xdata',SP.face_landmarks(:,1),'Ydata',SP.face_landmarks(:,2));
    set(SP.hC,'Xdata',cX(SP.inCentroids),'Ydata',cY(SP.inCentroids));
end
title(j);
pause(.01);
%montage({lFrame,contourI});
%Find SP with best SNR
[SP.topsp, SP.topspi] = max(spdata(6,:));



end