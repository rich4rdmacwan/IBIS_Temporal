function [] = launcherCasme_SRPDEV(varargin)
%analyseCasme Iterate through Casme db videos and analyse the .seg files
%from IBIS_Temporal
%   Varargin described as follows
% tracesFolder          : String pointing to the path of CasmeSq output traces
% folder
% vidFolder             : String pointing to the path of CasmeSq db
% WINDOW_LENGTH_SEC     : Int, Window length in seconds
% STEP_SEC              : Int, Step size in seconds
% method                : String to choose method. 'SP', 'SrPDE' or 'CHROM'
db = struct;
db.idx = 2; % CasmeSq=1, FixedData=2, MMSE=3;
method = 2; % SP=0 or SrPDE=1 or CHROM=2
SAVERESULTS=true;
close all;
i=1;
while nargin >1
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
            method = 0;
        elseif strcmp(varargin{i+1},'SrPDE')
            method = 1;
        elseif strcmp(varargin{i+1},'CHROM')
            method = 2;
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
addpath('../temporal-qz');
addpath('../ndsparse/'); %For 3d detrending
%setenv('LD_LIBRARY_PATH', '');
useFaceLandmarks = true;
modelFile='/home/richard/src/IBIS_Temporal/shape_predictor_68_face_landmarks.dat';

addpath('../mtimesx/');

computationtimes=[];

db = getNextVid(db);
while db.res
    %for vidx=6:numel(viddir)
    data = struct; %Structure to encompass all processing and data
    data.method = method;
    data.WRITEVIDEO = SAVERESULTS;
    data.pointTracker = vision.PointTracker('NumPyramidLevels',4,'MaxBidirectionalError',3,'BlockSize',[5,5]);
    
    
    data.vidName= db.nextVid;
    
    %Read original video
    vid = VideoReader(data.vidName);
    %Read labels video
    %lblVid = VideoReader([viddir(j).folder '/' viddir(j).name '/labels.avi']);
    vidPlayer = vision.VideoPlayer;
    STEP_SEC = .2;
    if db.idx==1
        data.Fs=30; %Casme2 database fps.
    elseif db.idx==2
        data.Fs = vid.NumFrames/vid.Duration;
    end
    
    %winLength = pow2(floor(log2(WINDOW_LENGTH_SEC*Fs))); % length of the sliding window for FFT
    data.winLength = 300; % %This is the n° of frames used in ibis for HR calculation. Can be changed
    traceSize = int32(vid.FrameRate*vid.Duration);
    data.stepSize = round(STEP_SEC*data.Fs);
    %data.maxLags = round(60/40*data.Fs); %Corresponding to 40bpm
    halfWin = (data.winLength/2);
    data.endGame = false; %Flag to depict that we are processing the last remaining frames
    
    data.nWindows = numel(halfWin:data.stepSize:traceSize-halfWin);
    HR_PPG = zeros(1,data.nWindows);
    
    %data.maxLags = round(1.5*Fs);
    if db.idx==1 %Casmesq
            data.vidDuration = double(traceSize/data.Fs);
        elseif db.idx==2
            data.vidDuration = vid.Duration;
    end
    
    data.frameIndex=0;
    data.skip = false;
    data.times = nan*ones(1,traceSize);
    
    %Motion compensation for pointtracker inexactitude
    data.useMotionCompensation = true;
    if method==0
        %SP
    end
    
    if method==1
        %% SrPDE processing
        if data.WRITEVIDEO
            data.videowriter = VideoWriter([db.tracesFolder   '/srpderaw.avi'],'Uncompressed AVI');
            data.videowriter.FrameRate = data.Fs;
            open(data.videowriter);
        end
        
        data.scale = 1;
        data.nFrames = traceSize;
        %Save framewise face rois for SrPDE metrics display later
        data.faceroi = zeros(traceSize,4);
    end
    if method==2
        
        %useFaceLandmarks = false;
        data.nFrames = traceSize;
        %Save framewise face rois for SrPDE metrics display later
        data.faceroi = zeros(traceSize,4);
    end
    
    %Sanity check:
    if data.nFrames ~=traceSize
        error('Trace size does not match the number of frames in the video!');
    end
    
    
    %for i = 0:nFrames-1
    ind = 1; %Temporal window index
    prevprog=-1;
    
    for i=halfWin:data.stepSize:traceSize-halfWin
        if(ind==1)
            %fprintf('\n%s',viddir(vidx).name);
            fprintf('\n %s : Accumulating %d frames for the first window (total %d)\n',db.nextVidName, data.winLength,numel(halfWin:data.stepSize:traceSize-halfWin));
        else
            prog = int32(100*ind/data.nWindows);
            
            if (prevprog~=prog)
                if(prevprog==-1)
                    fprintf('\n.%d%% ',prog);
                else
                    if mod(prog,2)==0
                        fprintf(repmat('\b',1,prog/2+2));
                        fprintf('%s%d%% ',repmat('.',1,prog/2),prog);
                    end
                end
                prevprog = prog;
            end
        end
        
        
        startInd = i-halfWin+1;
        endInd = i+halfWin;
        
        %% Accumulate frames for the first window
        if ind == 1 %First temporal window
            % Accumulate frames into tensor upto endInd-1
            for j = startInd:endInd
                %Read rgb frame from video
                data.frame = readFrame(vid);
                %Landmarks and tracker
                if useFaceLandmarks
                    % Find face landmarks and tracker
                    [data] = stepLandMarksDetectionandTracking(modelFile,data,j);
                    if data.skip
                        break;
                    end
                end
                %inCentroids = inpolygon(cX,cY,face_landmarks(:,1),face_landmarks(:,2));
                
                if method==0
                    %% SP
                end
                
                if method==1
                    %% SrPDE
                    [data] = stepSrPDE(j,ind,startInd,endInd,data);
                end
                if method==2
                    %% CHROM
                    [data] = stepCHROM(j,ind,startInd,endInd,data);
                end
            end
        else
            %Temporal windows 2 to last. Read frames worth step seconds and update the structures
            for j = endInd-data.stepSize+1:endInd
                data.frame = readFrame(vid);
                if useFaceLandmarks
                    [data] = stepLandMarksDetectionandTracking(modelFile,data,j);
                end
                if method==0
                    %% SP
                    [data] = stepSP(didx,j,data);
                    set(data.h223,'Xdata',j-data.winLength:j,'Ydata',squeeze(mean(mean(data.traces(:,:,j-data.winLength:j),2))));
                    set(data.s223,'XLim',[j-data.winLength,j]);
                end
                if method==2
                    %% CHROM
                    [data] = stepCHROM(j,ind,startInd,endInd,data);
                end
                if method==1
                    %% SrPDE
                    [data] = stepSrPDE(j,ind,startInd,endInd,data);
                    
                end
            end
        end
        ind = ind + 1;
        if data.skip
            fprintf('\t...skipping ');
            break;
        end
    end
    %Close the video writer at the last frame
    %There might be atmost stepSize-1 frames untreated
    if data.frameIndex<data.nFrames
        
        data.endGame = true;
        startInd = data.frameIndex+1;
        endInd = data.nFrames;
        for j = startInd:endInd
            data.frame = readFrame(vid);
            if useFaceLandmarks
                [data] = stepLandMarksDetectionandTracking(modelFile,data,j);
            end
            if method==1
                [data] = stepSrPDE(j,ind,startInd,endInd,data);
            end
            if method==2
                [data] = stepCHROM(j,ind,startInd,endInd,data);
            end
            
            
        end
    end
    
    
    if data.WRITEVIDEO && method ~=2
        close(data.videowriter);
    end
    data.computationtime=nanmean(data.times);
    
    if SAVERESULTS
        if method==0
            % SP
            %Nothing to be done for now
        end
        if method==1
            %Smooth data.sPulseTrace
            %Only smooth the non nan traces
            notnans = find(data.notnan(:,1));
            for nn = 1:numel(notnans)
                %Get the row,col index
                [r,c] = ind2sub([size(data.T,1), size(data.T,2)],notnans(nn));
                %Smooth the trace if it does not have any nan values. Pixels on
                %the face borders might have some nans at some temporal positions
                s = data.sPulseTrace(r,c,:);
                if numel(find(isnan(s)))==0
                    data.sPulseTrace(r,c,:) = smooth(s);
                end
            end
            %Normalize SNRpx_win temporally
            data.SNRpx_win = data.SNRpx_win - nanmean(data.SNRpx_win,3);
            data.SNRpx_win = data.SNRpx_win./nanstd(data.SNRpx_win,0,3);
            save([data.videowriter.Path '/pulseTrace'], 'data');
        end
        if method==2
            %CHROM
            save([db.tracesFolder 'pulseTraceCHROM-' num2str(data.roisize(1)) 'x' num2str(data.roisize(2)) '.mat' ], 'data');
        end
        
        
    end
    computationtimes(db.vidIdx) = nanmean(data.times);
    %Get next video
    db = getNextVid(db);
end

fprintf('\nAverage computation time per frame for CasmeSq db is %d', mean(computationtimes));
end

function [data] = stepCHROM(j,ind,startInd,endInd,data)
%Accumulate signal data upto endInd
if ind==1 && j==1
    %Init data structures
    data.H = size(data.frame,1);
    data.W = size(data.frame,2);
    data.roisize=[64,64];
    data.fnroimean = @(block_struct) nanmean(nanmean(block_struct.data));
    data.fnroiexpand = @(block_struct) repmat(block_struct.data,data.roisize(1),data.roisize(2));
    
    
end
data.frame = im2double(data.frame);
data.frame= imgaussfilt(data.frame, 2  );
data.frame(repmat(~data.skinmask,1,1,3))=-1;
if data.useMotionCompensation
    %Crop a rect around face such that the face remains stationary
    extents = [data.faceroi(1,2),data.faceroi(1,2)+data.faceroi(j,4), data.faceroi(1,1),data.faceroi(1,1)+data.faceroi(j,3)];
    if extents(2)>size(data.frame,1)
        extents(2) = size(data.frame,1);
    end
    if extents(4) > size(data.frame,2)
        extents(4) = size(data.frame,2);
    end
    data.frameD = data.frame(extents(1):extents(2),extents(3):extents(4),:);
    
end

data.frameD(data.frameD==-1) = nan;

frameBlock = blockproc(data.frameD,data.roisize,data.fnroimean);
% image(frameBlock);
% title(j);
% pause(.01);
if ind==1 && j==1
    data.nHCells = size(frameBlock,2);
    data.nVCells = size(frameBlock,1);
    data.sPulseTraceCHROM = zeros(data.nVCells, data.nHCells, data.nFrames);
    
    data.nHV = data.nHCells*data.nVCells;
    %RGB data stored as nxH*W to be easily used with detrendsignal
    data.R = zeros(data.nFrames,data.nHV);
    data.G = zeros(data.nFrames,data.nHV);
    data.B = zeros(data.nFrames,data.nHV);
    data.pulseTraceCHROM = zeros(1,data.nFrames);
    LOW_F = .7; UP_F = 3;
    FILTER_ORDER = 8;
    [data.bcoeff,data.acoeff] = butter(FILTER_ORDER,[LOW_F,UP_F ]/(data.Fs/2));
end


%For the first window, simply accumumate RGB
% if ind==1
data.R(j,:) = reshape(frameBlock(:,:,1),1,data.nHV);
data.G(j,:) = reshape(frameBlock(:,:,2),1,data.nHV);
data.B(j,:) = reshape(frameBlock(:,:,3),1,data.nHV);
%     %PRogress
if ind==1 && mod(j,16)==0
    fprintf('%d | ', j);
end
%
% else
%     %For windows starting from ind==2, remove the first few frames
%     %corresponding to STEP_SEC seconds and concatenate the new ones at the end
%     data.R(1,:) = []; data.G(1,:) = []; data.B(1,:) = [];
%     data.R = cat(1,data.R,reshape(frameBlock(:,:,1),1,data.nHV));
%     data.G = cat(1,data.G,reshape(frameBlock(:,:,2),1,data.nHV));
%     data.B = cat(1,data.B,reshape(frameBlock(:,:,3),1,data.nHV));
% end
%
% if j==endInd
%
%
%     data.R = filter(data.bcoeff,data.acoeff,data.R);
%     data.G = filter(data.bcoeff,data.acoeff,data.G);
%     data.B = filter(data.bcoeff,data.acoeff,data.B);
%     %If we are at the end of the window, perform CHROM
%     data.Xf = 3*data.R - 2*data.G;
%     data.Yf = 1.5*data.R+data.G-1.5*data.B;
%
%     alpha = std(data.Xf,0,1)./std(data.Yf,0,1);
%    % alpha = std(mean(data.Xf))/std(mean(data.Yf));
%     %pulseTraceCHROMWin = filter(data.bcoeff,data.acoeff,data.Xf - alpha.*data.Yf);
%     pulseTraceCHROMWin = filter(data.bcoeff,data.acoeff,data.Xf - alpha.*data.Yf);
%     sPulseTraceCHROMWin = reshape(pulseTraceCHROMWin',data.nVCells,data.nHCells,[]);
%
%     data.sPulseTraceCHROM(:,:,startInd:endInd) = data.sPulseTraceCHROM(:,:,startInd:endInd) + ((sPulseTraceCHROMWin(:,:,end-(endInd-startInd):end)));
%     data.pulseTraceCHROM(startInd:endInd) = data.pulseTraceCHROM(startInd:endInd) + smooth(detrendsignal(squeeze(nanmean(nanmean(data.sPulseTraceCHROM(:,:,startInd:endInd))))))';
% end

%For CHROM we dont use facedetection, so increment the frameindex here
if data.nFrames == data.frameIndex
    %All frames have been accumulated. Do CHROM
    data.R = data.R./mean(data.R);
    data.G = data.G./mean(data.G);
    data.B = data.B./mean(data.B);
    
    data.X = 3*data.R - 2*data.G;
    data.Y = 1.5*data.R+data.G-1.5*data.B;
    alpha = nanstd(data.X,0,1)./nanstd(data.Y,0,1);
    data.sPulseTraceCHROM = (data.X - alpha.*data.Y);
    for l = 1:size(data.sPulseTraceCHROM,2)
        data.sPulseTraceCHROM(:,l) = detrendsignal(data.sPulseTraceCHROM(:,l));
    end
end
%data.frameIndex = data.frameIndex + 1;

end

function [data] = stepSrPDE(j,ind,startInd,endInd,data)
tstart=tic;
displayOn=false;
if j==1
    %     data.H = floor(data.scale*data.faceroi(j,4));
    %     data.W = ceil(data.scale*data.faceroi(j,3));
    %Bug fix 1: To follow faceroi more accurately, T is going to cover the entire
    %frame, not just the faceroi(blocked down using roisize). This eliminates
    %the problem of T depending on faceroi and thus changing size, which cannot
    %be possible.
    data.H = size(data.frame,1);
    data.W = size(data.frame,2);
    %OUT = zeros(data.H,data.W,3,data.winLength);
    %Divide face into ROIs
    %data.roisize=[16,16];
    data.roisize=[16,16];
    data.fnroimean = @(block_struct) nanmean(nanmean(block_struct.data));
    data.fnroiexpand = @(block_struct) repmat(block_struct.data,data.roisize(1),data.roisize(2));
    
end
data.frame = im2double(data.frame);
data.frame= imgaussfilt(data.frame, 2  );
%frame = frame.*repmat(uint8(skinmask),1,1,3);
%data.frame(repmat(~data.skinmask,1,1,3))=-1;
%Face crop

%Bug fix 1: No need to crop based on faceroi. We already have non face
%pixels as -1/nan
% data.frame = (data.frame(data.faceroi(j,2):data.faceroi(j,2)+data.faceroi(j,4)-1,...
%     data.faceroi(j,1):data.faceroi(j,1)+data.faceroi(j,3)-1,:));
if data.useMotionCompensation
    %Crop a rect around face such that the face remains stationary
    data.frameC = (data.frame(data.faceroi(j,2):data.faceroi(j,2)+data.faceroi(j,4)-1,...
        data.faceroi(j,1):data.faceroi(j,1)+data.faceroi(j,3)-1,:));
else
    data.frameC = data.frame;
    
end
% subplot(1,2,1);
% image(data.frame);
% axis equal;
% subplot(1,2,2);
% image(data.frameC);
% axis equal;
% title(data.frameIndex);
% pause(.01);

%Face crop (manually done for the stationary videos)
data.frameD = (imresize(data.frameC,data.scale));
%data.frameD(data.frameD==-1) = nan;
frameBlock = blockproc(data.frameD,data.roisize,data.fnroimean);

if j==1
    data.nHCells = size(frameBlock,2);
    data.nVCells = size(frameBlock,1);
    data.notnan = zeros(data.nHCells*data.nVCells,data.nFrames);
    
    
    %T = zeros(H, W, 3, round(data.winLength)+data.maxLags);
    data.T = zeros(data.nVCells, data.nHCells, 3, round(data.winLength));
    data.HR_rPPG_avg = zeros(1,data.nWindows);
    data.HR_rPPG_win = zeros(size(data.T,1),size(data.T,2),data.nWindows);
    data.SNRpx_win = zeros(size(data.T,1),size(data.T,2),data.nWindows);
    data.MAEpx_win = zeros(size(data.T,1),size(data.T,2),data.nWindows);
    data.timeTrace = linspace(0,data.vidDuration,data.nFrames);
    data.pulseTrace = zeros(1,data.nFrames);
    %Extracted SrPDE
    data.sPulseTrace = zeros(data.nVCells, data.nHCells, data.nFrames);
    data.X_bar = zeros(data.nVCells, data.nHCells, 3);
end


if ind==1
    %For the first temporal window, just accumulate the frames into the
    %tensor
    data.T(:,:,:,j) = frameBlock;
else
    %For the rest, perform a sliding window. Remove the first frame and
    %concatenate new ones at the end. Basically, FIFO.
    data.T(:,:,:,1)= [];
    data.T = cat(4,data.T,frameBlock);
end
notnans = ~isnan(frameBlock(:,:,1));
data.notnan(notnans(:),j) = 1;
if ind==1 && mod(j,16)==0
    fprintf('%d | ', j);
end

%The following code should execute only after the first temporal window is
%populated with data, and at the end of each acquisition of stepSize frames

if(j==endInd)
    
    % Update the average tensor
    data.X_bar = nanmean(data.T,4);
    
    % Perform SrPDE here
    %Center the video data
    %Use real data for frames between data.winLength
    data.X_tilde = data.T - repmat(data.X_bar,[1 1 1 data.winLength]);
    
    %Extract the rPPG from averaged frames of the video
    data.RGB = squeeze(nanmean(squeeze(nanmean(data.X_tilde,1)),1));
    
    
    %Blockwise PVM
    data.SNRpx = zeros(size(data.X_tilde,1),size(data.X_tilde,2));
    
    data.SNRpx_all = zeros(size(data.X_tilde,1),size(data.X_tilde,2));
    
    %For FFT calculation. Distance around peak of fft to consider signal, the rest is noise
    WIDTH = 0.3;
    LOW_F = .7; UP_F = 3; %Max and min heart rate in heartz ~42-180bpm
    
    % Pseudo GT, from averaged RGB traces using PVM
    [lags, y, p_rPPG_freq, p_rPPG_power,rPPG_peaksLoc, wPVM] = getTauFromRGB(data.RGB,data.Fs);
    data.HR_rPPG_avg(ind) = rPPG_peaksLoc(1)*60;
    
    %Reshape X_tilde into a 3xnxhxw tensor
    X = data.X_tilde(:,:,:,1:end-lags);
    Xt = data.X_tilde(:,:,:,lags+1 : end);
    
    X_permuted = permute(X,[3 4 1 2]); % 3 x n x h x w
    
    Xt_permuted = permute(Xt,[4 3 1 2]); % n x 3 x h x w
    
    %Flatten last two WxW dimensions to h*W
    X_permuted = reshape(X_permuted,size(X_permuted,1),size(X_permuted,2),size(X_permuted,3)*size(X_permuted,4));
    Xt_permuted = reshape(Xt_permuted,size(Xt_permuted,1),size(Xt_permuted,2),size(Xt_permuted,3)*size(Xt_permuted,4));
    
    PHI_nan = mtimesx(X_permuted,'N',X_permuted,'T','SPEEDOMP','OMP_SET_NUM_THREADS(4)');
    PI_nan = mtimesx(X_permuted,'N',Xt_permuted,'N','SPEEDOMP','OMP_SET_NUM_THREADS(4)');
    
    notnan = ~isnan(PHI_nan);
    PHI = PHI_nan(:,:,squeeze(notnan(1,1,:)));
    PI = PI_nan(:,:,squeeze(notnan(1,1,:)));
    
    Vbest_nan = zeros(1,3,size(PHI_nan,3));
    [PI, PHI, alfr, alfi, beta, V, Vbest] = qz_temporal(PI,PHI,true);
    Vbest_nan(:,:,notnan(1,1,:))=Vbest;
    
    %Calculate Cx and Px tensors for tau, the period
    %Cx is pixelwise cov matrix of the whole signal
    X_prj = squeeze(mtimesx(Vbest_nan,'N',X_permuted(:,1:end,:),'SPEEDOMP','OMP_SET_NUM_THREADS(4)'));
    X_prj = reshape(permute(X_prj,[2,1]),size(X,1),size(X,2),[]);
    MAX = max(max(max(X_prj)));
    MIN = min(min(min(X_prj)));
    den = MAX-MIN;
    
    if data.endGame
        data.sPulseTrace(:,:,startInd:endInd) = data.sPulseTrace(:,:,startInd:endInd) + detrendsignal((X_prj(:,:,end-(endInd-startInd):end)));
        data.pulseTrace(startInd:endInd) = data.pulseTrace(startInd:endInd) + smooth(detrendsignal(squeeze(nanmean(nanmean(data.sPulseTrace(:,:,startInd:endInd))))))';
    else
        %Overlap add
        data.sPulseTrace(:,:,startInd:endInd - lags) = data.sPulseTrace(:,:,startInd:endInd - lags) + detrendsignal((X_prj(:,:,1:data.winLength - lags)));
        data.pulseTrace(startInd:endInd - lags) = data.pulseTrace(startInd:endInd-lags) + smooth(detrendsignal(squeeze(nanmean(nanmean(data.sPulseTrace(:,:,startInd:endInd - lags))))))';
        %Calculate block wise SNR compared to PVM on average face
        %for each roi, calculate rPPG fft and compare with PPG fft
        [R, C, L] = size(X_prj);
        data.SNRpx = nan*ones(R,C);
        
        for r= 1:R
            [rPPG_freq, rPPG_powers, rPPG_peaksLoc] = fftrPPG(squeeze(X_prj(r,:,:)),data.Fs,'noplot',true);
            for c = 1:C
                if ~isnan(rPPG_powers(c,1))
                    rPPG_power = rPPG_powers(c,:);
                    
                    % Get peaks rPPG
                    rangeRPPG = (rPPG_freq>LOW_F & rPPG_freq < UP_F);   % frequency range to find PSD peak
                    rPPG_power = rPPG_power(rangeRPPG);
                    rPPG_freq = rPPG_freq(rangeRPPG);
                    
                    range1 = (rPPG_freq>(rPPG_peaksLoc(c,1)-WIDTH/2) & rPPG_freq < (rPPG_peaksLoc(c,1)+WIDTH/2));
                    range2 = (rPPG_freq>(rPPG_peaksLoc(c,2)*2-WIDTH/2) & rPPG_freq < (rPPG_peaksLoc(c,2)*2+WIDTH/2));
                    range = range1 + range2;
                    signal = rPPG_power.*range;
                    noise = rPPG_power.*(~range);
                    data.SNRpx(r,c) = 10*log10(sum(signal)/sum(noise));
                    data.HR_rPPG_win(r,c,ind) = rPPG_peaksLoc(c,1)*60;
                end
            end
        end
        
        %Normalize SNRpx
        data.SNRpx = data.SNRpx - nanmean(nanmean(data.SNRpx));
        data.SNRpx = data.SNRpx/nanstd(nanstd(data.SNRpx));
        data.SNRpx_win(:,:,ind) = data.SNRpx;
        
        data.SNRpx_all = data.SNRpx_all + data.SNRpx;
        displayOn = false;
        finalPlotColor = [0 .45 .74];
        
    end
    
    %Save and (optionally) display the raw SrPDE
    endIndex = startInd + data.stepSize - 1;
    if data.endGame
        endIndex = endInd;
    end
    for I=startInd:endIndex
        data.frameD = data.X_bar +(data.SNRpx.*repmat(data.sPulseTrace(:,:,I),1,1,3));
        %% High def display
        %         if saveHighdef
        %             X_prj_big=blockproc(X_prj(:,:,I-startInd+1),[1 1],fnroiexpand);
        %             SNRbig=(blockproc(SNRpx,[ 1 1],fnroiexpand));
        %             %Crop to roi size
        %             szdelta = size(SNRbig,1) - size(X_orig_bar,1);
        %             SNRbig=(SNRbig(szdelta/2+1:end-szdelta/2,szdelta/2+1:end-szdelta/2));
        %             X_prj_big=(X_prj_big(szdelta/2+1:end-szdelta/2,szdelta/2+1:end-szdelta/2));
        %             frameoD = X_orig_bar + SNRbig.*repmat( X_prj_big,1,1,3);
        %             writeVideo(videowriter,rescale(frameoD));
        %         else
        if data.WRITEVIDEO
            writeVideo(data.videowriter,blockproc(rescale(data.frameD),[1 1],data.fnroiexpand));
        end
        %         end
        if(displayOn)
            figure(1);
            subplot(2,3,1);
            %             if saveHighdef
            %                 imagesc(frameoD);
            %             else
            imagesc(blockproc(data.frameD,[ 1 1],data.fnroiexpand));
            %             end
            %imagesc(sPulseTrace(:,:,I));
            title(num2str(I));
            subplot(2,3,2);
            %[PPGlag, PPG_peaksLoc] = getTauFromPPG(gtTimeWin,gtTraceWin, Fs_PPG);
            %[rPPG_freq,rPPG_power] = fftrPPG(data.pulseTrace(startInd:endInd - data.maxLags), data.Fs);
            subplot(2,3,[4 5 6]);
            grid on;
            hold on;
            
            if I<startInd + data.stepSize - 1
                %t = Ts*double(I : I+1);
                t = data.timeTrace(I:I+1);
                x = data.pulseTrace(I:I+1);
                p=plot(t,x,'-o','Color',finalPlotColor,'MarkerSize',3);
                set(p,'MarkerFaceColor',finalPlotColor);
            end
            
            hold off
            pause(0.01);
        end
    end
    
end
data.times(j) = toc(tstart);

end


function [data] = stepLandMarksDetectionandTracking(modelFile,data,j)
if j==1
    %First check if the forehead points are already present
    fnm=[data.vidName(1:end-4) '-foreheadExtent.mat'];
    
    if exist(fnm,'file')==2
        load(fnm,'foreheadExtent');
        data.foreheadExtent = foreheadExtent;
        %data.skip = true;
    else
        %User shall select 5 equidistant points to encompass the forehead once.
        %Ask the user to manually select
        fig=uifigure('Name','Select 5 equidistant points to encompass the forehead','HandleVisibility', 'on');
        ax = uiaxes(fig,'HandleVisibility', 'on');
        image(ax,data.frame);
        
        response='y';
        foreheadExtent = zeros(2,5);
        while(response=='y')
            for i=1:5
                [foreheadExtent(1,i), foreheadExtent(2,i) ] = selectDataPoints(ax);
            end
            %            foreheadExtent = ginput(5);
            msg = 'Saving forehead extent coordinates. Click on Restart if you want to go again.';
            txt = 'Confirm Save';
            selection = uiconfirm(fig,msg,txt,...
                'Options',{'Save','Restart'},...
                'DefaultOption',1);
            if strcmp(selection,'Save')==1
                response='n';
                save(fnm,'foreheadExtent');
                data.foreheadExtent = foreheadExtent;
            else
                response='y';
                foreheadExtent = zeros(2,5);
                cla(ax);
                image(ax,data.frame);
            end
        end
        close(fig);
    end
    % Find face landmarks and tracker
    lmd = find_face_landmarks(modelFile, data.frame);
    Xs = double(lmd.faces.landmarks(:,1)');
    Ys = double(lmd.faces.landmarks(:,2)');
    
    % Heuristic bounds to include the forehead. [x;y], not so accurate
    %Forehead
    %minX = min(Xs); minY = min(Ys);
    %maxX = max(Xs); maxY = max(Ys);
    %foreheadExtent = (maxY - minY)/4;
    
    % bounds = [ minX, minX+(maxX-minX)/5, ...
    %     Xs(28), maxX-(maxX-minX)/5,...
    %     maxX; minY,foreheadExtent,foreheadExtent-20,foreheadExtent,minY];
    
    Xs = [Xs, data.foreheadExtent(1,:)]; Ys = [Ys, data.foreheadExtent(2,:)];
    if data.useMotionCompensation
        %TODO: track a rect around the face coordinates based on the motion
        %such that the face remains stationary inside the rect
        data.hpad = 30;
        data.vpad = 30;
        
        %Create a mask to remove eye area so that face warping can work
        %properly
        data.eyeMaskCoords=[1,18:20,25:27,17,16,29,2];
        data.eyemask = roipoly(data.frame,Xs(data.eyeMaskCoords),Ys(data.eyeMaskCoords));
        
        %Track all face points to update data.eyemask
        data.face_landmarks=[Xs',Ys'];
        %Set up points around the eye to use for motion compensation
        data.fixedPointsCoords = [1:17,size(data.face_landmarks,1)-4:size(data.face_landmarks,1)];
        data.fixedPoints = int32([data.face_landmarks(data.fixedPointsCoords,1),data.face_landmarks(data.fixedPointsCoords,2)]);
        
        
        data.k = convhull(Xs,Ys);
        data.skinmask = roipoly(data.frame,Xs(data.k), Ys(data.k));
        data.skinmask = data.skinmask.*~data.eyemask;
        
    else
        data.hpad = 0;
        data.vpad = 0;
        %Convex hull
        data.k = convhull(Xs,Ys);
        %data.face_landmarks=[Xs(data.k)',Ys(data.k)'];
        data.face_landmarks=[Xs',Ys'];
        data.skinmask = roipoly(data.frame,Xs(data.k), Ys(data.k));
    end
    
    
    
    initialize(data.pointTracker,data.face_landmarks,data.frame);
    
    %Clean up face_landmarks, clamp points out of the image to borders
    
    yclamp = data.face_landmarks(:,2)>size(data.frame,1);
    xclamp = data.face_landmarks(:,1)>size(data.frame,2);
    data.face_landmarks(yclamp,2)=size(data.frame,1);
    data.face_landmarks(xclamp,1)=size(data.frame,2);
    %Defer faceroi from landmarks
    minX = round(min(data.face_landmarks(:,1))); minY=round(min(data.face_landmarks(:,2)));
    maxX = round(max(data.face_landmarks(:,1))); maxY=round(max(data.face_landmarks(:,2)));
    W = floor(max(data.face_landmarks(:,1))-minX); H=floor(max(data.face_landmarks(:,2))-minY);
    data.xoffset = 0; data.yoffset = 0;
    if mod(W+2*data.hpad,2)~=0
        data.xoffset = 0;
    end
    if mod(H+2*data.vpad,2)~=0
        data.yoffset = 0;
    end
    %if minY-data.vpad exceeds face boundaries
    if minY <= data.vpad
        data.vpad = 0;
    end
    
    if minX <= data.hpad
        data.hpad = 0;
    end
    %If extents go beyond the image size
    if size(data.frame,1)<=minY + H +2*data.vpad
        data.vpad = floor((size(data.frame,1) -minY -H )/2);
    end
    if size(data.frame,2)<=minX + W +data.hpad
        data.hpad = floor((size(data.frame,2) -minX -W )/2);
    end
    
    data.faceroi(j,:) = [minX-data.hpad,minY-data.vpad,W+2*data.hpad + data.xoffset,H+2*data.vpad+data.yoffset];
    
    
else
    %Defer faceroi from landmarks
    [data.face_landmarks, ~] = data.pointTracker(data.frame);
    minX = round(min(data.face_landmarks(:,1))); minY=round(min(data.face_landmarks(:,2)));
    W = floor(max(data.face_landmarks(:,1))-minX); H=floor(max(data.face_landmarks(:,2))-minY);
    
    if data.useMotionCompensation
        data.faceroi(j,:) = [minX-data.hpad,minY-data.vpad,data.faceroi(1,3), data.faceroi(1,4)];
        if norm(data.faceroi(j-1,1:2) - data.faceroi(j,1:2))~=0j
            
            %             %Warp changed faceroi to previous faceroi, simple rect
            %             outPts = [data.faceroi(1,1) data.faceroi(1,2);...
            %                 data.faceroi(1,1)+data.faceroi(1,3) data.faceroi(1,2);...
            %                 data.faceroi(1,1) data.faceroi(j-1,2)+data.faceroi(1,4);...
            %                 data.faceroi(1,1)+data.faceroi(1,3) data.faceroi(1,2)+data.faceroi(1,4)];
            %             inPts = [data.faceroi(j,1) data.faceroi(j,2);...
            %                 data.faceroi(j,1)+data.faceroi(j,3) data.faceroi(j,2);...
            %                 data.faceroi(j,1) data.faceroi(j,2)+data.faceroi(j,4);...
            %                 data.faceroi(j,1)++data.faceroi(j,3) data.faceroi(j,2)+data.faceroi(j,4)];
            
            %Warp using face outline to a fixed position, removing the
            %eyes.
            inPts = int32([data.face_landmarks(data.fixedPointsCoords,1), data.face_landmarks(data.fixedPointsCoords,2)]);
            %Estimate transform between current and original
            %data.fixedPointsCoords
            tran = estimateGeometricTransform(inPts,data.fixedPoints,'similarity');
            %Transform the face landmarks to the new projection space
            [data.face_landmarks(:,1),data.face_landmarks(:,2)] = ...
                tran.transformPointsForward(double(data.face_landmarks(:,1)),double(data.face_landmarks(:,2)));
            
            %Update faceroi
            [x, y]=tran.transformPointsForward(double(data.faceroi(j,1)),double(data.faceroi(j,2)));
            %data.faceroi(j,1:2) = int32([x,y]);
            %data.frame = imwarp(data.frame,tran,'nearest','OutputView',imref2d(size(data.frame)));
        end
        data.skinmask = roipoly(data.frame,data.face_landmarks(data.k,1), data.face_landmarks(data.k,2));
        data.skinmask = data.skinmask.*~data.eyemask;
        
    else
        data.faceroi(j,:) = [minX,minY,W,H];
        data.skinmask = roipoly(data.frame,data.face_landmarks(:,1), data.face_landmarks(:,2));
    end
    
    
end
data.frameIndex = data.frameIndex + 1;
%% Uncomment the following block to verify face landmarks
% if j>2
% figure(1);
% data.lmdisplay = image(mat2gray(data.frame).*repmat(data.skinmask,1,1,3));
%
% hold on;
% %data.lms = plot(data.face_landmarks(:,1),data.face_landmarks(:,2),'.','MarkerSize',8);
% data.lmtitle = title(j);
% %images.roi.Rectangle(gca,'Position',data.faceroi(j,:));
% drawnow;
% pause(.01);
% end
% figure(2);
% hold on;
% if j>1
%     plot(j-1:j,data.faceroi(j-1:j,1),'r');
%     %plot(j-1:j,data.faceroi(j-1:j,2),'g');
%     %plot(j-1:j,data.faceroi(j-1:j,3),'b');
%     %plot(j-1:j,data.faceroi(j-1:j,4),'k');
%
% end
end

function [SP] = stepSP(didx, j, SP)
spdata = [];
% Read SP trace values for this frame
if exist([SP.segfiles(didx).folder '/traces.mat' ],'file')
    load([SP.segfiles(didx).folder '/traces' ]);
    SP.traces = traces;
    SP.SPNumber = size(traces,2);
else
    fname = sprintf('%s/traces_%04d.seg', SP.segfiles(didx).folder,j);
    fileID = fopen(fname,'r');
    spdata = fscanf(fileID,'%f %f %f %f %f %f',[6 Inf]);
    fclose(fileID);
    if ~isfield(SP,'traces')
        SP.SPNumber = size(spdata,2);
        SP.traces = zeros(3,SP.SPNumber,SP.nFrames);
    end
    SP.traces(:,:,j) = spdata(1:3,:);
    SP.SPNumber = size(spdata,2);
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
    
    SP.s221=subplot(2,2,1);
    SP.h221 = image(lFrame);
    title('Superpixel RGB');
    hold on;
    %hlm = plot(face_landmarks(:,1),face_landmarks(:,2),'w.','MarkerSize',10);
    SP.hlm = plot(SP.face_landmarks(:,1),SP.face_landmarks(:,2),'w.','MarkerSize',10);
    SP.hC = plot(cX(SP.inCentroids), cY(SP.inCentroids),'x');
    
    axis equal; axis tight;
    SP.s222=subplot(2,2,2);
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


function [x, y] = selectDataPoints( ax)
roi = drawpoint(ax);
x = roi.Position(1);
y = roi.Position(2);
end


