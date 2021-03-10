function [] = analyseCasme(varargin)
%analyseCasme Iterate through Casme db videos and analyse the .seg files
%from IBIS_Temporal
%   Detailed explanation goes here
tracesFolder = '/mnt/nvmedisk/CasmeSq/traces_500_50/rawvideo/';
vidFolder = '/mnt/nvmedisk/CasmeSq/rawvideo/';
METHOD_SrPDE = 1; % SP=0 or SrPDE=1

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

addpath('/usr/local/interfaces/matlab/');% addpath('../');
% addpath('../../tools/tensor_toolbox_2.1');
% addpath('../../tools/skindetector');
% addpath('../../main-stable');

%setenv('LD_LIBRARY_PATH', '');
useFaceLandmarks = true;
modelFile='/home/richard/src/IBIS_Temporal/shape_predictor_68_face_landmarks.dat';

%Path to tensor-qz
addpath('../tensor-qz/');
addpath('../mtimesx/');
%% Expressions GT for CasmeSq
load([vidFolder '../codefinal.mat']);
%Function handle to extract expression name from second column of
%tcodefinal (e.g. anger1_1, anger1_2, etc.). Shall be used in cellfun, due
%to the tabular structure of the ground truth
cellsubstr=@(str) str(1:end-2);


%% Access the CasmeSq directory structure
dirs = dir(vidFolder);
computationtimes=[];
vididx = 1;
%Iterate through each dir. Start from 3 to skip . and .. listing
for didx=3:numel(dirs)
    %We have subject directories here: s15, s16,etc.
    viddir = dir([dirs(didx).folder '/' dirs(didx).name '/*.avi']);
    %We have the video directories here, we can access the .seg files now
    
    %Find all the expressions for this subject. Each subject has multiple
    %videos
    sid = str2num(dirs(didx).name(2:end));
    %all_expr_gt = tcodefinal(tnaming_rule1{:,1}==sid,:);
    
    for vidx=1:numel(viddir)
        %for vidx=6:numel(viddir)
        %Read expressions
        
        %Get expression list for current video from all_exp_gt
        
        %         videocode=viddir(vidx).name(4:7);
        %         %Get expression name from videocode
        %         li = ismember(tnaming_rule2{:,1},videocode);
        %         expr_name = tnaming_rule2{li,2}{:};
        %         %Use this expr_name to filter all_exp_gt
        %         all_exprs = cellfun(cellsubstr,all_expr_gt{:,2},'UniformOutput',false);
        %         filter = ismember(all_exprs,expr_name);
        %         gtrows = all_expr_gt(filter,:);
        %         %gtrows contains all expressions for this video
        %         expr_frame_indices = gtrows{:,3:5}; %int32 matrix
        %         expr_start = 0;
        %TODO: Add a summary of expressions somewhere in the window
        
        data = struct; %Structure to encompass all processing and data
        data.WRITEVIDEO = true;
        data.pointTracker = vision.PointTracker;
        data.tracesFolder = [tracesFolder  dirs(didx).name '/' viddir(vidx).name];
        if exist(data.tracesFolder,'dir')~=7
            mkdir(data.tracesFolder);
        end
%         if ~strcmp(viddir(vidx).name,'37_0502funnyerrors.avi')
%             continue;
%         end
        
        %Create data.tracesFolder if it does not exist
        data.vidName= [vidFolder dirs(didx).name '/' viddir(vidx).name ];
        
        %Read original video
        vid = VideoReader(data.vidName);
        %Read labels video
        %lblVid = VideoReader([viddir(j).folder '/' viddir(j).name '/labels.avi']);
        vidPlayer = vision.VideoPlayer;
        STEP_SEC = .2;
        data.Fs=30; %Casme2 database fps.
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
        data.vidDuration = double(traceSize/data.Fs);
        data.frameIndex=1;
        data.skip = false;
        data.times = nan*ones(1,traceSize);
        
        if ~METHOD_SrPDE
            %% Read SP contours
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
        else
            %% SrPDE processing
            data.videowriter = VideoWriter([data.tracesFolder   '/srpderaw.avi'],'Uncompressed AVI');
            data.videowriter.FrameRate = data.Fs;
            if data.WRITEVIDEO
                open(data.videowriter);
            end
            
            data.scale = 1;
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
                fprintf('\n %s : Accumulating %d frames for the first window (total %d)\n',viddir(vidx).name, data.winLength,numel(halfWin:data.stepSize:traceSize-halfWin));
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
            %             is_expr_frame =find(expr_frame_indices(:,1)==startInd);
            %             if is_expr_frame
            %                 expr_start = expr_frame_indices(is_expr_frame,1);
            %                 expr_peak = expr_frame_indices(is_expr_frame,3);
            %                 expr_end = expr_frame_indices(is_expr_frame,3);
            %             end
            %             %If we are in the middle of an expression, check if we have
            %             %reached the apex or end frames
            %             if expr_start~=0
            %                 if startInd == expr_end
            %                     %Reset variables
            %                     expr_start =0; expr_peak = 0; expr_end = 0;
            %                 else
            %                     %TODO:
            %                     %Logic for displaying expressions text in the plots
            %                     %Display a text annotation expr_start to expr_end frame
            %                     %Add gradient with a peak at expr_peak frame
            %                 end
            %
            %             end
            %
            
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
                        %% Uncomment the following block to verify face landmarks
                        %figure(1);
                        %data.lmdisplay = image(data.frame);
                        %hold on;
                        %data.lms = plot(Xs,Ys,'.','MarkerSize',8);
                        %data.lmtitle = title(j);
                        %drawnow;
                        %pause(.01);
                        %images.roi.Rectangle(gca,'Position',data.faceroi)
                        
                    end
                    %inCentroids = inpolygon(cX,cY,face_landmarks(:,1),face_landmarks(:,2));
                    
                    if ~METHOD_SrPDE
                        %% SP
                        [data] = stepSP(didx,j,data);
                        data.s223=subplot(2,2,3:4);
                        data.h223 = plot(1:data.winLength, zeros(1,data.winLength));
                        grid on;
                        if(j==data.nFrames-1)
                            %Save traces so that if we execute the code
                            %next time we don't need to do a lot of disk io
                            %to read the traces
                            save([segfiles(didx).folder '/traces' ],'SP.traces');
                        end
                    else
                        %% SrPDE
                        [data] = stepSrPDE(j,ind,startInd,endInd,data);
                        
                        
                    end
                end
            else
                %Temporal windows 2 to last. Read frames worth step seconds and update the structures
                for j = endInd-data.stepSize+1:endInd
                    data.frame = readFrame(vid);
                    if useFaceLandmarks
                        [data] = stepLandMarksDetectionandTracking(modelFile,data,j);
                    end
                    if ~METHOD_SrPDE
                        %% SP
                        [data] = stepSP(didx,j,data);
                        set(data.h223,'Xdata',j-data.winLength:j,'Ydata',squeeze(mean(mean(data.traces(:,:,j-data.winLength:j),2))));
                        set(data.s223,'XLim',[j-data.winLength,j]);
                    else
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
        if data.frameIndex+1<data.nFrames
            
            data.endGame = true;
            startInd = data.frameIndex+1;
            endInd = data.nFrames;
            for j = startInd:endInd
                data.frame = readFrame(vid);
                if useFaceLandmarks
                    [data] = stepLandMarksDetectionandTracking(modelFile,data,j);
                end
                [data] = stepSrPDE(j,ind,startInd,endInd,data);                
            end
        end
        
        %Normalize SNRpx_win temporally
        data.SNRpx_win = data.SNRpx_win - nanmean(data.SNRpx_win,3);
        data.SNRpx_win = data.SNRpx_win./nanstd(data.SNRpx_win,0,3);
        
        if data.WRITEVIDEO
            close(data.videowriter);
        end
        data.computationtime=nanmean(data.times);
        save([data.videowriter.Path '/pulseTrace'], 'data');
        computationtimes(vididx) = nanmean(data.times);
        vididx = vididx + 1;
    end
    
end
fprintf('\nAverage computation time per frame for CasmeSq db is %d', mean(computationtimes));
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
    data.roisize=[16,16];
    data.roisize=[8,8];
    data.nHCells = ceil(data.W/data.roisize(1));
    data.nVCells = floor(data.H/data.roisize(2));
    
    data.fnroimean = @(block_struct) nanmean(nanmean(block_struct.data));
    data.fnroiexpand = @(block_struct) repmat(block_struct.data,data.roisize(1),data.roisize(2));
    
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
data.frame = im2double(data.frame);
data.frame= imgaussfilt(data.frame, 2  );
%frame = frame.*repmat(uint8(skinmask),1,1,3);
data.frame(repmat(~data.skinmask,1,1,3))=-1;
%Face crop

%Bug fix 1: No need to crop based on faceroi. We already have non face
%pixels as -1/nan
% data.frame = (data.frame(data.faceroi(j,2):data.faceroi(j,2)+data.faceroi(j,4)-1,...
%     data.faceroi(j,1):data.faceroi(j,1)+data.faceroi(j,3)-1,:));

%Face crop (manually done for the stationary videos)
data.frameD = (imresize(data.frame,data.scale));
data.frameD(data.frameD==-1) = nan;
if ind==1
    %For the first temporal window, just accumulate the frames into the
    %tensor
    data.T(:,:,:,j) = blockproc(data.frameD,data.roisize,data.fnroimean);
else
    %For the rest, perform a sliding window. Remove the first frame and
    %concatenate new ones at the end. Basically, FIFO.
    data.T(:,:,:,1)= [];
    data.T = cat(4,data.T,blockproc(data.frameD,data.roisize,data.fnroimean));
    
end
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
    
    %Reshape X_tilde into a 3xNxWxW tensor
    X = data.X_tilde(:,:,:,1:end-lags);
    Xt = data.X_tilde(:,:,:,lags+1 : end);
    
    X_permuted = permute(X,[3 4 1 2]); % 3 x N x W x W
    
    Xt_permuted = permute(Xt,[4 3 1 2]); % 3 x N x W x W
    
    %Flatten last two WxW dimensions to W*W
    X_permuted = reshape(X_permuted,size(X_permuted,1),size(X_permuted,2),size(X_permuted,3)*size(X_permuted,4));
    Xt_permuted = reshape(Xt_permuted,size(Xt_permuted,1),size(Xt_permuted,2),size(Xt_permuted,3)*size(Xt_permuted,4));
    
    PHI_nan = mtimesx(X_permuted,'N',X_permuted,'T','SPEEDOMP','OMP_SET_NUM_THREADS(4)');
    PI_nan = mtimesx(X_permuted,'N',Xt_permuted,'N','SPEEDOMP','OMP_SET_NUM_THREADS(4)');
    
    notnan = ~isnan(PHI_nan);
    PHI = PHI_nan(:,:,squeeze(notnan(1,1,:)));
    PI = PI_nan(:,:,squeeze(notnan(1,1,:)));
    
    Vbest_nan = zeros(1,3,size(PHI_nan,3));
    [PI, PHI, alfr, alfi, beta, V, Vbest] = qz_tensor(PI,PHI,true);
    Vbest_nan(:,:,notnan(1,1,:))=Vbest;
    
    %Calculate Cx and Px tensors for tau, the period
    %Cx is pixelwise cov matrix of the whole signal
    X_prj = squeeze(mtimesx(Vbest_nan,'N',X_permuted(:,1:end,:),'SPEEDOMP','OMP_SET_NUM_THREADS(4)'));
    X_prj = reshape(permute(X_prj,[2,1]),size(X,1),size(X,2),[]);
    MAX = max(max(max(X_prj)));
    MIN = min(min(min(X_prj)));
    den = MAX-MIN;
    
    if data.endGame
        data.sPulseTrace(:,:,startInd:endInd) = data.sPulseTrace(:,:,startInd:endInd) + smooth3(X_prj(:,:,end-(endInd-startInd):end));
        data.pulseTrace(startInd:endInd) = data.pulseTrace(startInd:endInd) + smooth(detrendsignal(squeeze(nanmean(nanmean(data.sPulseTrace(:,:,startInd:endInd))))))';
    else
        %Overlap add
        data.sPulseTrace(:,:,startInd:endInd - lags) = data.sPulseTrace(:,:,startInd:endInd - lags) + smooth3(X_prj(:,:,1:data.winLength - lags));
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
    %Convex hull
    k = convhull(Xs,Ys);
    data.face_landmarks=[Xs(k)',Ys(k)'];
    initialize(data.pointTracker,data.face_landmarks,data.frame);
    data.skinmask = roipoly(data.frame,Xs(k), Ys(k));
    %Defer faceroi from landmarks
    minX = round(min(data.face_landmarks(:,1))); minY=round(min(data.face_landmarks(:,2)));
    W = floor(max(data.face_landmarks(:,1))-minX); H=floor(max(data.face_landmarks(:,2))-minY);
    data.faceroi(j,:) = [minX,minY,W,H];
else
    %Defer faceroi from landmarks
    [data.face_landmarks, ~] = data.pointTracker(data.frame);
    minX = round(min(data.face_landmarks(:,1))); minY=round(min(data.face_landmarks(:,2)));
    W = floor(max(data.face_landmarks(:,1))-minX); H=floor(max(data.face_landmarks(:,2))-minY);
    data.faceroi(j,:) = [minX,minY,W,H];
    data.skinmask = roipoly(data.frame,data.face_landmarks(:,1), data.face_landmarks(:,2));
    
end
data.frameIndex = data.frameIndex + 1;
end

function [x, y] = selectDataPoints( ax)
roi = drawpoint(ax);
x = roi.Position(1);
y = roi.Position(2);
end