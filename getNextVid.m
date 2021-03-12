function [db] = getNextVid(db)
% Get next video from the database. The function takes care of iterating
% according to the directory structure of the database. After calling this
% function, db.nextVid contains the path to the next video folder
% db is a structure. db.res contains true if there is a next video
% db.idx = 1 (CasmeSq), 2(FixedData) or 3 (MMSE);
% db.vidToProcess can be used to limit processing to single videos
%Populate all the video paths in a list so that iteration is easy
i=1;

switch db.idx
    case 1 %CasmeSq
        if ~exist('db.videos','var')
            %disp('Populating Casmesq video list');
            tracesFolder = '/mnt/nvmedisk/CasmeSq/traces_500_50_MC/rawvideo/';
            db.vidFolder = '/mnt/nvmedisk/CasmeSq/rawvideo/';
            dirs = dir(db.vidFolder);
            %s26,s27 face detection needs to be better
            db.vidToProcess = '26_0101disgustingteeth.avi';
            %Process videos between vidStartIdx and vidEndIdx
            db.vidEndIdx = 1;
            db.vidIdx = 1; %Current video index
            
            %Iterate through each dir. Start from 3 to skip . and .. listing
            for didx=3:numel(dirs)
                %We have subject directories here: s15, s16,etc.
                viddir = dir([dirs(didx).folder '/' dirs(didx).name '/*.avi']);
                for vidx=1:numel(viddir)
                    %disp(viddir(vidx).name);
                    db.videonames{i} = viddir(vidx).name;
                    db.videos{i} = [viddir(vidx).folder '/' viddir(vidx).name];
                    db.tracesFolders{i} = [tracesFolder  dirs(didx).name '/' viddir(vidx).name];
                    %Create data.tracesFolder if it does not exist
                    if exist(db.tracesFolders{i},'dir')~=7
                        mkdir(data.tracesFolders);
                    end
                    i = i+1;
                end
            end
        end
        
        %Skip videos upto vidToProcess
        while strcmp(db.vidToProcess,db.videonames{db.vidIdx})~=1
            db.vidIdx = db.vidIdx + 1;
            if db.vidEndIdx < db.vidIdx
                db.vidEndIdx = db.vidIdx;
            end
        end
        
        %Get next video from list
        if db.vidIdx <=numel(db.videos) && db.vidIdx <= db.vidEndIdx
            db.nextVid = db.videos{db.vidIdx};
            db.nextVidName = db.videonames{db.vidIdx};
            db.tracesFolder = db.tracesFolders{db.vidIdx};
            db.res = true;
            db.vidIdx = db.vidIdx + 1;
        else
            db.res = false;
        end
        
    case 2 %FixedData
        if ~isfield(db,'videos')
            disp('Populating FixedData video list');
            fixedDataFolder = '//mnt/nvmedisk/fixedData20122017/';
            %db.vidToProcess = '2017_12_20-15_35_40';
            db.vidToProcess = []; %Processs all videos
            folders = dir(fixedDataFolder);
            %ignoreList = {'2017_12_20-14_55_23', '2017_12_20-15_02_30','2017_12_20-15_06_34'};
            idx = 1;
            for i=1:size(folders,1)
                skip = false;
                if ~strcmp(folders(i).name,'.') && ~strcmp(folders(i).name,'..') 
                    rppgOut = dir([fixedDataFolder folders(i).name '\rppgOut.avi']);
                    if ~isempty(rppgOut) && rppgOut.bytes ~=0
                        fprintf('%s already processed, skipping... \n',folders(i).name);
                        skip = true;
                    end
                    if(~skip)
                        db.videonames{idx} = folders(i).name;
                        db.videos{idx} = [fixedDataFolder folders(i).name '/'];
                        idx = idx + 1;
                    end
                end
            end
            db.vidIdx = 1;
        end
        
        %Skip videos upto vidToProcess
        while ~isempty(db.vidToProcess) && db.vidIdx<=numel(db.videos) && strcmp(db.vidToProcess,db.videonames{db.vidIdx})~=1
            db.vidIdx = db.vidIdx + 1;
%             if db.vidEndIdx < db.vidIdx
%                 db.vidEndIdx = db.vidIdx;
%             end
        end
        
        if db.vidIdx <= numel(db.videos) 
            db.nextVid = [db.videos{db.vidIdx} 'vid.avi'];
            db.nextVidName = db.videonames{db.vidIdx};
            db.tracesFolder = db.videos{db.vidIdx};
            db.res = true;
            db.vidIdx = db.vidIdx + 1;
        else
            db.res = false;
        end
        
        

    case 3 %MMSE
        disp('Populating MMSE video list');
end


end