
%Erode points by N pixels
function [outX, outY] = dilatePoints(X, Y, SPNumber, frame)
%Guess N from SPNumber, higher the SPNumber, smaller N should be
N=round(2000/SPNumber);

cX = mean(X); cY = mean(Y);
leftPoints = X<cX;
rightPoints = X>cY;

topPoints = Y<cY;
bottomPoints = Y>cY;

%Dilate by N pixels
% X(leftPoints) = X(leftPoints) - N;
X(rightPoints) = X(rightPoints) + N;

% Y(topPoints) = Y(topPoints) - N;
Y(bottomPoints) = Y(bottomPoints) + N;

%Dilate a bit more, to make sure a dilation of N points in the direction of
%the point from the center, and not just in hor and ver
%Y(topPoints&leftPoints) = Y(topPoints&leftPoints) - sqrt(N*(N-1));
Y(bottomPoints&leftPoints) = Y(bottomPoints&leftPoints) + sqrt(N*(N+1));

%Y(topPoints&rightPoints) = Y(topPoints&rightPoints) - sqrt(N*(N-1));
Y(bottomPoints&rightPoints) = Y(bottomPoints&rightPoints) + sqrt(N*(N+1));


outX = X;
outY = Y;
end
