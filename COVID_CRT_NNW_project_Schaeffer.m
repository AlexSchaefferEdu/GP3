clear all
close all
clc

% Size of each picture
m = 299;
n = 299;

% Number of sample pictures
N = 100;

avg = zeros(m*n,1);  % the average face
A = [];
AC = [];
AN = [];
%% Load COVID
count = 0;
for j = 901:1000
    %figure(1)
    ff = ['COVIDData/COVID-',num2str(j,'%2d'),'.png'];
    u = imread(ff); % Read the image into a matrix
  %  imshow(u)
   % title('COVID Infected Lungs')
    if(size(u,3)==1)
        M=double(u);
    else
        M=double(rgb2gray(u)); 
    end
   % pause(0.3);
    RC = reshape(M,m*n,1);
    AC = [AC,RC];
   avg = avg + RC;
   count = count + 1;
end
avgC=avg;
countC=count;
%avg = avg /count;


%% Load Normal
for j = 901:1000
   % figure(2)
    ff = ['COVIDData/Normal-',num2str(j,'%2d'),'.png'];
   u = imread(ff); % Read the image into a matrix
  %  imshow(u)
  %  title('Normal Lungs')
    M = double(u(:,:,1));
    
    RN = reshape(M,m*n,1);
    AN = [AN,RN];
  %  pause(0.3);
   avg = avg + RN;
   count = count + 1;
end
countN=count-countC;
avgN=avg-avgC;
avgC=avgC/countC;
avgAll = avg/count;
avgN=avgN/countN;

%% Center the sample pictures at the "origin"

A=[AC AN];% this is the entire data
A2=[AC AN];% this is the entire data
for j = 1:2*N
    A(:,j) = A(:,j) - avg;
end

%%  Computing the SVD
[U,S,V] = svd(A,'econ');
Phi = U(:,1:2*N);
Phi(:,1) = -1*Phi(:,1);
count = 1;

%% project each image onto basis 
for j = 1:N
    imvec = A(:,j);
    ARN(:,j) = imvec'*Phi(:,1:5);
end
for j = 1:N
    imvec = A(:,N+j);
    STAL(:,j) = imvec'*Phi(:,1:5);
end
%  ARN = COVID  P components; we will keep the first five components for each
%  images ARN(1,:),ARN(2,:),ARN(3,:), ARN(4,:),ARN(5,:),
% STAL= Normal P components; we will keep the first five components for each
%  images STAL(1,:),STAL(2,:),STAL(3,:), STAL(4,:),STAL(5,:),

mode=mode(A2);
mean=mean(A2);
median=median(A2);
std=std(A2);
range=range(A2);
%min=min(A2); no need for min, it is a functionof range and max
max=max(A2);
Cov=ones(N,1);
Nor=zeros(N,1);
O=[Cov;Nor];
PC=[ARN';STAL'];
x=1:2*N;
CRT_NNW_Data=[x',mode',mean',median',std',PC,O];

filename='MLdata.xlsx';
% xlswrite(filename,));
output_table = table(CRT_NNW_Data);
