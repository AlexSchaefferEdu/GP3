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
    figure(1)
    ff = ['COVIDData/COVID-',num2str(j,'%2d'),'.png'];
    u = imread(ff); % Read the image into a matrix
    imshow(u)
    title('COVID Infected Lungs')
    if(size(u,3)==1)
        M=double(u);
    else
        M=double(rgb2gray(u)); 
    end
    pause(0.3);
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
    figure(2)
    ff = ['COVIDData/Normal-',num2str(j,'%2d'),'.png'];
    u = imread(ff); % Read the image into a matrix
    imshow(u)
    title('Normal Lungs')
    M = double(u(:,:,1));
    
    RN = reshape(M,m*n,1);
    AN = [AN,RN];
    pause(0.3);
   avg = avg + RN;
   count = count + 1;
end
countN=count-countC;
avgN=avg-avgC;
avgC=avgC/countC;
avgAll = avg/count;
avgN=avgN/countN;
%% Calculate the "averaged" COVID
avgTSC = uint8(reshape(avgC,m,n));
figure(10), imshow(avgTSC);
title('Average of sample COVID images')
%% Calculate the "averaged" Normal
avgTSN = uint8(reshape(avgN,m,n));
figure(11), imshow(avgTSN);
title('Average of sample Normal images')
%% Calculate the "averaged" both COVID & Normal
avgTSB = uint8(reshape(avgAll,m,n));
figure(12), imshow(avgTSB);
title('Average of Normal & COVID images')
%% Center the sample pictures at the "origin"
%figure(2)
A=[AC AN];
for j = 1:2*N
    A(:,j) = A(:,j) - avg;
    %R = reshape(A(:,j),m,n);
    %imshow(R);
    %set(gcf,'Position',[100 900 500 500])    
    %pause(.3);
end

%%  Computing the SVD
[U,S,V] = svd(A,'econ');
Phi = U(:,1:2*N);
Phi(:,1) = -1*Phi(:,1);
%figure(2)
count = 1;
% for i=1:3
%     for j=1:3
%         subplot(3,3,count)
%         imshow(uint8(25000*reshape(Phi(:,count),m,n)));
%         count = count + 1;
%     end
% end


%% project each image onto basis 
for j = 1:N
    imvec = A(:,j);
    ARN(:,j) = imvec'*Phi(:,1:3);
end
for j = 1:N
    imvec = A(:,N+j);
    STAL(:,j) = imvec'*Phi(:,1:3);
end

figure(3)

plot3(ARN(1,:),ARN(2,:),ARN(3,:),'r.','MarkerSize',30)
hold on
plot3(STAL(1,:),STAL(2,:),STAL(3,:),'b.','MarkerSize',30)
xlabel('PCA 1')
ylabel('PCA 2')
zlabel('PCA 3')
legend('COVID','Normal')

%% add some test images of CoVID (red +) and Normal (blue *)
u = imread('COVIDData/testCovid.png');        
figure(4)
subplot(1,2,1)
imshow(u);
title('Normal Lungs')
if(size(u,3)==1)
        u=double(u);
    else
        u=double(rgb2gray(u)); 
    end
ustal = reshape(u,m*n,1)-avg;
stalpts = ustal'*Phi(:,1:3);
v = imread('COVIDData/testNormal.png');
subplot(1,2,2)
imshow(v);
title('COVID infected lungs')
if(size(v,3)==1)
        v=double(v);
    else
        v=double(rgb2gray(v)); 
    end
vterm = reshape(v,m*n,1)-avg;
termpts = vterm'*Phi(:,1:3);
%%
figure(3)
plot3(stalpts(1),stalpts(2),stalpts(3),'r+','MarkerSize',20)
plot3(termpts(1),termpts(2),termpts(3),'b*','MarkerSize',20)
grid on
close(figure(1))
close(figure(2))


%% here we compute basic stats of each dataset
sn=size(AN);
Ndata = reshape(AN,sn(1)*sn(2),1);
%Basic stats of Normal lung dataset
meanN=mean(Ndata)
minN=min(Ndata)
maxN=max(Ndata)
stdN=std(Ndata)
sc=size(AC);
Cdata = reshape(AC,sc(1)*sc(2),1);
%Basic stats of COVID infected lung dataset
meanC=mean(Cdata)
minC=min(Cdata)
maxC=max(Cdata)
stdC=std(Cdata)

%% boxplot of datasets
figure(13)
boxplot([avgN,avgC],'Notch','on','Labels',{'Normal','COVID'})
%title(' ')
xlabel('COVID vurses Normal')
ylabel('Average values of grayscale images')

