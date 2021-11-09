image1 = rescale(imread("images/wall/im1.pgm"));
image2 = rescale(imread("images/wall/im2.pgm"));
G = fspecial('gaussian',5,0.5);
image1 = imfilter(image1,G);
image2 = imfilter(image2,G);

%%
% task 3
% get eigen vectors

[features1, eigen1] = eigen_vectors(image1, 0.5422);
[row1,col1] = find(features1);
[features2, eigen2] = eigen_vectors(image2, 0.49725);
[row2,col2] = find(features2);

%%
% crop rotate features using location of features and eigen vectors

rotate_features1 = rotate(image1,eigen1,features1)
rotate_features2 = rotate(image2,eigen2,features2)
%%
% get SSD and NCC using cropped features

ssd = [];
ncc = [];
for i = 1:size(rotate_features1,3)
    for j = 1:size(rotate_features2,3)
        ssd(i,j) = sum(sum((rotate_features1(:,:,i) - rotate_features2(:,:,j)).^2));
        mean1 = mean(rotate_features1(:,:,i) ,'all');
        mean2 = mean(rotate_features2(:,:,j),'all');
        std1 = std(rotate_features1(:,:,i),0,'all');
        std2 = std(rotate_features2(:,:,j),0,'all');
        ncc(i,j) = (1/(21*21)) * sum((rotate_features1(:,:,i) - mean1) .* (rotate_features2(:,:,j) - mean2),'all') / (std1*std2);
    end
end


%%
ssd_im = imresize(ssd, 100,'nearest');
imshow(rescale(ssd_im))

%%
ncc_im = imresize(ncc, 100,'nearest');
imshow(rescale(ncc_im))

%%
% drop unmatching features using threshold and match features
col_filt_ssd = ssd(:,min(ssd,[],1)<40);
row2 = row2(min(ssd,[],1)<40,:);
col2 = col2(min(ssd,[],1)<40,:);
[feat1_index, feat2_index] = find(col_filt_ssd == min(col_filt_ssd,[],1));
filt_ssd = col_filt_ssd(feat1_index,:);
row1 = row1(feat1_index);
col1 = col1(feat1_index);
imshow(rescale(imresize(filt_ssd, 100,'nearest')));

%%
% show matching features

imshow(image1)
hold on
for i = 1:size(col1)
    text(col1(i), row1(i), num2str(i),'color','red', 'fontsize', 20)
end
hold off
%%
imshow(image2)
hold on
for i = 1:size(col2)
    text(col2(i), row2(i), num2str(i),'color','red', 'fontsize', 20)
end
hold off

%%
function newframe = rotate(image, eigen,features)
    newframe = [];
    [row, col] = find(features);
    
    % get cos and sin from eigen vectors
    for i = 1:size(eigen,2)
        cos = eigen(1,i);
        sin = sqrt(1-cos^2);
        if(eigen(2,i) < 0)
            sin = -sin;
        end
        % rotate features using rotation matrices 
        rot = [cos sin; -sin cos];
        tform = rigid2d(rot,[0 0]);
        rotate_im = imwarp(image, tform);
        
        % add padding for preventing error from cropping
        rotate_im = [zeros(size(rotate_im,1),10) rotate_im zeros(size(rotate_im,1),10)];
        rotate_im = [zeros(10, size(rotate_im,2)); rotate_im; zeros(10, size(rotate_im,2))];
        
        % find location of rotated feature using imwarp
        feat_loc = zeros(size(features,1),size(features,2));
        feat_loc(row(i),col(i)) = 1;
        rotate_feat = imwarp(feat_loc, tform);
        [rot_row, rot_col] = find(rotate_feat);
        
        % crop rotated features
        newframe(:,:,i) = rotate_im(rot_row:rot_row+20, rot_col:rot_col+20);
    end
end
%%
function [features, eigen_vec] = eigen_vectors(image, t)
    features = harris(image, t);
    % get structure tensor
    filt = [3 2 1 0 -1 -2 -3];
    filty = repmat(filt,7,1);
    filtx = transpose(filty);
    grad_x = filter2(filtx, image);
    grad_x(1:10,:) = 0;
    grad_x(end-10:end,:) = 0;
    grad_x(:,1:10) = 0;
    grad_x(:,end-10:end) = 0;
    grad_y = filter2(filty, image);
    grad_y(1:10,:) = 0;
    grad_y(end-10:end,:) = 0;
    grad_y(:,1:10) = 0;
    grad_y(:,end-10:end) = 0;
    sumfilt = ones(21,21);
    grad_x_pow = grad_x.^2;
    grad_y_pow = grad_y.^2;
    grad_xy = grad_x.*grad_y;
    grad_x_pow_sum = filter2(sumfilt, grad_x_pow);
    grad_y_pow_sum = filter2(sumfilt, grad_y_pow);
    grad_xy_sum = filter2(sumfilt, grad_xy);
    
    % find location of features
    [row, col] = find(features);
    
    % get eigen vector from each feature
    eigen_vec = zeros(2, sum(sum(features>0)));
    for i = 1:sum(sum(features>0))
        a = grad_x_pow_sum(row(i),col(i));
        d = grad_y_pow_sum(row(i),col(i));
        bc = grad_xy_sum(row(i),col(i));
        [v,D] = eigs([a bc; bc d],1);
        eigen_vec(:,i) = v;
    end
end

%%
function nms_features = harris(image, t)
    % Prewitt filter
     filt = [3 2 1 0 -1 -2 -3];
     filty = repmat(filt,7,1);
     filtx = transpose(filty);
     
    % get gradient
    grad_x = filter2(filtx, image);
    grad_y = filter2(filty, image);
    grad_x(1:10,:) = 0;
    grad_x(end-10:end,:) = 0;
    grad_x(:,1:10) = 0;
    grad_x(:,end-10:end) = 0;
    grad_y(1:10,:) = 0;
    grad_y(end-10:end,:) = 0;
    grad_y(:,1:10) = 0;
    grad_y(:,end-10:end) = 0;
    
    % structer tensor using sum filter
    sumfilt = ones(21,21);

    grad_x_pow = grad_x.^2;
    grad_y_pow = grad_y.^2;
    grad_xy = grad_x.*grad_y;

    grad_x_pow_sum = filter2(sumfilt, grad_x_pow);
    grad_y_pow_sum = filter2(sumfilt, grad_y_pow);
    grad_xy_sum = filter2(sumfilt, grad_xy);

    % Harris corner detector
    det = grad_x_pow_sum .* grad_y_pow_sum - grad_xy_sum.^2;
    trace = grad_x_pow_sum + grad_y_pow_sum;
    k = 0.05;
    R = det - k*(trace.^2);
    
    % get strong features
    rescale_R =rescale(R);
    features = rescale_R > t;
    strong_features = rescale_R .* features;

    disp(sum(sum(strong_features>0)))
    
    % make padding for nms
    col_padding = zeros(size(image,1),10);
    padding_features = [col_padding, strong_features, col_padding];
    row_padding = zeros(10, size(padding_features, 2));
    padding_features = [row_padding; padding_features; row_padding];

    % do nms
    nms_features = zeros(size(image));
    circle = fspecial('disk',10);
    circle_binary = circle>0;
    for row = 1:size(image,1)
        for col = 1:size(image,2)
            nms_features(row, col) = nms(padding_features,row+10,col+10,10,circle_binary);
        end
    end
end
%% 
function ret = nms(image, row, col, radius, circle)
   frame = image(row-radius:row+radius, col-radius:col+radius);
   frame = frame .* circle;
   if max(max(frame)) == image(row,col)
       ret = image(row,col);
   else
       ret = 0;
   end
end