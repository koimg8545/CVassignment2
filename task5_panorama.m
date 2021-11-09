image1 = rescale(imread("images/panorama/uttower_right.jpg"));
image2 = rescale(imread("images/panorama/uttower_left.jpg"));
% convert color images to gray images
image1 = im2gray(image1);
image2 = im2gray(image2);
G = fspecial('gaussian',5,0.5);
image1 = imfilter(image1,G);
image2 = imfilter(image2,G);
image1 = histeq(image1);
image2 = histeq(image2);

%%
% task 4
% same with task3, get SSD, NCC and matching features using SSD

[features1, eigen1] = eigen_vectors(image1, 0.1712);
[row1,col1] = find(features1);
feat_vec1 = [row1 col1];
[features2, eigen2] = eigen_vectors(image2, 0.19212);
[row2,col2] = find(features2);

rotate_features1 = rotate(image1,eigen1,features1);
rotate_features2 = rotate(image2,eigen2,features2);

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
% original ssd
ssd_im = rescale(ssd);
ssd_im = rescale(imresize(ssd, 10,'nearest'));
imshow(ssd_im);

%%
% filtered ssd
col_filt_ssd = ssd(:,min(ssd,[],1)<15);
row2 = row2(min(ssd,[],1)<15,:);
col2 = col2(min(ssd,[],1)<15,:);
[feat1_index, feat2_index] = find(col_filt_ssd == min(col_filt_ssd,[],1))
filt_ssd = col_filt_ssd(feat1_index,:)
row1 = row1(feat1_index)
col1 = col1(feat1_index)
imshow(rescale(imresize(filt_ssd, 20,'nearest')))

%%
% show location of matching features

imshow(image1)
hold on
for i = 1:size(col1)
    text(col1(i), row1(i), num2str(i),'color','red', 'fontsize', 15)
end
hold off

%%
imshow(image2)
hold on
for i = 1:size(col2)
    text(col2(i), row2(i), num2str(i),'color','red', 'fontsize', 15)
end
hold off

%%
% get transform matrix, number of inlier using homography RANSAC
[trans_mat_iter, match_count_iter, max_inliner, trans_mat, feat_index] = homography_ransac(row1, col1, row2, col2, 10000);
%%
% reload color images
image1 = rescale(imread("images/panorama/uttower_right.jpg"));
image2 = rescale(imread("images/panorama/uttower_left.jpg"));

G = fspecial('gaussian',5,0.5);
image1 = imfilter(image1,G);
image2 = imfilter(image2,G);

tform = projective2d(transpose(trans_mat));
homography_im = imwarp(image2, tform);
disp(max_inliner)
imshow(homography_im)

%%
% overlap images on the big image

% get matched features for translate image and overlapping
domain_feat_index = feat_index;
range_feat_index = feat_index;

% make big image and a matrix that checking overlapping
big_image = zeros(1500,2500,3);
check_overlap = zeros(1500,2500,3);
imshow(big_image)

% find a location of the feature in transformed image, for translating
feature_image2 = zeros(size(image2));
feature_image2(row2(range_feat_index), col2(range_feat_index)) = 1;
feature_image2 = imwarp(feature_image2, tform);

% find a location of the feature in original image
[feature_row2, feature_col2] = find(feature_image2);
[feature_row1, feature_col1] = deal(row1(range_feat_index), col1(range_feat_index));

% get displacement for overlapping
translate_row = feature_row2(1) - feature_row1;
translate_col = feature_col2(1) - feature_col1;

% make panorama using displacement
big_image(300:299+size(homography_im,1),300:299+size(homography_im,2),:) = homography_im;
check_overlap(big_image>0) = 1;

big_image(300+translate_row:299+translate_row+size(image1,1),...
                300+translate_col:299+translate_col+size(image1,2),:) = ... 
                big_image(300+translate_row:299+translate_row+size(image1,1),...
                300+translate_col:299+translate_col+size(image1,2),:) + image1;
            
check_overlap(300+translate_row:299+translate_row+size(image1,1),...
                300+translate_col:299+translate_col+size(image1,2),:) = ...
                check_overlap(300+translate_row:299+translate_row+size(image1,1),...
                300+translate_col:299+translate_col+size(image1,2),:) + 1;

% checking overlapped area and divide the value in half
big_image(check_overlap==2) = big_image(check_overlap==2)/2;

imshow(big_image)


%%
function [trans_mat_iter, match_count_iter, max_inlier, affine_mat, feat_index] = affine_ransac(row1, col1, row2, col2, iter)
    % save transform matrices, matched features, 
    % index of matched feature in each iteration
    trans_mat_iter = [];
    match_count_iter = [];
    feat_index_iter = [];
    for i = 1:iter
        % randomly choose 3 matching features
        points = randperm(size(row1,1), 3);
        [x_r1, x_r2, x_r3] = deal(col1(points(1)), col1(points(2)), col1(points(3)));
        [y_r1, y_r2, y_r3] = deal(row1(points(1)), row1(points(2)), row1(points(3)));

        [x_d1, x_d2, x_d3] = deal(col2(points(1)), col2(points(2)), col2(points(3)));
        [y_d1, y_d2, y_d3] = deal(row2(points(1)), row2(points(2)), row2(points(3)));

        % formula for obtaining affine matrix
        domain = [x_d1 y_d1 0 0 1 0;...
                        0 0 x_d1 y_d1 0 1;...
                        x_d2 y_d2 0 0 1 0;...
                        0 0 x_d2 y_d2 0 1;...
                        x_d3 y_d3 0 0 1 0;...
                        0 0 x_d3 y_d3  0 1];

        range = [x_r1; y_r1; x_r2; y_r2; x_r3; y_r3];

        trans_var = domain\range;
        trans_mat = [trans_var(1) trans_var(2) trans_var(5);...
                            trans_var(3) trans_var(4) trans_var(6);...
                            0 0 1];

        % transform locations of features for checking 
        % whether the point is matching and counting
        trans_vec = trans_mat * transpose([col2 row2 ones(size(row2, 1),1)]);
        trans_vec = trans_vec(1:2,:);
        [trans_row, trans_col] = deal(trans_vec(2,:), trans_vec(1,:));

        % count number of inliers using error threshold
        match_count=0;
        for j = 1:size(trans_vec,2)
            error = sqrt((row1(j) - trans_row(j))^2 + (col1(j) - trans_col(j))^2);
            if error < 2
                match_count = match_count+1;
                feat_index_iter(i) = points(1);
            end
        end
        % save matrix and inlier count
        trans_mat_iter(:,:,i) = trans_mat;
        match_count_iter(i) = match_count;
    end
    % get max inlier number, index, affine matrix, index of matched feature
    max_inlier = max(match_count_iter);
    max_index = find(match_count_iter==max_inlier);
    affine_mat = trans_mat_iter(:,:,max_index(1));
    feat_index = feat_index_iter(max_index(1));
end

%%
function [trans_mat_iter, match_count_iter, max_inner, homography_mat, feat_index] = homography_ransac(row1, col1, row2, col2, iter)
    % save transform matrices, matched features, 
    % index of matched feature in each iteration
    trans_mat_iter = [];
    match_count_iter = [];
    feat_index_iter = [];
    for i = 1:iter
        % randomly choose 4 matching features
        points_r = randperm(size(row1,1), 4);
        [x_r1, x_r2, x_r3, x_r4] = deal(col1(points_r(1)), col1(points_r(2)), col1(points_r(3)),col1(points_r(4)));
        [y_r1, y_r2, y_r3, y_r4] = deal(row1(points_r(1)), row1(points_r(2)), row1(points_r(3)), row1(points_r(4)));

        [x_d1, x_d2, x_d3, x_d4] = deal(col2(points_r(1)), col2(points_r(2)), col2(points_r(3)), col2(points_r(4)));
        [y_d1, y_d2, y_d3, y_d4] = deal(row2(points_r(1)), row2(points_r(2)), row2(points_r(3)), row2(points_r(4)));

        % formula for obtaining homography matrix
        domain = [x_d1 y_d1 1 0 0 0 -x_d1*x_r1 -y_d1*x_r1;...
                        x_d2 y_d2 1 0 0 0 -x_d2*x_r2 -y_d2*x_r2;...
                        x_d3 y_d3 1 0 0 0 -x_d3*x_r3 -y_d3*x_r3;...
                        x_d4 y_d4 1 0 0 0 -x_d4*x_r4 -y_d4*x_r4;...
                        0 0 0 x_d1 y_d1 1 -x_d1*y_r1 -y_d1*y_r1;...
                        0 0 0 x_d2 y_d2 1 -x_d2*y_r2 -y_d2*y_r2;...
                        0 0 0 x_d3 y_d3 1 -x_d3*y_r3 -y_d3*y_r3;...
                        0 0 0 x_d4 y_d4 1 -x_d4*y_r4 -y_d4*y_r4];
                        
        range = [x_r1; x_r2; x_r3; x_r4; y_r1; y_r2; y_r3; y_r4;];

        trans_var = domain\range;
        trans_mat = [trans_var(1) trans_var(2) trans_var(3);...
                            trans_var(4) trans_var(5) trans_var(6);...
                            trans_var(7) trans_var(8) 1];

        % transform locations of features for checking 
        % whether the point is matching and counting
        trans_vec = trans_mat * transpose([col2 row2 ones(size(row2, 1),1)]);
        adjust_one = repmat(trans_vec(3,:),3,1);
        trans_vec = trans_vec ./ adjust_one;
        trans_vec = trans_vec(1:2,:);
   
        [trans_row, trans_col] = deal(trans_vec(2,:), trans_vec(1,:));
        
        % count number of inliers using error threshold
        match_count=0;
        for j = 1:size(trans_vec,2)
            error = sqrt((row1(j) - trans_row(j))^2 + (col1(j) - trans_col(j))^2);
            if error < 2
                match_count = match_count+1;
                feat_index_iter(i) = points_r(1);
            end
        end
        % save matrix and inlier count
        trans_mat_iter(:,:,i) = trans_mat;
        match_count_iter(i) = match_count;
    end
    % get max inlier number, index, homography matrix, index of matched feature
    max_inner = max(match_count_iter);
    max_index = find(match_count_iter==max_inner);
    homography_mat = trans_mat_iter(:,:,max_index(1));
    feat_index = feat_index_iter(max_index(1));
end

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