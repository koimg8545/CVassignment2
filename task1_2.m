image = imread("images/wall/im1.pgm");

%% 
% task1 start

G = fspecial('gaussian',5,0.5);
im_g = imfilter(image,G);


%%
% gradient filter (7*7 Prewitt filter)

filt = [3 2 1 0 -1 -2 -3];
filty = repmat(filt,7,1);
filtx = transpose(filty);

%%
% set the values within 10 range from edges as 0, 
% for removing abnormal values from zero padding

grad_x = filter2(filtx, im_g);
grad_y = filter2(filty, im_g);
grad_x(1:10,:) = 0;
grad_x(end-10:end,:) = 0;
grad_x(:,1:10) = 0;
grad_x(:,end-10:end) = 0;

grad_y(1:10,:) = 0;
grad_y(end-10:end,:) = 0;
grad_y(:,1:10) = 0;
grad_y(:,end-10:end) = 0;
%%
imshow(rescale(grad_y))
%%
% making structure tensor using sum filter

sumfilt = ones(21,21)

grad_x_pow = grad_x.^2;
grad_y_pow = grad_y.^2;
grad_xy = grad_x.*grad_y;

grad_x_pow_sum = filter2(sumfilt, grad_x_pow);
grad_y_pow_sum = filter2(sumfilt, grad_y_pow);
grad_xy_sum = filter2(sumfilt, grad_xy);
%%
% Harris corner detector

det = grad_x_pow_sum .* grad_y_pow_sum - grad_xy_sum.^2;
trace = grad_x_pow_sum + grad_y_pow_sum;
k = 0.05
R = det - k*(trace.^2);
%%
rescale_R = rescale(R);
%%
% get 1000 strong corner points using proper threshold

sum(sum(rescale_R > 0.5422))
features = rescale_R > 0.5422;
strong_features = rescale_R .* features;

%%
% task1 end

imshow(strong_features)

%%
% task2 start
% make padding for nms

col_padding = zeros(size(image,1),10)
padding_features = [col_padding, strong_features, col_padding]
row_padding = zeros(10, size(padding_features, 2))
padding_features = [row_padding; padding_features; row_padding]
    
%%
% do nms using binary disk filter

nms_features = zeros(size(image))
circle = fspecial('disk',10);
circle_binary = circle>0;
for row = 1:size(image,1)
    for col = 1:size(image,2)
        nms_features(row, col) = nms(padding_features,row+10,col+10,10,circle_binary);
    end
end

%%
% show the locations of nms_features

[row1,col1] = find(nms_features);

for i = 1:size(row1,1)
    nms_features(row1(i)-5:row1(i)+5,col1(i)-5:col1(i)+5) = 1;
end

imshow(nms_features)
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