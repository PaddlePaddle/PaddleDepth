function [im, gau] = add_noise_lr(im)
    %im_input = imresize(im, 1/scale, 'bicubic');
    im_input = uint8(im(:,:,1));
    temp = 651.0;
    [r,c] = size(im_input);
    mu = zeros(r*c,1);
    im_vect = reshape(im_input,r*c,1);
    sigma = sqrt(temp./double(im_vect));
    im_gau = normrnd(mu,sigma);
    indx = find(im_gau == inf);
    im_gau(indx) = 0;
    indx = find(im_gau == -inf);
    im_gau(indx) = 0;
    im_gau = reshape(im_gau,r,c);
    im_input = double(im_input) + im_gau;
    %im_out = zeros(r,c,3);
    %im_out(:,:,1) = im_input;
    %im_out(:,:,2) = im_input;
    %im_out(:,:,3) = im_input;
    %im_input = imresize(uint8(im_input), scale, 'bicubic');
    im = uint8(im_input);
    gau = im_gau;
end