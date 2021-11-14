% function skw = compute_sketch(file, c, k, r, eta)
% This function compute the empirical sketch of some given patches
% file: the file contains the patches to be compressed
% c: to define the size of the sketch
% k: the number of model components
% r: the desired rank
% eta: the scale parameter to design the frequencies

function skw = compute_sketch(file, c, k, r, eta)
    load(file, 'patches');
    [P, n] = size(patches);
    m = c*k*(P*r + 1);
    skw = sketch_wrapper(patches, m, 'eta', eta);
    disp('computing the sketch...');
%     for exemple n = 4e6
    s_1 = fix(n/500);
%     preferred number of workers in a parallel pool: 12
    skw.set_sketch(patches(:,1:s));
    s_2 = 492;
    parfor i = 1:s_2
        skw.update_sketch(patches(:,i*s_1+1:(i+1)*s_1));
    end
    skw.update_sketch(patches(:,(s_2+1)*s_1+1:end));
    save('compressed_patches.mat','skw');
end
