function [P,W,E] = sdr(K,y,m,varargin)
% Estimating an SDR subspace
% Input:
%    K - kernel matrix of input features
%    y - label vector
%    m - rank of the SDR subspace
% Returns the projection onto the SDR subspace (P), the projector (W), and
%    eigenvalues (E)
%
% Author: Zilong Tan (zilongt@cs.cmu.edu)
% Date: June 28th 2019

n = size(K,1);

opt = inputParser;
opt.addParameter( 'ns',  0,     @(x) floor(x) > 1 & floor(x) <= n/2); % 1 for auto select
opt.addParameter( 'eta', 1/n,   @(x) floor(x) >= 0);
opt.addParameter( 'inv', false, @(x) islogical(x));
opt.parse(varargin{:});
opt = opt.Results;

% Partition the data by response range
% This works for both regression and classification
[y,Idx] = sort(y,'ascend');
IdxInv = zeros(1,n);
IdxInv(Idx) = 1:n;
K = K(Idx,Idx);
[~,nun] = unique(y);
ns = opt.ns;
if ns == 0 % auto select
    ns = min(length(nun),floor(n/2));
end
nun = [nun(2:end)-nun(1:end-1); n+1-nun(end)];
if length(nun) <= ns
    csz = nun;
else
    csz(1,1) = 0;
    sz = n/ns;
    i = 1;
    for j = 1:length(nun)
        if csz(i,1) >= sz
            i = i + 1;
            csz(i,1) = 0;
        end
        csz(i,1) = csz(i,1) + nun(j);
    end
end

A   = zeros(n,n);
ns  = length(csz); % actual number of slices
pos = cumsum([1;csz]);
for i = 1:length(csz)
    idx = pos(i):pos(i+1)-1;
    A(idx,:) = K(idx,:) - mean(K(idx,:));
end
C = K; C = C - mean(C);

if opt.inv
    C(1:n+1:end) = C(1:n+1:end) + n*opt.eta;
    [W,E] = eigs(A,C,m);
else
    A(1:n+1:end) = A(1:n+1:end) + n*opt.eta;
    [W,E] = eigs(C,A,m);
end

P = (K-mean(K))*W;
P = P(IdxInv,:);
W = W(IdxInv,:);
E = 1-1./diag(E);

end
