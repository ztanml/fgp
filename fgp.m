function hyp = fgp(X,y,S,m,d,eps,varargin)
% Fair Gaussian Processes
%
% Input:
%    X, y, S are the n-by-p feature matrix, n-by-1 label vector, and n-by-*
%       protected attributes 
%    m, d specify the dimension of the SDR subspaces with respect to S and
%       y, respectively
%    eps is the fairness-accuracy trade-off and takes value in [0,1]: 1 for
%    total fairness and 0 for total accuracy
%
% Optional:
%    EM Parameters:
%        maxiter is the max number of EM iterations
%        tol specifies the minimum change in the per-instance log-likelihood
%           value between two consecutive EM iterations.
%
%    SDR Parameters:
%        ns  is the maximum number of slices, each slice corresponds to a range of y.
%           For classification, a slice contains one or more classes
%        eta is small postive number used to improve the condition of A
%
%    Mean/Covariance Functions:
%        lambda is the mean function regularization parameter
%        efn specifies the error function, 
%           'cov' - use covariance kernel, 'lin' - linear, 'ker' - kernel Ridge
%        meankfn/covkfn give the mean and covariance kernel functions
%
% Returns the model struct hyp. Some important members are
%    hyp.f is the fitted target function f: X -> [Y,varY]
%    hyp.mf is the fitted mean function mf: X -> Y
%    hyp.nlp is a vector of negative log likelihood
%
% Author: Zilong Tan (zilongt@cs.cmu.edu)
% Date: June 28th 2019

hyp = struct();
[n,p] = size(X);

opt = inputParser;
opt.addParameter( 'maxiter',   20,        @(x) floor(x) > 0 );
opt.addParameter( 'tol',       1e-8,      @(x) floor(x) >= 0);
opt.addParameter( 'eta',       1e-4,      @(x) floor(x) >= 0);
opt.addParameter( 'lambda',    0,         @(x) floor(x) >= 0);
% Specify the fairness condition, specify 'sp' for either sp or proxy nondiscrimination
opt.addParameter( 'fair',      'sp',      @(x) strcmp(x,'sp')|strcmp(x,'eop')|strcmp(x,'eo'));
opt.addParameter( 'efn',       'cov',     @(x) strcmp(x,'lin')|strcmp(x,'ker')|strcmp(x,'cov'));
opt.addParameter( 'meankfn',   @fgp_lin,  @(x) feval(x) >= 0); % used only if opt.efn == 'ker'
opt.addParameter( 'covkfn',    @fgp_rbf,  @(x) feval(x) >= 0);
opt.addParameter( 'meankpar',  [],        @(x) true);
opt.addParameter( 'covkpar',   1,         @(x) true);
opt.addParameter( 'normalize', false,     @(x) islogical(x));
opt.parse(varargin{:});
opt = opt.Results;    

hyp.opt = opt;

use_lin_efn = 0;
use_ker_efn = 0;
use_cov_efn = 0;
if strcmp(opt.efn,'lin')
    use_lin_efn = 1;
elseif strcmp(opt.efn,'cov')
    use_cov_efn = 1; % use a mean function in the covariance kernel RKHS
elseif strcmp(opt.efn,'ker')
    use_ker_efn = 1; % use a mean function the RKHS induced by meankfn
end

% Center the data, if needed
if opt.normalize
    hyp.Xmu  = mean(X);
    hyp.Xstd = max(std(X)*sqrt(p),1e-12);
else
    hyp.Xmu  = zeros(1,size(X,2));
    hyp.Xstd = ones(1,size(X,2));
end
X = X - hyp.Xmu;
X = X ./ hyp.Xstd;

meankfn = @(X,Z,param) feval(opt.meankfn,X,Z,param);
covkfn  = @(X,Z,param) feval(opt.covkfn,X,Z,param);
CK = covkfn(X,[],opt.covkpar);

% Compute the fair subspace (i.e., Q)
if strcmp(opt.fair,'eo')
    W = zeros(n,size(S,2)*2*m);
else
    W = zeros(n,size(S,2)*m);
end
wid = 0;
if strcmp(opt.fair,'eo') || strcmp(opt.fair,'eop')
    pid = (y==1);
    for i = 1:size(S,2)
        [~,W(pid,wid+1:wid+m),~] = sdr(CK(pid,pid),S(pid,i),m,'eta',opt.eta);
        wid = wid + m;
    end
    if strcmp(opt.fair,'eo')
        nid = (y~=1); % this support {0,1} and {-1,1}
        for i = 1:size(S,2)
            [~,W(nid,wid+1:wid+m),~] = sdr(CK(nid,nid),S(nid,i),m,'eta',opt.eta);
            wid = wid + m;
        end
    end
else
    for i = 1:size(S,2)
        [~,W(:,wid+1:wid+m),~] = sdr(CK,S(:,i),m,'eta',opt.eta);
        wid = wid + m;
    end
end
    
KGK = CK - mean(CK);
KGK = KGK'*KGK;
Q   = null(W'*KGK);

% Compute orthonormal fair basis
[M,Lbd,~] = svd(Q'*CK*Q);
Lbd = diag(Lbd);
LbdISQ = 1./sqrt(Lbd);

% Compute orthonormal accurate basis
[~,A,~] = sdr(CK,y,d,'eta',opt.eta);
[T,Omg,~] = svd(A'*CK*A);
Omg = diag(Omg);
OmgISQ = 1./sqrt(Omg);

% Compute canonical angles
[U,D,V] = svds(LbdISQ.*M'*Q'*CK*A*T.*OmgISQ',d);

% Compute model basis
FB = Q*M.*LbdISQ'*U;
AB = A*T.*OmgISQ'*V;
Sigmas = D(1:size(D,1)+1:end)';
Gammas = max(Sigmas,eps);
Rhos = sqrt((1-Gammas.^2)./(1-Sigmas.^2));
E    = FB.*(Gammas - Rhos.*Sigmas)' + AB.*Rhos';

%% Initialize other parameters
MCK = mean(CK);
KE  = CK*E;
PiX = KE - mean(KE);
PTP = PiX'*PiX;
beta = zeros(d,1);
err = y;
s2  = opt.eta;
G   = zeros(n,n);
iSb = zeros(d,d);

if use_cov_efn
    ETKE = KE'*E;
    efn = @(varargin) fgp_efn_cov(y,PiX,ETKE,varargin{:});
elseif use_ker_efn
    MK  = meankfn(X,[],opt.meankpar);
    MMK = mean(MK);
    efn = @(varargin) fgp_efn_ker(y,MK,varargin{:});
elseif use_lin_efn
    efn = @(varargin) fgp_efn_lin(y,X,varargin{:});
end

hyp.nlp = [];

%% EM iterations
for i = 1:opt.maxiter
    % Update the variance components
    Sv  = inv(PTP/s2 + iSb);
    beta = Sv/s2*PiX'*err;
    Sb  = beta*beta' + Sv;
    iSb = inv(Sb);
    res = err - PiX*beta;
    s2  = s2 + (sum(res.^2) - s2^2*trace(G))/n;
    % Update the mean function
    G   = compG(PiX*pdsqrtm(Sb),s2);
    alp = efn(G,opt.lambda);
    err = efn(alp);
    % Negative log-likelihood
    nlp = (log(2*pi)*(n+d) + pdlogdet(Sb) + n*log(s2) + ...
           beta'*iSb*beta + sum((res/sqrt(s2)).^2))/2/n;
    hyp.nlp = [hyp.nlp; nlp];
    if length(hyp.nlp) > 1 && hyp.nlp(end-1) - hyp.nlp(end) < opt.tol
        break;
    end
end

hyp.Basis   = E;
hyp.alpha   = alp;
hyp.beta    = beta;
hyp.SigBeta = Sb;   % estimate of beta variance
hyp.Pi      = PiX;  % SDR projection
hyp.s2      = s2;

MF = E*beta;
CF = E*sqrtm(Sv);

hyp.covkfn = @(Z) covkfn((Z-hyp.Xmu)./hyp.Xstd,X,opt.covkpar) - MCK;
if use_cov_efn
    hyp.mf = @(Z)-fgp_efn_cov(zeros(size(Z,1),1),hyp.covkfn(Z)*E,[],alp);    
elseif use_ker_efn
    hyp.mf = @(Z)-fgp_efn_ker(zeros(size(Z,1),1),...
                meankfn((Z-hyp.Xmu)./hyp.Xstd,X,opt.meankpar)-MMK,alp);    
elseif use_lin_efn
    hyp.mf = @(Z)-fgp_efn_lin(zeros(size(Z,1),1),(Z-hyp.Xmu)./hyp.Xstd,alp);    
end

hyp.f = @(Z) fgp_pred(hyp.covkfn(Z),hyp.mf(Z),MF,CF,s2);

end

function [pmu,pvar] = fgp_pred(KZ,muZ,MF,CF,s2)
pmu  = muZ + KZ*MF;
pvar = sum((KZ*CF).^2,2) + s2;
end

% Linear mean function
function val = fgp_efn_lin(y,X,G,lambda)
if nargin < 3, val = 'p+1'; return, end
if nargin == 3, val = y - X*G(2:end) - G(1); return, end
n = size(X,1);
p = size(X,2);
rs = sum(G);
ss = sum(sum(G));
GL = G - rs'/ss*rs;
if p > n
    % Dual estimator
    XTGL = X'*GL;
    KGL = X*XTGL;
    KGL(1:n+1:end) = KGL(1:n+1:end) + lambda;
    val = XTGL/KGL*y;
else
    CGL = X'*GL*X;
    CGL(1:p+1:end) = CGL(1:p+1:end) + lambda;
    val = CGL\(X'*GL*y);
end
val = [rs/ss*(y-X*val);val];
end

% Mean function based on covariance kernel
function val = fgp_efn_cov(y,PiX,S,G,lambda)
if nargin < 4, val = 'd+1'; return, end
if nargin == 4, val = y - PiX*G(2:end) - G(1); return, end
n  = size(G,1);
rs = sum(G);
ss = sum(rs);
PTGL= PiX'*G - PiX'*rs'/ss*rs;
val = (PTGL*PiX + n*lambda*S)\(PTGL*y);
val = [rs/ss*(y-PiX*val); val];
end

% Kernel Ridge mean function
function val = fgp_efn_ker(y,K,G,lambda)
if nargin < 3, val = 'n+1'; return, end
if nargin == 3, val = y - K*G(2:end) - G(1); return, end
n = size(K,1);
rs = sum(G);
ss = sum(rs);
GL = G - rs'/ss*rs;
GLK = GL*K;
GLK(1:n+1:end) = GLK(1:n+1:end) + lambda;
val = GLK\(GL*y);
val = [rs/ss*(y-K*val); val];
end

function G = compG(PiX,s2)
n = size(PiX,1);
m = size(PiX,2);
G = PiX'*PiX;
G(1:m+1:end) = G(1:m+1:end) + s2;
G = -PiX/G*PiX';
G(1:n+1:end) = G(1:n+1:end) + 1;
G = G/s2;
end

function MSQ = pdsqrtm(X)
[MSQ,S,~] = svd(X);
MSQ = MSQ.*sqrt(diag(S))'*MSQ';
end

function val = pdlogdet(X)
S = svd(X);
val = sum(log(S));
end

function SE = pairwise_sqerr(X,Z)
sqX = sum(X.^2,2);
sqZ = sum(Z.^2,2);
SE = bsxfun(@minus, sqX, (2*X)*Z');
SE = bsxfun(@plus, sqZ', SE);
SE = max(SE,0); % may be negative due to rounding errors
end

function K = fgp_poly(X,Z,param)
if nargin == 0 || isempty(X), K = 2; return, end
if nargin == 3
    X = X/param(1);
    if ~isempty(Z)
        Z = Z/param(1);
    else
        Z = X;
    end
end
K = (1 + X*Z').^param(2); % param(2) must be a positive integer
end

function K = fgp_lin(X,Z,param)
if nargin == 0 || isempty(X), K = 0; return, end
if nargin == 3 && isempty(Z), Z = X; end
K = X*Z';
end

% only applies to 1D X and Z
function K = fgp_per(X,Z,param)
if nargin == 0 || isempty(X), K = 2; return, end
if nargin == 3
    X = X*(pi/param(1));
    if ~isempty(Z)
        Z = Z*(pi/param(1));
    else
        Z = X;
    end
end
K = exp(-sin(X-Z').^2/param(2)^2);
end

function K = fgp_rbf(X,Z,band)
if nargin == 0 || isempty(X), K = 1; return, end
if nargin == 3
    X = X/band;
    if ~isempty(Z)
        Z = Z/band;
    else
        Z = X;
    end
end
% basically -pairwise_sqerr(X,Z), negate vectors rather than the matrix
sqX = -sum(X.^2,2);
sqZ = -sum(Z.^2,2);
K = bsxfun(@plus, sqX, (2*X)*Z');
K = bsxfun(@plus, sqZ', K);
K = exp(K);
end

% Rational quadratic kernel
% k(x,z) = (1 + (x-z)'(x-z)/alpha/band^2)^(-alpha)
function K = fgp_rq(X,Z,param)
if nargin == 0 || isempty(X), K = 1; return, end
if nargin == 3
    X = X/sqrt(param(1))/param(2);
    if ~isempty(Z)
        Z = Z/sqrt(param(1))/param(2);
    else
        Z = X;
    end
end
K = (1+pairwise_sqerr(X,Z)).^(-param(1));
end
