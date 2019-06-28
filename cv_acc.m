function hyp = cv_acc(X,y,S,m,d,eps,k)

band  = optimizableVariable('band',[1,1e2],'Transform','log');

minfn = @(z) objfn(k,X,y,S,m,d,eps,z.band);

res = bayesopt(minfn,band,'IsObjectiveDeterministic',true,...
    'AcquisitionFunctionName','expected-improvement-plus', ...
    'ExplorationRatio',0.5,'MaxObjectiveEvaluations',30);

hyp = fgp(X,y,S,m,d,eps,'efn','cov','covkfn','fgp_rbf',...
      'covkpar',res.XAtMinObjective.band,'fair','eo');

end

function loss = objfn(k,X,y,S,m,d,eps,band)
c    = cvpartition(length(y),'kFold',k);
obj  = @(xT,yT,xt,yt) partLoss(xT,yT(:,1),yT(:,2:end),...
        xt,yt(:,1),yt(:,2:end),m,d,eps,band);
loss = crossval(obj,X,[y S],'partition',c);
loss = mean(loss);
end

function loss = partLoss(xT,yT,sT,xt,yt,st,m,d,eps,band)
hyp  = fgp(xT,yT,sT,m,d,eps,'efn','cov','covkfn','fgp_rbf',...
    'covkpar',band,'fair','eo');
loss = norm(hyp.f(xt)-yt)/sqrt(length(yt));
end
