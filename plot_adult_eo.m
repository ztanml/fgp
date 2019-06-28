load('data/adult.mat');

NA  = 30; % number of angles
ang = linspace(0,pi/2,NA);
Err = zeros(NA,1);
EO = zeros(NA,1);
sid = 10; % index of protected attr (9 - race, 10 - sex)
S   = x_train(:,sid);
xid = ones(size(x_train,2),1)==1;
xid(sid) = 0;
X   = x_train(:,xid);
y   = y_train;
XT  = x_test(:,xid);
ST  = x_test(:,sid);
yT  = y_test;

for a = 1:length(ang)
    eps = cos(ang(a));
    hyp = fgp(X,y,S,1,1,eps,...
          'efn','cov','covkfn','fgp_rbf','covkpar',8.7249,'fair','eo');
    pred = hyp.f(XT);
    aspe = fair_stats(pred,yT,ST);
    Err(a) = 1-aspe(1);
    EO(a) = aspe(4);
end

plot(EO,Err,'ro-');
hold on;

Mdl = fitrgp(X,y,'KernelFunction','squaredexponential');
yp  = predict(Mdl,XT);
aspe= fair_stats(yp,yT,ST);

scatter(aspe(4),1-aspe(1),50,'filled');

legend('FGP','GP');
grid on;
xlabel('EO');
ylabel('Prediction Error');
