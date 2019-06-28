load('data/adult.mat');

NA  = 30; % number of angles
ang = linspace(0,pi/2,NA);
SP_Err = zeros(NA,1);
SP_Cor = zeros(NA,1);
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
          'efn','cov','covkfn','fgp_rbf','covkpar',8.4389);
    pred = hyp.f(XT);
    SP_Err(a) = mean(sign(pred)~=yT);
    SP_Cor(a) = max(abs(corr(pred,ST)));
end

plot(SP_Cor,SP_Err,'ro-');
hold on;

Mdl = fitrgp(X,y,'KernelFunction','squaredexponential');
yp  = predict(Mdl,XT);

scatter(abs(corr(yp,ST)),mean(sign(yp)~=yT),50,'filled');

legend('FGP','GP');
grid on;
xlabel('SP');
ylabel('Prediction Error');
