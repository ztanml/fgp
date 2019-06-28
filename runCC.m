x_data=csvread('data/cc_x.csv');
x_data=zscore(x_data);
y_data=csvread('data/cc_y.csv');
s_data=csvread('data/cc_s.csv');

n  = 960;
NA = 30; % number of angles
m  = 4;
d  = 4;
band = 33.841;

x_train = x_data(1:n,:);
x_test  = x_data(n+1:end,:);
y_train = y_data(1:n);
y_test  = y_data(n+1:end);
s_train = s_data(1:n,:);
s_test  = s_data(n+1:end,:);

ang = linspace(0,pi/2,NA);
Err = zeros(NA,1);
SP  = zeros(NA,size(s_train,2));

for a = 1:length(ang)
    eps = cos(ang(a));
    hyp = fgp(x_train,y_train,s_train,m,d,eps,...
          'efn','cov','covkfn','fgp_rbf','covkpar',band);
    pred = hyp.f(x_test);
    Err(a) = norm(pred-y_test)/sqrt(length(pred));
    SP(a,:)= abs(corr(pred,s_test));
end

Mdl = fitrgp(x_train,y_train,'KernelFunction','squaredexponential');
yp  = predict(Mdl,x_test);
RMSE_GP = norm(yp-y_test)/sqrt(length(yp));
SP_GP = abs(corr(yp,s_test));

plot(SP(:,1),Err,'ro-');
hold on;
plot(SP(:,2),Err,'ro--');
scatter(SP_GP(:,1),RMSE_GP,50,'filled');
scatter(SP_GP(:,2),RMSE_GP,50,'*');

legend('FGP','GP');
set(gca,'fontsize',12);
grid on;
xlabel('SP');
ylabel('RMSE');
