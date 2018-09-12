% MIT License
% 
% Copyright (c) 2018 Andri Ashfahani
% 
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

% eT2QFNN DESCRIPTION
% 
% This is the code of evolving fuzzy neural network, namely evolving Type-2
% Quantum Fuzzy Neural Network (eT2QFNN), which features an interval type-2 
% quantum fuzzy set with uncertain jump positions. The quantum fuzzy set 
% possesses a graded membership degree which enables better identification
% of overlaps between classes. The eT2QFNN works fully in the online mode 
% where all parameters including the number of rules are automatically 
% adjusted and generated on the fly. The parameter adjustment scenario
% relies on decoupled extended Kalman filter method.

% CITATION REQUEST
% 
% Please if you use the code for any purpose,  include the provided references and acknowledge the corresponding author.
% @article{DBLP:journals/corr/abs-1805-07715,
%   author    = {Andri Ashfahani and
%                Mahardhika Pratama and
%                Edwin Lughofer and
%                Qing Cai and
%                Huang Sheng},
%   title     = {An Online {RFID} Localization in the Manufacturing Shopfloor},
%   journal   = {CoRR},
%   volume    = {abs/1805.07715},
%   year      = {2018},
%   url       = {http://arxiv.org/abs/1805.07715},
%   archivePrefix = {arXiv},
%   eprint    = {1805.07715},
%   timestamp = {Mon, 13 Aug 2018 16:47:53 +0200},
%   biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1805-07715},
%   bibsource = {dblp computer science bibliography, https://dblp.org}
% }

function [y,rule,network,performance] = eT2QFNN(data,I,lr,mode,fix_themodel)
tic
disp('eT2QFNN: running ...');
[N,mn] = size(data);                            %number of input
M = mn-I;                                       %number of output
K = 1;                                          %number of rule
nrule = zeros(N,1);
nrule(1,1) = K;
x=data(:,1:I);
z=data(:,(I+1):mn);
ns=round(0.5*I);
if ns == 1
    ns = 2;
end
kk=randperm(fix_themodel,30);
gmixture = fitgmdist(data(kk,1:I),2,'CovarianceType','diagonal');    %this is matlab function of gaussian mixture model

%% Matrix initialization
yi = zeros(1,ns);
yu = zeros(1,ns);
Ql = zeros(I,K);
Qu = zeros(I,K);
Ru = zeros(1,K);
Rl = zeros(1,K);
wr = zeros(1,N);
phi_low = zeros(M,K);
phi_up = zeros(M,K);
y  = zeros(N,M);
out = y;
e  = zeros(N,M);
dRlm   = zeros(1,ns);
dRum   = zeros(1,ns);
dRldtl = zeros(I,ns);
dRudtu = zeros(I,ns);
dRldm  = zeros(1,I);
dRudm  = zeros(1,I);
dydw_low = zeros(I+1,M);
dydw_up  = zeros(I+1,M);
dydql  = zeros(1,M);
dydqr  = zeros(1,M);
dyldRl = zeros(1,M);
dyldRu = zeros(1,M);
dyrdRl = zeros(1,M);
dyrdRu = zeros(1,M);
dymdm  = zeros(I,1);
dymdtl = zeros(I,ns);
dymdtu = zeros(I,ns);

%% initial DEKF parameter
Z = 2*M*(2+I)+I*(2*ns+1);
P(:,:,K) = eye(Z);                      %initial covariance matrix for each rule
R = lr*eye(M);                       %learning rate

%% GMM Parameter Estimation
gmm.k = gmixture.NumComponents;
gmm.weight = gmixture.ComponentProportion;
gmm.miu = gmixture.mu';
for model = 1:gmm.k
    gmm.var(:,:,model) = diag(gmixture.Sigma(:,:,model));
end

%% Network parameter
[~,gmm.index] = max(gmm.weight,[],2);
center = zeros(I,1);
var = zeros(I);
for model = 1:gmm.k
    center = center + gmm.weight(model)*gmm.miu(:,model);
    var = var + gmm.weight(model)*gmm.var(:,:,model);
end
m = x(1,:)';
grow = 1;
fprintf('The new rule no 1 is FORMED around sample 1\n')
if mode == 1
    b = 10;                                                      %slope coefficient
elseif mode == 2
    b = 5;
elseif mode == 3
    b = 1;
end
sigma_up(:,:,K) = var;
sigma_low(:,:,K) = 0.7*sigma_up(:,:,K); %0.49
D_up  = diag(sqrt(sigma_up));
D_low = 0.7*D_up;
nr = 1:ns;
theta_up  = zeros(I,ns,K);
theta_low = zeros(I,ns,K);
for i = 1:I
    theta_up (i,:,K) = nr*D_up (i,K)/((ns+1)/2);        %initial quantum interval lower K x ns matrix
    theta_low(i,:,K) = nr*D_low(i,K)/((ns+1)/2);
end
omega_up  = 0.6*ones(M,I+1,K);                                        %initial omega up value K x I+1 x M vector
omega_low = 0.4*ones(M,I+1,K);                                        %initial omega low value K x I+1 x M vector
nomega_up(K) = norm(omega_up (:,:,K));
nomega_low(K)= norm(omega_low(:,:,K));
qleft  = 0.4*ones(1,M)';                                              %initial design factor ql 1 x K vector
qright = 0.7*ones(1,M)';                                              %initial design factor qr 1 x K vector
index = 1;

%% feedforward data iteration
for k = 1:N
    %% create hypothetical rule
    if k <= fix_themodel
        
        m(:,K+1) = x(k,:)';
        sigma_up (:,:,K+1) = diag(diag((x(k,:)-center')'*(x(k,:)-center')));
        sigma_low(:,:,K+1) = 0.7*sigma_up (:,:,K+1);
        for o = 1:M
            omega_up (o,:,K+1) = omega_up (o,:,index);
            omega_low(o,:,K+1) = omega_low(o,:,index);
        end
        D_up  = diag(sqrt(sigma_up(:,:,K+1)));
        D_low = 0.7*D_up;
        for i = 1:I
            theta_up (i,:,K+1) = nr*D_up (i)/((ns+1)/2);        %initial quantum interval lower K x ns matrix
            theta_low(i,:,K+1) = nr*D_low(i)/((ns+1)/2);
        end
        nomega_up (K+1)= norm(omega_up (:,:,K+1));
        nomega_low(K+1)= norm(omega_low(:,:,K+1));
        nqleft = norm(qleft);
        nqright = norm(qright);
        N_up  = zeros(gmm.k,K);
        N_low = zeros(gmm.k,K);
        E = zeros(1,K);
        
        %% calculate the statistical contribution
        for i=1:K+1
            n_up  = zeros(1,gmm.k);
            n_low = zeros(1,gmm.k);
            for j=1:gmm.k
                n_up (j) = exp(-0.5*(m(:,i) - gmm.miu(:,j))'/((sigma_up (:,:,i))/2 + gmm.var(:,:,j))*(m(:,i) - gmm.miu(:,j)));
                n_low(j) = exp(-0.5*(m(:,i) - gmm.miu(:,j))'/((sigma_low(:,:,i))/2 + gmm.var(:,:,j))*(m(:,i) - gmm.miu(:,j)));
            end
            N_up (:,i) = n_up';
            N_low(:,i) = n_low';
            Er = nqright    *nomega_up (i)*((pi)^(I/2)*det(sigma_up (:,:,i))^0.5*N_up (:,i)'*gmm.weight')^(0.5) + ...
                (1-nqright)*nomega_low(i)*((pi)^(I/2)*det(sigma_low(:,:,i))^0.5*N_low(:,i)'*gmm.weight')^(0.5);
            El = nqleft     *nomega_up (i)*((pi)^(I/2)*det(sigma_up (:,:,i))^0.5*N_up (:,i)'*gmm.weight')^(0.5) + ...
                (1-nqleft) *nomega_low(i)*((pi)^(I/2)*det(sigma_low(:,:,i))^0.5*N_low(:,i)'*gmm.weight')^(0.5);
            E(i) = (abs(Er) + abs(El));
        end
        
        %% rule growing
        if E(K+1) > 0.7*sum(E(1:K)) && k > 1
            K = K + 1;
            P(:,:,K)  = eye(Z);
            for j = 1:K-1
                P(:,:,j)= P(:,:,j)*(K^2+1)/K^2;
            end
            grow = 1;
            fprintf('The new rule no %d is FORMED around sample %d\n', K, k)
        else
            grow = 0;
        end
        nrule(k) = K;
    end
    
    %% second layer: QMF layer
    for j = 1:K
        for i=1:I
            for r = 1 : ns
                if x(k,i) <= m(i,j)
                    yi(r) = 1/(1+exp(-b*(x(k,i)-m(i,j)+abs(theta_low(i,r,j)))));
                    yu(r) = 1/(1+exp(-b*(x(k,i)-m(i,j)+abs(theta_up (i,r,j)))));
                else
                    if isnan(exp(-b*(x(k,i)-m(i,j)-abs(theta_low(i,r,j))))) || isinf(exp(-b*(x(k,i)-m(i,j)-abs(theta_low(i,r,j)))))
                        yi(j)=1;
                    else
                        yi(r) = exp(-b*(x(k,i)-m(i,j)-abs(theta_low(i,r,j))))/(1+exp(-b*(x(k,i)-m(i,j)-abs(theta_low(i,r,j)))));
                    end
                    if isnan(exp(-b*(x(k,i)-m(i,j)-abs(theta_up (i,r,j))))) || isinf(exp(-b*(x(k,i)-m(i,j)-abs(theta_low(i,r,j)))))
                        yu(j)=1;
                    else
                        yu(r) = exp(-b*(x(k,i)-m(i,j)-abs(theta_up (i,r,j))))/(1+exp(-b*(x(k,i)-m(i,j)-abs(theta_up (i,r,j)))));
                    end
                end
            end
            Ql(i,j) = (1/ns)*sum(yi);
            Qu(i,j) = (1/ns)*sum(yu);
        end
        
        %% third layer: rule layer
        Ru(j) = prod(Qu(:,j));
        Rl(j) = prod(Ql(:,j));
    end
    sumrurl = (sum(Ru) + sum(Rl));
    
    %% determine the winning rule
    if k <= fix_themodel && grow == 0
        Ra = 0.5*(Ru+Rl);
        [~,index] = max(Ra(1:K),[],2);
        wr(k) = index;
        omega_low_winner = omega_low(:,:,index);
        omega_up_winner  = omega_up (:,:,index);
        theta = [reshape(omega_low_winner,[M*(I+1) 1]);reshape(omega_up_winner,[M*(I+1) 1]);qleft;qright;m(:,index);reshape(theta_low(:,:,index),[I*ns 1]);reshape(theta_up(:,:,index),[I*ns 1])];        %% parameter to be updated
    end
    
    %% fourth layer: output layer
    Xe = [1,x(k,:)]';
    for o = 1:M
        for j = 1:K
            phi_low(o,j) = omega_low(o,:,j)*Xe;
            phi_up (o,j) = omega_up (o,:,j)*Xe;
        end
    end
    for o  = 1:M
        yl = ((((1-qleft (o))*Rl*phi_low(o,:)')) + (qleft (o)*Ru*phi_low(o,:)'))/sumrurl;
        yr = ((((1-qright(o))*Rl*phi_up (o,:)')) + (qright(o)*Ru*phi_up (o,:)'))/sumrurl;
        y(k,o) = yl + yr;                           %% REGRESSION OUTPUT
        
        %% Classification Output
        if mode == 1                %% Binary Classification
            if y(k,o) > 0.5
                out(k,o) = 1;
            else
                out(k,o) = 0;
            end
        elseif mode == 2            %% Multiclass Classification
            [~,ww] = max(y(k,:));
            out(k,:) = 0;
            out(k,ww) = 1;
        end
    end
    
    %% calculate error
    e(k,:) = z(k,:) - y(k,:);
    
    %% DEKF
    if k <= fix_themodel && grow == 0
        for i=1:I
            Qlw = Ql(:,index);
            Qlw(i) = [];
            Quw = Qu(:,index);
            Quw(i) = [];
            for r = 1 : ns
                drlm = (b*exp(-b*(x(k,i)-m(i,index)+abs(theta_low(i,r,index)))))/(1+exp(-b*(x(k,i)-m(i,index)+abs(theta_low(i,r,index)))))^2;
                drum = (b*exp(-b*(x(k,i)-m(i,index)+abs(theta_up (i,r,index)))))/(1+exp(-b*(x(k,i)-m(i,index)+abs(theta_up (i,r,index)))))^2;
                if x(k,i) <= m(i,index)
                    if isnan(exp(-b*(x(k,i)-m(i,index)+abs(theta_low(i,r,index))))) || isinf(exp(-b*(x(k,i)-m(i,index)+abs(theta_low(i,r,index)))))
                        dRlm(r) = 0;
                    else
                        dRlm(r) = -drlm;
                    end
                    if isnan(exp(-b*(x(k,i)-m(i,index)+abs(theta_up (i,r,index))))) || isinf(exp(-b*(x(k,i)-m(i,index)+abs(theta_up (i,r,index)))))
                        dRum(r) = 0;
                    else
                        dRum(r) = -drum;
                    end
                    if theta_low(i,r,index) >= 0
                        C = (1/ns)*( b*exp(-b*(x(k,i)-m(i,index)+theta_low(i,r,index))))/(1+exp(-b*(x(k,i)-m(i,index)+theta_low(i,r,index))))^2;
                    else
                        C = (1/ns)*(-b*exp(-b*(x(k,i)-m(i,index)-theta_low(i,r,index))))/(1+exp(-b*(x(k,i)-m(i,index)-theta_low(i,r,index))))^2;
                    end
                    if theta_up(i,r,index) >= 0
                        D = (1/ns)*( b*exp(-b*(x(k,i)-m(i,index)+theta_up (i,r,index))))/(1+exp(-b*(x(k,i)-m(i,index)+theta_up(i,r,index))))^2;
                    else
                        D = (1/ns)*(-b*exp(-b*(x(k,i)-m(i,index)-theta_up (i,r,index))))/(1+exp(-b*(x(k,i)-m(i,index)-theta_up(i,r,index))))^2;
                    end
                else
                    if isnan(exp(-b*(x(k,i)-m(i,index)+abs(theta_low(i,r,index))))) || isinf(exp(-b*(x(k,i)-m(i,index)+abs(theta_low(i,r,index)))))
                        dRlm(r) = 0;
                    else
                        dRlm(r) = drlm;
                    end
                    if isnan(exp(-b*(x(k,i)-m(i,index)+abs(theta_up (i,r,index))))) || isinf(exp(-b*(x(k,i)-m(i,index)+abs(theta_up (i,r,index)))))
                        dRum(r) = 0;
                    else
                        dRum(r) = drum;
                    end
                    if theta_low(i,r,index) >= 0
                        C = (1/ns)*(-b*exp(-b*(x(k,i)-m(i,index)+theta_low(i,r,index))))/((1+exp(-b*(x(k,i)-m(i,index)+theta_low(i,r,index))))^2);
                    else
                        C = (1/ns)*( b*exp(-b*(x(k,i)-m(i,index)-theta_low(i,r,index))))/((1+exp(-b*(x(k,i)-m(i,index)-theta_low(i,r,index))))^2);
                    end
                    if theta_up(i,r,index) >= 0
                        D = (1/ns)*(-b*exp(-b*(x(k,i)-m(i,index)+theta_up(i,r,index))))/((1+exp(-b*(x(k,i)-m(i,index)+theta_up(i,r,index))))^2);
                    else
                        D = (1/ns)*( b*exp(-b*(x(k,i)-m(i,index)-theta_up(i,r,index))))/((1+exp(-b*(x(k,i)-m(i,index)-theta_up(i,r,index))))^2);
                    end
                end
                dRldtl(i,r) = prod(Qlw)*C;
                dRudtu(i,r) = prod(Quw)*D;
            end
            A = (1/ns)*sum(dRlm);
            B = (1/ns)*sum(dRum);
            dRldm(i) = prod(Qlw)*A;
            dRudm(i) = prod(Quw)*B;
        end
        
        h1 = [];
        h2 = [];
        h3 = [];
        h4 = [];
        h6 = zeros(I*ns,M);
        h7 = zeros(I*ns,M);
        for o = 1:M
            % partial derivative of y w.r.t. the weight of winning rule
            dydw_low(:,o) = (((((1-qleft (o))*Rl(index))) + (qleft (o)*Ru(index)))/sumrurl)*Xe';            %% DYDWL
            dydw_up (:,o) = (((((1-qright(o))*Rl(index))) + (qright(o)*Ru(index)))/sumrurl)*Xe';            %% DYDWU
            
            % partial derivative of y w.r.t. design factor
            dydql(o) = (((-Rl*phi_low(o,:)')) + (Ru*phi_low(o,:)'))/sumrurl;                    %% DYDQL
            dydqr(o) = (((-Rl*phi_up (o,:)')) + (Ru*phi_up (o,:)'))/sumrurl;                    %% DYDQR
            
            % partial derivative of yl and yr w.r.t. center, theta_low, and theta_up
            % the left term
            dyldRl(o) = ((((1-qleft(o)) *phi_low(o,index))))/sumrurl - ((((1-qleft (o))*Rl*phi_low(o,:)')) + (qleft (o)*Ru*phi_low(o,:)'))/sumrurl^2;
            dyldRu(o) = ((((qleft  (o)) *phi_low(o,index))))/sumrurl - ((((1-qleft (o))*Rl*phi_low(o,:)')) + (qleft (o)*Ru*phi_low(o,:)'))/sumrurl^2;
            % the right term
            dyrdRl(o) = ((((1-qright(o))*phi_up (o,index))))/sumrurl - ((((1-qright(o))*Rl*phi_up (o,:)')) + (qright(o)*Ru*phi_up (o,:)'))/sumrurl^2;
            dyrdRu(o) = ((((qright  (o))*phi_up (o,index))))/sumrurl - ((((1-qright(o))*Rl*phi_up (o,:)')) + (qright(o)*Ru*phi_up (o,:)'))/sumrurl^2;
            
            % partial derivative of y w.r.t. center
            dymdm(:,o) = (dyldRl(o)*dRldm' + dyldRu(o)*dRudm') + (dyrdRl(o)*dRldm' + dyrdRu(o)*dRudm');             %% DYDM
            
            % partial derivative of y w.r.t. lower and upper quantum interval
            dymdtl(:,:,o) = (dyldRl(o)*dRldtl) + (dyrdRl(o)*dRldtl);                                                %% DYDTL
            dymdtu(:,:,o) = (dyldRu(o)*dRudtu) + (dyrdRu(o)*dRudtu);                                                %% DYDTU
            
            h1 = blkdiag(h1,dydw_low(:,o));
            h2 = blkdiag(h2,dydw_up(:,o));
            h3 = blkdiag(h3,dydql(:,o));
            h4 = blkdiag(h4,dydqr(:,o));
            h6(:,o) = reshape(dymdtl(:,:,o),[I*ns 1]);
            h7(:,o) = reshape(dymdtu(:,:,o),[I*ns 1]);
        end
        H=[h1;h2;h3;h4;dymdm;h6;h7];
        
        %% DEKF updating procedure
        G            = P(:,:,index)*H/(R+H'*P(:,:,index)*H);
        P(:,:,index) = (eye(Z)-G*H')*P(:,:,index);
        theta        = theta + G*e(k,:)';
        
        %% Substitute back the updated parameters
        omega_low(:,:,index) = reshape(theta(1:M*(I+1)),[M I+1]);
        omega_up (:,:,index) = reshape(theta(M*(I+1)+1:2*M*(I+1)),[M I+1]);
        nomega_up (index)    = norm(omega_up (:,:,index));
        nomega_low(index)    = norm(omega_low(:,:,index));
        qleft                = theta(2*M*(I+1)+1:2*M*(I+1)+1+(M-1));
        qright               = theta(2*M*(I+1)+1+(M-1)+1:2*M*(I+1)+1+2*(M-1)+1);
        m(:,index)           = theta(2*M*(I+1)+1+2*(M-1)+1+1:2*M*(I+1)+1+2*(M-1)+1+(I-1)+1);
        theta_low(:,:,index) = reshape(theta(2*M*(I+1)+1+2*(M-1)+1+(I-1)+1+1:2*M*(I+1)+1+2*(M-1)+1+(I-1)+1+1+(I*ns-1)),[I ns]);
        theta_up(:,:,index)  = reshape(theta(2*M*(I+1)+1+2*(M-1)+1+(I-1)+1+1+(I*ns-1)+1:2*M*(I+1)+1+2*(M-1)+1+(I-1)+1+1+2*(I*ns-1)+1),[I ns]);
        sigma_low(:,:,index) = diag(min(abs(theta_low(:,:,index)),[],2).*((ns+1)/2)).^2;
        sigma_up (:,:,index) = diag(min(abs(theta_up (:,:,index)),[],2).*((ns+1)/2)).^2;
        fprintf('The rule no %d is UPDATED around sample %d\n', index, k)
    end
end
y(1,:) = z(1,:);
e(1,:) = 0;
if mode == 1 || mode == 2
    y = out;
    if mode == 1                %% Binary Classification
        if y(1,o) > 0.5
            y(1,o) = 1;
        else
            y(1,o) = 0;
        end
    elseif mode == 2            %% Multiclass Classification
        [~,ww] = max(y(1,:));
        y(1,:) = 0;
        y(1,ww) = 1;
    end
end
network.I = I;
network.miu = m(:,1:K);
network.ns = ns;
network.b = b;
network.theta_up = theta_up(:,:,1:K);
network.theta_low = theta_low(:,:,1:K);
network.omega_up = omega_up(:,:,1:K);
network.omega_low = omega_low(:,:,1:K);
network.qleft = qleft;
network.qright = qright;
rule.nrule = nrule;
rule.K = K;
rule.wr = wr;
disp('eT2QFNN: finished');
time = toc;
performance.time = time;

%% Performance measurement
if mode == 3
    if M == 1
        MSEtrain = sum(e(1:fix_themodel).^2)/(fix_themodel);
        RMSEtrain = sqrt(MSEtrain);
        NDEItrain = RMSEtrain/std(z(1:fix_themodel));
        MSEtest = sum(e(fix_themodel+1:end).^2)/(N-fix_themodel);
        RMSEtest = sqrt(MSEtest);
        NDEItest = RMSEtest/std(z(fix_themodel+1:end));
        MSE = sum(e.^2)/N;
        RMSE = sqrt(MSE);
        NDEI = RMSE/std(z);
        remark = {'Number of data','MSE','RMSE','NDEI'};
        location1 = [fix_themodel MSEtrain RMSEtrain NDEItrain];
        location2 = [N-fix_themodel MSEtest RMSEtest NDEItest];
        location3 = [N MSE RMSE NDEI];
        T = table;
        T.Remark = remark';
        T.Training = location1';
        T.Validation = location2';
        T.Total = location3';
        disp(T);
        performance.RMSE_training = RMSEtrain;
        performance.NDEI_training = NDEItrain;
        performance.RMSE_validation = RMSEtest;
        performance.NDEI_validation = NDEItest;
    elseif M >= 2
        for o = 1:M
            MSEtrain = sum(e(1:fix_themodel,o).^2)/(fix_themodel);
            RMSEtrain = sqrt(MSEtrain);
            NDEItrain = RMSEtrain/std(z(1:fix_themodel,o));
            MSEtest = sum(e(fix_themodel+1:end,o).^2)/(N-fix_themodel);
            RMSEtest = sqrt(MSEtest);
            NDEItest = RMSEtest/std(z(fix_themodel+1:end,o));
            MSE = sum(e(:,o).^2)/N;
            RMSE = sqrt(MSE);
            NDEI = RMSE/std(z(:,o));
            remark = {'Number of data','MSE','RMSE','NDEI'};
            location1 = [fix_themodel MSEtrain RMSEtrain NDEItrain];
            location2 = [N-fix_themodel MSEtest RMSEtest NDEItest];
            location3 = [N MSE RMSE NDEI];
            T = table;
            T.Remark = remark';
            T.Training = location1';
            T.Validation = location2';
            T.Total = location3';
            fprintf('Output %d\n',o)
            disp(T);
            performance.RMSE_training(1,o) = RMSEtrain;
            performance.NDEI_training(1,o) = NDEItrain;
            performance.RMSE_validation(1,o) = RMSEtest;
            performance.NDEI_validation(1,o) = NDEItest;
        end
    end
    T2 = table;
    T2.Rule = K;
    T2.Computation_Time = time;
    disp(T2);
elseif mode == 1 || mode == 2
    count=0;
    for k=1:fix_themodel
        if z(k,:) == y(k,:)
            count = count+1;
        end
    end
    classification_rate_training = count/fix_themodel;
    count1=0;
    for k = fix_themodel+1:N
        if z(k,:) == y(k,:)
            count1 = count1+1;
        end
    end
    classification_rate_validation = count1/(N-fix_themodel);
    remark = {'Number of data','Classification rate'};
    location1 = [fix_themodel classification_rate_training];
    location2 = [N-fix_themodel classification_rate_validation];
    T = table;
    T.Remark = remark';
    T.Training = location1';
    T.Validation = location2';
    disp(T);
    T2 = table;
    T2.Rule = K;
    T2.Computation_Time = time;
    disp(T2);
    performance.classification_rate_training = classification_rate_training;
    performance.classification_rate_validation = classification_rate_validation;
end

%% Plot the resutl
if mode == 3
    plot(wr);
    title('winning rule');
    xlabel('iteration');
    ylabel('Rule no.');
    xlim([0 N]);
    for o = 1:M
        figure;
        plot(e(:,o));
        xlabel('iteration');
        ylabel('error');
        xlim([0 N]);
    end
    for o = 1:M
        figure;
        plot(z(:,o),'-b','Linewidth',.5)
        hold on
        plot(y(:,o),'-.r','Linewidth',1)
        xlabel('iteration');
        ylabel('y');
        legend('target','output','Location','Best');
        xlim([0 N]);
        ylim([min(y(:,o))-0.1 max(y(:,o))+0.1]);
        hold off;
    end
end
end