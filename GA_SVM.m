clear 
clc
%% 加载数据
load SVM_test.mat
load SVM_train.mat

%% 提取训练\测试集数据和标签
%提取训练集数据和标签
train_data = SVM_train(:,1:4);
train_label = SVM_train(:,5);

%%  GA优化

% GA的参数选项初始化
ga_option.maxgen = 200; %最大进化代数
ga_option.sizepop = 20; %最大种群数量
ga_option.cbound = [0,100];%c参数
ga_option.gbound = [0,1000];%g参数
ga_option.v = 5;%SVM Cross Validation参数
ga_option.ggap = 0.9;%代沟

MAXGEN = ga_option.maxgen;
NIND = ga_option.sizepop;
NVAR = 2;%个体所含变量数
PRECI = 20;%个体长度
GGAP = ga_option.ggap;
trace = zeros(MAXGEN,2);
FieldID = ...
[rep([PRECI],[1,NVAR]);[ga_option.cbound(1),ga_option.gbound(1);ga_option.cbound(2),ga_option.gbound(2)]; ...
 [1,1;0,0;0,1;1,1]];

%初始化种群
Chrom = crtbp(NIND,NVAR*PRECI);%初始化个体

gen = 1;%初始化代数计数器
v = ga_option.v;%SVM Cross Validation参数
BestCVaccuracy = 0;
Bestc = 0;
Bestg = 0;
cg = bs2rv(Chrom,FieldID);%初始种群从二进制到十进制，相当于给种群中每个个体赋c,g初值
%初始支持向量机精度
for nind = 1:NIND
    cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2))];
    ObjV(nind,1) = svmtrain(train_label,train_data,cmd); %#ok<SVMTRAIN>
end
%初始最好的c,g
[BestCVaccuracy,I] = max(ObjV);
Bestc = cg(I,1);
Bestg = cg(I,2);
%遗传迭代
for gen = 1:MAXGEN
    FitnV = ranking(-ObjV);%适应度，以精度表示
    SelCh = select('sus',Chrom,FitnV,GGAP);%选择
    SelCh = recombin('xovsp',SelCh,0.7);%交叉
    SelCh = mut(SelCh);%变异
    
    cg = bs2rv(SelCh,FieldID);%解码获得新的c,g参数
    %计算进化一次后（子代）的svm精度
    for nind = 1:size(SelCh,1)
        cmd = ['-v ',num2str(v),' -c ',num2str(cg(nind,1)),' -g ',num2str(cg(nind,2))];
        ObjVSel(nind,1) = svmtrain(train_label,train_data,cmd);
    end
    [Chrom,ObjV] = reins(Chrom,SelCh,1,1,ObjV,ObjVSel);%子代插入到父代，得到新种群
    
    %如果子代svm最大精度小于50%则表明此时进化失败，直接进入下次迭代
    if max(ObjV) <= 50
        continue;
    end
    
    [NewBestCVaccuracy,I] = max(ObjV);%子代的最大精度
    cg_temp = bs2rv(Chrom,FieldID);%子代的c,g值
    temp_NewBestCVaccuracy = NewBestCVaccuracy;%获得子代最好的精度
    
    %获取子代最优c,g参数
    if NewBestCVaccuracy > BestCVaccuracy
       BestCVaccuracy = NewBestCVaccuracy;
       Bestc = cg_temp(I,1);
       Bestg = cg_temp(I,2);
    end
    
    if abs( NewBestCVaccuracy-BestCVaccuracy ) <= 10^(-2) && ...
        cg_temp(I,1) < Bestc
       BestCVaccuracy = NewBestCVaccuracy;
       Bestc = cg_temp(I,1);
       Bestg = cg_temp(I,2);
    end    
   
    trace(gen,1) = max(ObjV);%计算最佳适应度
    trace(gen,2) = sum(ObjV)/length(ObjV);%计算平均适应度
    best_ObjV = trace(gen,1);
    average_ObjV = trace(gen,2);
    disp('***************')
    fprintf('best_ObjV = %d\n',best_ObjV);
    fprintf('average_ObjV = %d\n',average_ObjV);
    disp('***************')
end

%作图
figure;
hold on;
trace = round(trace*10000)/10000;
h1 = plot(trace(1:gen,1),'r*-','LineWidth',1.5);
h2 = plot(trace(1:gen,2),'o-','LineWidth',1.5);
legend([h1(1),h2(1)],'最佳适应度','平均适应度');
xlabel('进化代数','FontSize',12);
ylabel('适应度','FontSize',12);
axis([0 gen 0 100]);
grid on;
axis auto;

line1 = '适应度曲线Accuracy[GAmethod]';
line2 = ['(终止代数=', ...
    num2str(gen),',种群数量pop=', ...
    num2str(NIND),')'];
line3 = ['Best c=',num2str(Bestc),' g=',num2str(Bestg), ...
    ' CVAccuracy=',num2str(BestCVaccuracy),'%'];
title({line1;line2;line3},'FontSize',12);




