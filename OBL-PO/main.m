%_________________________________________________________________________________
%  Political Optimizer: A novel socio-inspired meta-heuristic 
%                       for global optimization source codes version 1.0
%
%  Developed in MATLAB R2015a
%
%  Author and programmer: Qamar Askari
%
%         e-Mail: l165502@lhr.nu.edu.pk
%                 syedqamar@gift.edu.pk
%
%
%   Main paper:
%   Askari, Q., Younas, I., & Saeed, M. (2020). Political Optimizer: 
%       A novel socio-inspired meta-heuristic for global optimization.
%   Knowledge-Based Systems, 2020, 
%   DOI: https://doi.org/10.1016/j.knosys.2020.105709
%____________________________________________________________________________________
%% 永安：注意，该代码中是以最小适应值为目标去优化的，因此需要输入负数进去

%%
clear all 
clc
global X1
global X2
global X3
global dim;%修改dim记得修改model_num
dim=40;
[X1,X2,X3]=xlsread('泰勒图.xlsx');
%%%%%%%%%%%%%%%%%%%%%%Adjustable parameters%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
parties = [20 50 100 200];        %Number of political parties
lambda = 1.0;       %Max limit of party switching rate
fEvals = 20900;     %Number of function evaluations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,runs] = size(parties);
ZPO_cg_curve=[];
ZBest_score_0=[];
ZBest_pos=[];
for fn = 1:1%这个代码没意义了

    Function_name=strcat('F',num2str(fn)); % Name of the test function %这个代码也没意义了
    [lb,ub,dim,fobj]=get_fun();%修改适应值函数和纬度（注意如果优化最小值需要乘以负数）
    %[lb,ub,dim,fobj]=Get_Functions_Details_Uni(Function_name);
    
    %Function_name=strcat('MF',num2str(fn)); % Name of the test function 
    %[lb,ub,dim,fobj]=Get_Functions_Details_Multi(Function_name);

    % Calling algorithm
    Best_score_T = zeros(1,runs);
    for run=1:runs
        parties1=parties(run);
        areas = parties1;                
        populationSize=parties1 * areas; % Number of search agents
        Max_iteration = 50;
        
        
        rng('shuffle');
        [Best_score_0,Best_pos,PO_cg_curve]=PO(populationSize,areas,parties1,lambda,Max_iteration,lb,ub,dim,fobj);
        Best_score_T(1,run) = Best_score_0;
        
        Best_score_0
        ZBest_score_0=[ZBest_score_0;Best_score_0];
        ZBest_pos=[ZBest_pos;Best_pos];
        
        figure
        plot(PO_cg_curve,'linewidth',1.5);
        title('GWO-SVM收敛曲线')
        xlabel('迭代次数')
        ylabel('适应度值')
        grid on;
        ZPO_cg_curve=[ZPO_cg_curve;PO_cg_curve];

    end

    %Finding statistics
    Best_score_Best = min(Best_score_T);
    Best_score_Worst = max(Best_score_T);
    Best_score_Median = median(Best_score_T,2);
    Best_Score_Mean = mean(Best_score_T,2);
    Best_Score_std = std(Best_score_T);


    %Printing results
    display(['Fn = ', num2str(fn)]);
    display(['Best, Worst, Median, Mean, and Std. are as: ', num2str(Best_score_Best),'  ', ...
        num2str(Best_score_Worst),'  ', num2str(Best_score_Median),'  ', num2str(Best_Score_Mean),'  ', num2str(Best_Score_std)]);

    
    
end
