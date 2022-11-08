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

function [Leader_score,Leader_pos,Convergence_curve]=PO(SearchAgents_no,areas,parties,lambda,Max_iter,lb,ub,dim,fobj)
% initialize position vector and score for the leader
Leader_pos=zeros(1,dim);
Leader_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);
auxPositions = Positions;
prevPositions = Positions;
Convergence_curve=zeros(1,Max_iter);
fitness=zeros(SearchAgents_no, 1);

%Running phases for initializations
Election;   %Run election phase
auxFitness = fitness;
prevFitness = fitness;
GovernmentFormation;

t=0;% Loop counter
while t<Max_iter
    prevFitness = auxFitness;
    prevPositions = auxPositions;
    auxFitness = fitness;
    auxPositions = Positions;

    ElectionCampaign;%计算各个个体的适应值，划分出选区最佳和党内最佳，然后先党内最佳得去移动、再选区最佳得去移动    
    PartySwitching;%按照莱姆达（慢慢递减到0），对每个党派里的成员，都选定然后随机找个其他党派的最差党员换掉
    Election;%更新每个个体的适应值，保存最佳适应值的个体
    GovernmentFormation;%选区领导人和党领导人
    Parliamentarism;%每个区的领导人都组成一个团队，这个团队内的每个成员都来次遍历，如果在议会里随机找的领导人靠近适应值会提高，则就确定靠近
        
    t=t+1;
    if t==45
        a=1;
    end
    Convergence_curve(t)=Leader_score;
    [t Leader_score];
end

