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

    ElectionCampaign;%��������������Ӧֵ�����ֳ�ѡ����Ѻ͵�����ѣ�Ȼ���ȵ�����ѵ�ȥ�ƶ�����ѡ����ѵ�ȥ�ƶ�    
    PartySwitching;%������ķ������ݼ���0������ÿ��������ĳ�Ա����ѡ��Ȼ������Ҹ��������ɵ���Ա����
    Election;%����ÿ���������Ӧֵ�����������Ӧֵ�ĸ���
    GovernmentFormation;%ѡ���쵼�˺͵��쵼��
    Parliamentarism;%ÿ�������쵼�˶����һ���Ŷӣ�����Ŷ��ڵ�ÿ����Ա�����α�������������������ҵ��쵼�˿�����Ӧֵ����ߣ����ȷ������
        
    t=t+1;
    if t==45
        a=1;
    end
    Convergence_curve(t)=Leader_score;
    [t Leader_score];
end

