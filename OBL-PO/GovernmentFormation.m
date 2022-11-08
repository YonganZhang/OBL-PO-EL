%%%%%%%%%%%%%%%%%%%%% Govt. Formation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
aWinnerInd=zeros(areas,1);   %Indices of area winners in x
aWinners = zeros(areas,dim); %Area winners are stored separately
for a = 1:areas
	[aWinnerFitness,aWinnerParty]=min(fitness(a:areas:SearchAgents_no));
	aWinnerInd(a,1) = (aWinnerParty-1) * areas + a;
    aWinners(a,:) = Positions(aWinnerInd(a,1),:);
end    

%Finding party leaders
pLeaderInd=zeros(parties,1);    %Indices of party leaders in x
pLeaders = zeros(parties,dim);  %Positions of party leaders in x
for p = 1:parties
	pStIndex = (p-1) * areas + 1;
	pEndIndex = pStIndex + areas - 1;
	[partyLeaderFitness,leadIndex]=min(fitness(pStIndex:pEndIndex)); 
	pLeaderInd(p,1) = (pStIndex - 1) + leadIndex; %Indexof party leader
    pLeaders(p,:) = Positions(pLeaderInd(p,1),:);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% opposite
oppoaWinnerInd=zeros(areas,1);   %Indices of area winners in x
oppoaWinners = zeros(areas,dim); %Area winners are stored separately
for a = 1:areas
	[aWinnerFitness,aWinnerParty]=max(fitness(a:areas:SearchAgents_no));
	oppoaWinnerInd(a,1) = (aWinnerParty-1) * areas + a;
    oppoaWinners(a,:) = Positions(oppoaWinnerInd(a,1),:);
end    

%Finding party leaders
oppopLeaderInd=zeros(parties,1);    %Indices of party leaders in x
oppopLeaders = zeros(parties,dim);  %Positions of party leaders in x
for p = 1:parties
	pStIndex = (p-1) * areas + 1;
	pEndIndex = pStIndex + areas - 1;
	[partyLeaderFitness,leadIndex]=max(fitness(pStIndex:pEndIndex)); 
	oppopLeaderInd(p,1) = (pStIndex - 1) + leadIndex; %Indexof party leader
    oppopLeaders(p,:) = Positions(oppopLeaderInd(p,1),:);
end

