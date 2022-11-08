%%%%%%%%%%%%%%%%%%%%% Parliamentarism %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for a=1:areas
    newAWinner = aWinners(a,:);
    i = aWinnerInd(a);    
    
    toa = randi(areas);
    while(toa == a)
        toa = randi(areas);%�����������Ҹ��쵼�˿����������Ӧֵ����ߣ����ȷ������
    end
	toAWinner = aWinners(toa,:);
    for j = 1:dim
        distance = abs(toAWinner(1,j) - newAWinner(1,j));
        newAWinner(1,j) = toAWinner(1,j) + (2*rand()-1) * distance;
    end
    newAWFitness=fobj(newAWinner(1,:));
    
	%Replace only if improves
	if newAWFitness < fitness(i) 
        Positions(i,:) = newAWinner(1,:);
        fitness(i) = newAWFitness;
        aWinners(a,:) = newAWinner(1,:);
	end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Opposite
for a=1:areas
    newAWinner = oppoaWinners(a,:);
    i = oppoaWinnerInd(a);    
    
    toa = randi(areas);
    while(toa == a)
        toa = randi(areas);%�����������Ҹ��쵼�˿����������Ӧֵ����ߣ����ȷ������
    end
	toAWinner = oppoaWinners(toa,:);
    for j = 1:dim
        distance = abs(toAWinner(1,j) - newAWinner(1,j));
        newAWinner(1,j) = toAWinner(1,j) + (2*rand()-1) * distance;
    end
    newAWFitness=fobj(newAWinner(1,:));
    
	%Replace only if improves
	if newAWFitness < fitness(i) 
        Positions(i,:) = newAWinner(1,:);
        fitness(i) = newAWFitness;
        aWinners(a,:) = newAWinner(1,:);
	end
end