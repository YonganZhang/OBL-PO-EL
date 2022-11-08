
% This function initialize the first population of search agents
function Positions=initialization(SearchAgents_no,dim,ub,lb)
Boundary_no= size(ub,2); % numnber of boundaries, bounds are vector
Positions = zeros(SearchAgents_no,dim); %Declaration

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(1:SearchAgents_no/2,i)=rand(SearchAgents_no/2,1).*(ub_i-lb_i)+lb_i;
    end
    for i=SearchAgents_no/2+1:SearchAgents_no-1
        Positions(i,:)=ub_i-Positions(SearchAgents_no-i,:)+lb_i;
    end
end

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no/2,dim).*(ub-lb)+lb;
    for i=SearchAgents_no/2+1:SearchAgents_no
        ub_i=ub;
        lb_i=lb;
        Positions(i,:)=ub_i-Positions(SearchAgents_no-i+1,:)+lb_i;
    end
end
end
