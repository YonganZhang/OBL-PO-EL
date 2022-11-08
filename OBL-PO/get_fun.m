function [lb,ub,dim,fobj] = get_fun()

fobj = @F1;
       lb=0;
       ub=1;
       global dim


function Zr2 = F1(Positions)
test_size=84;
K=10;
model_num=dim;
all_num=test_size*K*model_num;
count=0;
ture=zeros(test_size*K,2);
pre=zeros(test_size*K,2);
global X1


for i=1:model_num
    if(Positions(i)>0)
        count=Positions(i)+count;
        for j=1:K
            %每一折都加上这个模型
            start=1+(test_size*(i-1))+(j-1)*test_size*model_num;%第i个模型在第j折的开始，(j-1)*test_size*model_num是折的位置、1+(test_size*(i-1))是每折中模型的位置
            en=test_size*i+(j-1)*test_size*model_num;
            start2=1+test_size*(j-1);%汇总后，每一折的结果是相同的，如168(testsize)、5（K）、39（模型数），那么每个模型都加到同一折里就是总共168*5的数据量
            en2=test_size*(j);%从每折开始，每折结束
            pre(start2:en2,:)=pre(start2:en2,:)+Positions(i)*X1(start:en,3:4);
            ture(start2:en2,:)=ture(start2:en2,:)+Positions(i)*X1(start:en,1:2);
        end
        
        
%         for j=1:test_size*model_num:all_num
%             pre=[pre;X1(j+(test_size*(i-1)):j+(test_size*i)-1,2)];
%             ture=[ture;X1(j+(test_size*(i-1)):j+(test_size*i)-1,1)];
%         end
    end
end
    pre_all=pre/count;
    ture_all=ture/count;
    Zr2=0;
    [numm,len]=size(pre_all);
    for i=1:len
        pre=pre_all(:,i);
        ture=ture_all(:,i);
        %Zr2 = (sum((pre - ture).^2) / sum((pre - mean(ture)).^2))+Zr2;% 计算RMSE，为了最小化，原是1-XX=Zr2
        Zr2 = (sum((pre - ture).^2))/numm+Zr2;
    end
    Zr2=Zr2/len;
end

end

