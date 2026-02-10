function [OffDec,OffMask] = OperatorNew(Problem,ParentDec,ParentMask,FitnessLayer,LayerMax,Delta,Local_Knowlege,Global_Knowlege,NSV,SV,theta,Fitness)

    %% Parameter setting
    [N,D]       = size(ParentDec);
    Parent1Mask = ParentMask(1:N/2,:);
    Parent2Mask = ParentMask(N/2+1:end,:);   
    %%
     %% Crossover for mask
    OffMask = Parent1Mask;
    for i = 1 : N/2
        if rand < theta
            allOne   = Local_Knowlege(1,:);
            other    = Local_Knowlege(3,:);
            if rand < 0.5
                index = find(Parent1Mask(i,other(NSV))~=Parent2Mask(i,other(NSV)));
                index = index(TS(-Delta(index)));
                OffMask(i,index) = 1;
                index1 = find(Parent1Mask(i,other(SV))~=Parent2Mask(i,other(SV)));
                index1 = index1(TS(-Delta(index1)));
                OffMask(i,index1) = 1;
            else
                index = find(Parent1Mask(i,other(NSV))~=Parent2Mask(i,other(NSV)));
                index = index(TS(Delta(index)));
                OffMask(i,index) = 0;
                index1 = find(Parent1Mask(i,other(SV))~=Parent2Mask(i,other(SV)));
                index1 = index1(TS(Delta(index1)));
                OffMask(i,index1) = 0;
            end
            OffMask(allOne)    = true;
        else
            allOne   = Global_Knowlege(1,:);
            other    = Global_Knowlege(3,:);
            if rand < 0.5
                index = find(Parent1Mask(i,other(NSV))~=Parent2Mask(i,other(NSV)));
                index = index(TS(-Delta(index)));
                OffMask(i,index)=1;
                index1 = find(Parent1Mask(i,other(SV))~=Parent2Mask(i,other(SV)));
                index1 = index1(TS(-Delta(index1)));
                OffMask(i,index1)=1;
            else
                index = find(Parent1Mask(i,other(NSV))~=Parent2Mask(i,other(NSV)));
                index = index(TS(Delta(index)));
                OffMask(i,index)=0;
                index1 = find(Parent1Mask(i,other(SV))~=Parent2Mask(i,other(SV)));
                index1 = index1(TS(Delta(index1)));
                OffMask(i,index1)=0;
            end
            OffMask(allOne)    = true;
        end
    end
    %%   
    %%
    %%    
    for i = 1 : N/2 
        PointUp   = 1; 
        PointDown = LayerMax; 
        for j = 1 : LayerMax  
            TargetUpLayer   = find(FitnessLayer == PointUp);
            TargetUp        = TargetUpLayer(OffMask(i,TargetUpLayer) == 0);
            TargetDownLayer = find(FitnessLayer == PointDown);      
            TargetDown      = TargetDownLayer(OffMask(i,TargetDownLayer) == 1);
            if rand < 0.5 
                if ~isempty(TargetUp)
                    if rand < 0.5
                        TargetUp = TargetUp(TS(-Delta(TargetUp)));
                        OffMask(i,TargetUp) = 1;
                    end
                end
                if rand < 0.5
                    PointUp = PointUp + 1;
                else
                    break;
                end
            else
                if ~isempty(TargetDown)
                    if rand < 0.5  
                        TargetDown = TargetDown(TS(Delta(TargetDown)));
                        OffMask(i,TargetDown) = 0;
                    end
                end
                if rand < 0.5
                    PointDown = PointDown - 1;
                else
                    break; 
                end
            end
            if PointUp >= PointDown 
                break;
            end            
        end
    end

    %% Crossover and mutation for dec
    if any(Problem.encoding~=4)
        OffDec = OperatorGAhalf(Problem,ParentDec);
        OffDec(:,Problem.encoding==4) = 1;
    else
        OffDec = ones(N/2,D);
    end
end

%%
%%
function index = TS(Delta)
    if isempty(Delta)
        index = [];
    else
        index = TournamentSelection(2,1,Delta);
    end
end