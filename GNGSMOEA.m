classdef GNGSMOEA < ALGORITHM
% <2024> <multi> <real/binary> <large/none> <constrained/none> <sparse>
% Multi-granularity clustering based evolutionary algorithm

%------------------------------- Reference --------------------------------
% Y. Tian, S. Shao, G. Xie, and Y. Jin, A multi-granularity clustering
% based evolutionary algorithm for large-scale sparse multi-objective
% optimization, Swarm and Evolutionary Computation, 2024, 84: 101453.
%------------------------------- Copyright --------------------------------
% Copyright (c) 2024 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    methods
        function main(Algorithm,Problem)
            %%
             %% Parameters for GNG network
            params.N = Problem.N;
            params.MaxIt = 50;
            params.L = 30;
            params.epsilon_b = 0.2;
            params.epsilon_n = 0.006;
            params.alpha = 0.5;
            params.delta = 0.995;
            params.T = 30;
            genFlag = [];
            MaxGen = ceil(Problem.maxFE/Problem.N);
            netInitialized = 0;
            Pop1get = 0;
            gen = 0; 
            %%        
            %% Population initialization
             [~,FitnessOpt,Fitness,SparseRate,TDec,TMask,TempPop] = FitnessCal(Problem);
            [Population,Dec,Mask,FitnessSpea2,FrontNo] = EnvironmentalSelection(TempPop,TDec,TMask,Problem.N);
            NearStage = ceil(Problem.FE/(Problem.maxFE/10));
            [FitnessLayer,LayerMax] = UpdateLayer(SparseRate,NearStage,FitnessOpt,Problem,[]); 
            %%
            Delta = zeros(1,Problem.D);
            GROUP = [];     % In order to prevent outliers in Kmeans algorithm
            net = InitilizeGrowingGasNet(Population,params,FitnessSpea2);
            %% Optimization           
            while Algorithm.NotTerminated(Population)
               %% 
                % decision variable Sparsity analysis
                [Local_Knowlege,Global_Knowledge,Fitness,NSV,SV,theta] = SparsityAnalysis(Problem,Mask,FrontNo,Fitness,GROUP);
                GROUP = [NSV;SV];
                %%
                delta      = Relief(Problem,Population,FrontNo);
                Delta      = Delta + abs(delta);
                %%
                [NearStage,FitnessOpt,FitnessLayer,LayerMax] = ControlStage(SparseRate,NearStage,Mask,Dec,FitnessOpt,FitnessLayer,LayerMax,Problem);
                gen = gen + 1;
                %% Initial the GNG network when can
                if ~netInitialized
                    NDNum = sum(FitnessSpea2<1);
                    if NDNum >= 2 
                        net = InitilizeGrowingGasNet(Population,params,FitnessSpea2);
                        netInitialized = 1;
                    end
                end 
                if ~netInitialized || gen < 0.4 * MaxGen
                %% 
                   MatingPool = TournamentSelection(2,2*Problem.N,FitnessSpea2);
                   [OffDec,OffMask] = OperatorNew(Problem,Dec(MatingPool,:),Mask(MatingPool,:),FitnessLayer,LayerMax,Delta,Local_Knowlege,Global_Knowledge,NSV,SV,theta,Fitness);
                   Offspring = Problem.Evaluation(OffDec.*OffMask);
                   [Population,Dec,Mask,FitnessSpea2,FrontNo,net,genFlag,Fitness1] = EnvironmentalSelection1([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N,Problem,params,net,genFlag);
                 %%
                else
                    if Pop1get == 0
                        V = net.w;
                        Population1 = Population;
                        Fitness1 = CalFitnessSup(Population1.decs,V,Population1.objs);
                        Pop1get = 1;
                    end
                   MatingPool1 = TournamentSelection(2,2*Problem.N,FitnessSpea2);
                   [OffDec1,OffMask1] = OperatorNew(Problem,Dec(MatingPool1,:),Mask(MatingPool1,:),FitnessLayer,LayerMax,Delta,Local_Knowlege,Global_Knowledge,NSV,SV,theta,Fitness);
                   Offspring1 = Problem.Evaluation(OffDec1.*OffMask1); 
                   
                   MatingPool2 = TournamentSelection(2,2*Problem.N,-Fitness1);
                   [OffDec2,OffMask2] = OperatorNew(Problem,Dec(MatingPool2,:),Mask(MatingPool2,:),FitnessLayer,LayerMax,Delta,Local_Knowlege,Global_Knowledge,NSV,SV,theta,Fitness);
                   Offspring2 = Problem.Evaluation(OffDec2.*OffMask2);
                   
                   Offspring = [Offspring1,Offspring2];
                   OffDec = [OffDec1;OffDec2];
                   OffMask = [OffMask1;OffMask2];
                   [Population,Dec,Mask,FitnessSpea2,FrontNo,net,genFlag,Fitness1] = EnvironmentalSelection1([Population,Offspring],[Dec;OffDec],[Mask;OffMask],Problem.N,Problem,params,net,genFlag);                 
                end
            end                       
        end
    end
end