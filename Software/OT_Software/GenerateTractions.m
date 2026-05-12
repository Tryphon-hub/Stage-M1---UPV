function EqTractions = GenerateTractions(NumSamples,TractionFile,GenerateNew)


% Generate a random traction set in the 16 dimensional space
%Tractions = 2*(rand(16,NumSamples)-0.5);

if ~GenerateNew && exist(TractionFile,'file')
    load(TractionFile);

    if size(EqTractions,3)>=NumSamples
        EqTractions = EqTractions(:,:,1:NumSamples);
        return
    end
end

Tractions = 2*(rand(16,NumSamples)-0.5);
%disp('Tracciones fáciles')
%Tractions = 2*(rand(16,1)-0.5);
%Tractions = repmat(Tractions,1,NumSamples);
%Tractions([5,7],:) = 2*(rand(2,NumSamples)-0.5);
%Tractions([13,15],:) = Tractions([7,5],:);

% Equilibrate the traction proposal

EqualMatrix = eye(16);
ConstraintMatrix = zeros(3,16);

% Equilibrium in x direction
ConstraintMatrix(1,:) = [0 1 0 1   1 0 1 0  0 -1 0 -1  -1 0 -1 0 ];

% Equilibrium in y direction
ConstraintMatrix(2,:) = [-1 0 -1 0   0 1 0 1   1 0 1 0  0 -1 0 -1];

% Moment equilibrium
ConstraintMatrix(3,:) = [0 1 0 1     0 1 0 1   0 1 0 1  0 1 0 1] + ...
                    1/3*[1 0 -1 0   1 0 -1 0  1 0 -1 0  1 0 -1 0];

EquilibrationMatrix = [EqualMatrix ConstraintMatrix'; ConstraintMatrix zeros(3)];

EqTractions = EquilibrationMatrix\[Tractions; zeros(3,size(Tractions,2))];

EqTractions = EqTractions(1:16,:);

EqTractions = bsxfun(@times,EqTractions,1./sqrt(sum(EqTractions.^2,1)));

EqTractions = reshape(EqTractions,2,8,NumSamples);

save(TractionFile,'EqTractions')

