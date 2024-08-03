function [H, X, Y, WmmseTime] = Generate_IMAC_function(num_BS, num_User, num_H, R, minR_ratio, seed, var_noise)
    % Start timer
    tic
    % Set random seed
    rng(seed);
    K = num_User * num_BS;
    X = zeros(K * num_BS, num_H);
    Y = zeros(K, num_H);
    HHH = zeros(K, K, num_H);
    temp_H = zeros(num_BS, K);
    
    % Initialize the Position field of Cell object
    Cell.Position = cell(num_BS, 1);
    for n = 1:num_BS
        theta = rand * 2 * pi;
        Cell.Position{n} = [R * cos(theta), R * sin(theta)];
    end
    
    % Iterate over each channel realization
    for i = 1:num_H
        % Generate interference channels
        HHH(:,:,i) = generate_IBC_channel(num_User, R, num_BS, minR_ratio);
        
        % Extract channels for each base station
        for l = 1:num_BS
            temp_H(l,:) = HHH((l-1)*num_User+1,:,i);
        end
        
        % Calculate sum rate using WMMSE algorithm
        Y(:,i) = WMMSE_sum_rate(ones(K, 1), HHH(:,:,i), ones(K, 1), var_noise);
        
        % Flatten the channel matrix into a vector
        X(:,i) = reshape(temp_H, K * num_BS, 1);
    end
    
    % End timer
    WmmseTime = toc;
    % Reshape the channel matrix
    H = reshape(HHH, K * K, num_H);
    
    % Save results
    save(fullfile('C:\Users\11345\Documents\project\TSP-DNN-master\TSP-DNN-master', ...
    sprintf('IMAC_%d_%d_%d_%d_%d.mat', num_BS, num_User, num_H, R, minR_ratio)), 'H', 'X', 'Y', 'WmmseTime');
end

function p_opt = WMMSE_sum_rate(p_int, H, Pmax, var_noise)
    K = length(Pmax);
    b = sqrt(p_int);
    b_pre = zeros(K, 1);
    f = zeros(K, 1);
    w = f;
    vnew = 0;
    
    % Initial WMMSE iteration
    for i = 1:K
        f(i) = H(i, i) * b(i) / ((H(i, :).^2) * (b.^2) + var_noise);
        w(i) = 1 / (1 - f(i) * b(i) * H(i, i));
        vnew = vnew + log2(w(i));
    end

    iter = 0;
    while true
        iter = iter + 1;
        vold = vnew;
        
        % Update transmit power
        for i = 1:K
            btmp = w(i) * f(i) * H(i, i) / sum(w .* (f.^2) .* (H(:, i).^2));
            b_pre(i) = b(i);
            b(i) = min(btmp, sqrt(Pmax(i))) + max(btmp, 0) - btmp;
        end

        % Update WMMSE
        vnew = 0;
        for i = 1:K
            f(i) = H(i, i) * b(i) / ((H(i, :).^2) * (b.^2) + var_noise);
            w(i) = 1 / (1 - f(i) * b(i) * H(i, i));
            vnew = vnew + log2(w(i));
        end

        if vnew - vold <= 1e-5 || iter > 500
            break;
        end
    end
    p_opt = b.^2;
end

function H_eq = generate_IBC_channel(Num_of_user_in_each_cell, cell_distance, Num_of_cell, minR_ratio)
    % Generate interference channels for Num_of_cells with Num_of_user_in_each_cell users each
    T = 1; % Number of antennas at the base station
    BaseNum = 1; % Number of base stations per cell
    UserNum = Num_of_user_in_each_cell;
    Distance = cell_distance;
    CellNum = Num_of_cell;

    % Cell environment channel
    Cell.Ncell = CellNum; % Number of coordinated cells
    Cell.Nintra = UserNum; % Number of users per cell
    Cell.NintraBase = BaseNum; % Number of base stations per cell
    Cell.Rcell = Distance * 2 / sqrt(3); % Cell radius
    
    % Initialize cell positions
    Cell.Position = cell(CellNum, 1);
    for n = 1:CellNum
        theta = rand * 2 * pi;
        Cell.Position{n} = [Distance * cos(theta), Distance * sin(theta)];
    end
    
    % Generate the corresponding channel matrix for the cell environment
    [MS, BS] = usergenerator(Cell, minR_ratio);
    [HLarge] = channelsample(BS, MS, Cell);
    H = (randn(T, BaseNum, CellNum, UserNum, CellNum) + sqrt(-1) * randn(T, BaseNum, CellNum, UserNum, CellNum)) / sqrt(2);
    for Base = 1:BaseNum
        for CellBS = 1:CellNum
            for User = 1:UserNum
                for CellMS = 1:CellNum
                    H(:, Base, CellBS, User, CellMS) = H(:, Base, CellBS, User, CellMS) * sqrt(HLarge(User, CellMS, Base, CellBS));
                end
            end
        end
    end

    total_user_Num = CellNum * UserNum;
    H_eq = zeros(total_user_Num, total_user_Num);
    k = 0;
    for CellMS = 1:CellNum
        for User = 1:UserNum
            k = k + 1;
            k_INF = 0;
            for INFCellMS = 1:CellNum
                for INFUser = 1:UserNum
                    k_INF = k_INF + 1;
                    H_eq(k, k_INF) = abs(H(:, Base, CellMS, INFUser, INFCellMS));
                end
            end
        end
    end
end

function [Hlarge] = channelsample(BS, MS, Cell)
    Ncell = Cell.Ncell; % Number of cells
    Nintra = Cell.Nintra; % Number of users per cell
    Nbs = Cell.NintraBase; % Number of base stations per cell

    % Channels between base stations and users within cells
    Hlarge = zeros(Nintra, Ncell, Nbs, Ncell);

    % Large scale fading
    for CellBS = 1:Ncell
        for CellMS = 1:Ncell
            for Base = 1:Nbs
                for User = 1:Nintra
                    d = norm(MS.Position{CellMS}(User,:) - BS.Position{CellBS}(Base,:));
                    PL = 10^(randn * 8 / 10) * (200 / d)^(3);
                    Hlarge(User, CellMS, Base, CellBS) = PL;
                end
            end
        end
    end
end

function [MS, BS] = usergenerator(Cell, minR_ratio)
    Ncell = Cell.Ncell; % Number of cells
    Nintra = Cell.Nintra; % Number of users per cell
    NintraBase = Cell.NintraBase; % Number of base stations per cell
    MS.Position = [];
    BS.Position = [];
    Nms = Nintra;
    NmsBase = NintraBase;
    Rcellmin = minR_ratio * Cell.Rcell;

    % User distribution
    MS.Position = cell(Ncell, 1);
    if Nms >= 1
        for n = 1:Ncell % Generate users for each cell
            theta = rand(Nms, 1) * 2 * pi;
            [x, y] = distrnd(Cell.Rcell, Rcellmin, theta);
            MS.Position{n} = [x + Cell.Position{n}(1), y + Cell.Position{n}(2)];
        end
    end

    % Base station distribution
    BS.Position = cell(Ncell, 1);
    if NmsBase >= 1
        for n = 1:Ncell % Generate base stations for each cell
            theta = rand(NmsBase - 1, 1) * 2 * pi;
            [x, y] = distrnd(Cell.Rcell, Rcellmin, theta);
            BS.Position{n} = [x + Cell.Position{n}(1), y + Cell.Position{n}(2)];
            BS.Position{n} = [Cell.Position{n}; BS.Position{n}];
        end
    end
end

% Generate random distances for users within a cell
function [x, y] = distrnd(Rcell, Rcellmin, theta)
    MsNum = numel(theta);
    R = Rcell - Rcellmin; % Effective cell radius

    % Generate random distances
    d = sum(rand(MsNum, 2), 2) * R; 
    d(d > R) = 2 * R - d(d > R);
    d = d + Rcellmin; 

    % Convert to Cartesian coordinates
    x = d .* cos(theta);
    y = d .* sin(theta);
end
