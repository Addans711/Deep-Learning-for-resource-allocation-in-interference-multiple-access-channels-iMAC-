% Define Cell structure
Cell.Rcell = 100;  % Set the cell radius (distance between centers of adjacent cells)
Cell.Ncell = 3;   % Set the number of cells (adjusted for illustration)
UserPerCell = 18;  % Set the number of users per cell
r_inner = 99; % Set the inner circle radius

%% locations of BSs
Rcell = Cell.Rcell;       % radius of each cell
Nbs   = Cell.Ncell;    % total number of cells in the scenario

% Initialize positions matrix for users
Cell.UserPosition = cell(Nbs, 1);

% Set positions for the centers of each cell
CellCenter = zeros(Nbs, 2);
if (Nbs > 1)
    theta = (0:Nbs-2)' * pi/3;
    CellCenter(2:end, 1:2) = sqrt(3) * Rcell * [cos(theta) sin(theta)];
end
if (Nbs > 7)
    theta = -pi/6:pi/3:5/3*pi;
    x = 3 * Rcell * cos(theta);
    y = 3 * Rcell * sin(theta);
    theta = 0:pi/3:5/3*pi;
    x = reshape([x; 2*sqrt(3)*Rcell*cos(theta)], numel([x; 2*sqrt(3)*Rcell*cos(theta)]), 1);
    y = reshape([y; 2*sqrt(3)*Rcell*sin(theta)], numel([y; 2*sqrt(3)*Rcell*sin(theta)]), 1);
    if Nbs > 19
        CellCenter(8:19, 1:2) = [x y];
    else
        CellCenter(8:Nbs, 1:2) = [x(1:(Nbs-7)) y(1:(Nbs-7))];
    end
end
if (Nbs > 19) && (Nbs < 38)
    theta = -asin(3/sqrt(21)):pi/3:5/3*pi;
    x1 = sqrt(21) * Rcell * cos(theta);
    y1 = sqrt(21) * Rcell * sin(theta);
    theta = -asin(3/2/sqrt(21)):pi/3:5/3*pi;
    x2 = sqrt(21) * Rcell * cos(theta);
    y2 = sqrt(21) * Rcell * sin(theta);
    theta = 0:pi/3:5/3*pi;
    x3 = 3 * sqrt(3) * Rcell * cos(theta);
    y3 = 3 * sqrt(3) * Rcell * sin(theta);
    x = reshape([x1;x2;x3], numel([x1;x2;x3]), 1);
    y = reshape([y1;y2;y3], numel([y1;y2;y3]), 1);
    CellCenter(20:Nbs, 1:2) = [x(1:(Nbs-19)) y(1:(Nbs-19))];
end

% Set positions for each user in each cell
for i = 1:Nbs
    for j = 1:UserPerCell
        while true
            % Generate uniformly distributed point within bounding circle
            r = r_inner + (Rcell - r_inner) * sqrt(rand);  
            theta = 2 * pi * rand;
            [x, y] = pol2cart(theta, r);
            
            % Check if the point is inside the hexagon
            if isInHexagon(x, y, Rcell)
                break;
            end
        end
        
        % Calculate user position relative to cell center
        Cell.UserPosition{i}(j, :) = [x, y] + CellCenter(i, 1:2);
    end
end

%% Plot cells layout
flag = 1; % set to 1 to enable plot
if flag
    figure
    hold on;
    for i = 1 : Nbs
        % Plot cell center (BS position)
        plot(CellCenter(i, 1), CellCenter(i, 2), 'r^', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
        
        % Plot users in the cell
        for j = 1 : UserPerCell
            x0 = Cell.UserPosition{i}(j, 1);
            y0 = Cell.UserPosition{i}(j, 2);
            plot(x0, y0, 'o', 'MarkerSize', 6, 'color', [0 0 1], 'MarkerFaceColor', 'none');
        end
        
        % Plot inner circle
        t = linspace(0, 2*pi, 100);
        x_inner = r_inner * cos(t) + CellCenter(i, 1);
        y_inner = r_inner * sin(t) + CellCenter(i, 2);
        plot(x_inner, y_inner, 'r');
        
        % Plot cell boundary
        x1 = Rcell * cos(-pi/6:pi/3:2*pi) + CellCenter(i, 1);
        y1 = Rcell * sin(-pi/6:pi/3:2*pi) + CellCenter(i, 2);
        plot(x1, y1, 'k');
        text(CellCenter(i, 1) + 15, CellCenter(i, 2) + 15, num2str(i), 'FontSize', 12);
    end
    axis equal;
    xlabel('x axis position (meter)');
    ylabel('y axis position (meter)');
end

% Function to check if a point is inside a hexagon
function inside = isInHexagon(x, y, Rcell)
    % Normalize coordinates to cell center
    x = abs(x);
    y = abs(y);
    inside = x <= Rcell && y <= Rcell * sqrt(3)/2 && x/sqrt(3) + y <= Rcell;
end

% Generate IMAC channel according to Rayleigh fading distribution
function H = generate_IMAC_channel(Cell, UserPerCell, var_noise)
    Ncells = Cell.Ncell;
    K = UserPerCell;
    H = zeros(Ncells * K, Ncells);
    for i = 1:Ncells
        for j = 1:K
            for l = 1:Ncells
                d = norm(Cell.UserPosition{i}(j, :) - CellCenter(l, :));
                L = 10^(randn * 8 / 10) * (200 / d)^(3);
                H((i-1)*K + j, l) = abs(sqrt(L/2) * (randn + 1i*randn)); % Rayleigh fading
            end
        end
    end
end
